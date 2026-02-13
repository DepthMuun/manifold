import torch
import torch.nn as nn

class AdaptiveIntegrator(nn.Module):
    """
    Adaptive Manifold Resolver (AMR) - Paper 07.
    
    Wraps a base integrator (e.g., Heun, RK4) and applies adaptive time-stepping.
    
    Mechanism:
    - Step Doubling (Richardson Extrapolation):
        1. Take one step of size h (x1)
        2. Take two steps of size h/2 (x2)
        3. Error estimate E = ||x1 - x2|| / (2^p - 1)
    - If E > tolerance, recurse (subdivide).
    - If E < tolerance, accept x2 (higher precision).
    
    This ensures that high-curvature regions (where error is high) automatically 
    receive more computational steps (higher temporal resolution).
    """
    def __init__(self, base_integrator, tolerance=1e-3, max_depth=3):
        super().__init__()
        self.base_integrator = base_integrator
        self.tolerance = tolerance
        self.max_depth = max_depth
        
        # Determine order p of base integrator for error scaling
        # Heun/Verlet are 2nd order (p=2) -> scale 1/3
        # RK4 is 4th order (p=4) -> scale 1/15
        integrator_name = base_integrator.__class__.__name__.lower()
        if 'rk4' in integrator_name or 'forest' in integrator_name:
            self.error_scale = 1.0 / 15.0
        else:
            self.error_scale = 1.0 / 3.0 # Conservatively assume 2nd order

    def forward(self, x, v, force=None, dt_scale=1.0, depth=0, **kwargs):
        """
        Recursive adaptive step.
        """
        dt = self.base_integrator.dt * dt_scale
        
        # 1. Full Step (h)
        x1, v1 = self.base_integrator(x, v, force=force, dt_scale=dt_scale, steps=1, **kwargs)
        
        # 2. Two Half Steps (h/2)
        # Note: We pass dt_scale/2 to the integrator for the half steps
        x_mid, v_mid = self.base_integrator(x, v, force=force, dt_scale=dt_scale * 0.5, steps=1, **kwargs)
        x2, v2 = self.base_integrator(x_mid, v_mid, force=force, dt_scale=dt_scale * 0.5, steps=1, **kwargs)
        
        # 3. Error Estimate (Local Truncation Error)
        # We check error in Position (x) primarily
        error = torch.norm(x1 - x2, dim=-1).max() * self.error_scale
        
        # 4. AMR Logic
        if error > self.tolerance and depth < self.max_depth:
            # Recursively refine the two half-steps
            # We treat the first half-step as a new problem, then the second
            
            # Sub-problem 1: t -> t + h/2
            x_half, v_half = self.forward(x, v, force, dt_scale * 0.5, depth + 1, **kwargs)
            
            # Sub-problem 2: t + h/2 -> t + h
            # Note: We use the output of sub-problem 1 as input
            x_final, v_final = self.forward(x_half, v_half, force, dt_scale * 0.5, depth + 1, **kwargs)
            
            return x_final, v_final
            
        else:
            # Accept the more precise solution (x2) - Local Extrapolation
            # Optionally we could perform Richardson boost: x_boost = x2 + (x2 - x1) / (2^p - 1)
            # But plain x2 is safer for symplectic conservation stability.
            return x2, v2
