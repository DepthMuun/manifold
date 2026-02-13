import torch
import torch.nn as nn

class StochasticIntegrator(nn.Module):
    """
    Wrapper for any integrator to add Geometric Noise (Paper 16).
    
    This implements the Langevin update:
        1. Deterministic Geodesic Step (via base_integrator)
        2. Stochastic Impulse (via geometric_noise)
    """
    def __init__(self, base_integrator, geometric_noise):
        super().__init__()
        self.base_integrator = base_integrator
        self.geometric_noise = geometric_noise
        
        # Expose base integrator properties
        self.dt = base_integrator.dt
        self.christoffel = base_integrator.christoffel
        
    def forward(self, x, v, force=None, dt_scale=1.0, **kwargs):
        # 1. Deterministic Step
        # Note: We take 1 step or as many as configured
        x_next, v_next = self.base_integrator(x, v, force=force, dt_scale=dt_scale, **kwargs)
        
        # 2. Stochastic Step
        # Compute dt for this specific step
        dt = self.dt * dt_scale
        
        # Get impulse (Noise + Drift)
        impulse = self.geometric_noise(x, v, self.christoffel, dt=dt, **kwargs)
        
        # Apply to velocity
        v_stochastic = v_next + impulse
        
        return x_next, v_stochastic
