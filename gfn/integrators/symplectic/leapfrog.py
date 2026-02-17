
"""
Leapfrog (Kick-Drift-Kick) Symplectic Integrator.

IMPORTANT NOTES FROM AUDITORIA LOGICA (2026-02-06):

1. INTEGRATOR VARIANT:
   This is a VARIANT of the standard Stormer-Verlet / Leapfrog integrator,
   modified for systems with position-dependent friction.
   
   Standard Leapfrog:
   v(t+0.5h) = v(t) + 0.5h * a(x(t))
   x(t+h) = x(t) + h * v(t+0.5h)
   v(t+h) = v(t+0.5h) + 0.5h * a(x(t+h))
   
   Our variant (with friction mu(x)):
   v(t+0.5h) = (v(t) + 0.5h * (F - Gamma)) / (1 + 0.5h * mu(x(t)))
   x(t+h) = x(t) + h * v(t+0.5h)
   v(t+h) = (v(t+0.5h) + 0.5h * (F - Gamma)) / (1 + 0.5h * mu(x(t+h)))

2. CONSERVATION PROPERTIES:
   - In ABSENCE of friction (mu = 0), energy is conserved
   - With FRICTION, energy is DISSIPATED (as expected physically)
   - VOLUME preservation is LOST when friction != 0
     (this is correct: dissipative systems shrink phase space volume)

3. PRODUCTION FIX (2026-02-07):
   - FRICTION_SCALE reduced to 0.05 for proper symplectic behavior
   - EPSILON_STANDARD set to 1e-8 for cleaner gradients
   - Added smooth boundary wrapping for differentiable gradients

References:
- Hairer, E., Lubich, C., & Wanner, G. (2006). Geometric Numerical Integration
- Leimkuhler, B., & Reich, S. (2004). Simulating Hamiltonian Dynamics
"""

import torch
import torch.nn as nn

try:
    from gfn.cuda.ops import leapfrog_fused, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

from gfn.geometry.boundaries import apply_boundary_python, resolve_topology_id, get_boundary_features

try:
    from gfn.constants import (
        FRICTION_SCALE,
        EPSILON_STANDARD,
        EPSILON_SMOOTH,
        DEFAULT_DT,
        VELOCITY_FRICTION_SCALE,
    )
except ImportError:
    FRICTION_SCALE = 0.02
    EPSILON_STANDARD = 1e-7
    EPSILON_SMOOTH = 1e-7
    DEFAULT_DT = 0.05
    VELOCITY_FRICTION_SCALE = 0.0


class LeapfrogIntegrator(nn.Module):
    """
    Variant Stormer-Verlet integrator with position-dependent friction.
    
    For use with LowRankChristoffel and ReactiveChristoffel geometries.
    
    PRODUCTION FIX (2026-02-07):
    - Uses updated FRICTION_SCALE=0.02
    - Uses EPSILON_STANDARD=1e-7
    - Supports smooth boundary wrapping
    """
    def __init__(self, christoffel, dt=None):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt if dt is not None else DEFAULT_DT

    def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
        if force is None:
            force = torch.zeros_like(x)
            
        # Try Professional Fused CUDA Kernel
        if CUDA_AVAILABLE and x.is_cuda and not collect_christ:
            try:
                # Logic matrices
                U = getattr(self.christoffel, 'U', None)
                W = getattr(self.christoffel, 'W', None)
                V_w = getattr(self.christoffel, 'V_w', None)
                
                if U is not None and W is not None:
                    topology_id = resolve_topology_id(self.christoffel, kwargs.get('topology'))
                    Wf = kwargs.get('W_forget_stack', None)

                    bf = kwargs.get('b_forget_stack', None)
                    Wi = kwargs.get('W_input_stack', None)
                    velocity_friction_scale = kwargs.get(
                        'velocity_friction_scale',
                        getattr(self.christoffel, 'velocity_friction_scale', VELOCITY_FRICTION_SCALE),
                    )
                    if Wf is not None and Wf.dim() == 3:
                        Wf = Wf[0]
                    if bf is not None and bf.dim() == 2:
                        bf = bf[0]
                    if Wi is not None and Wi.dim() == 3:
                        Wi = Wi[0]
                    res_fused = leapfrog_fused(
                        x, v, force, U, W, self.dt, dt_scale, steps=steps, topology=topology_id,
                        Wf=Wf, bf=bf, W_input=Wi, V_w=V_w,
                        velocity_friction_scale=velocity_friction_scale,
                    )
                    if isinstance(res_fused, tuple) and len(res_fused) == 3:
                        xf, vf, _ = res_fused
                        return xf, vf
                    return res_fused
            except Exception as e:
                print(f"[GFN:WARN] CUDA leapfrog_fused failed: {e}, falling back to PyTorch")
                # Fall through to PyTorch implementation

        curr_x, curr_v = x, v
        christ_out = None
        
        # Tell Christoffel to return friction separately for implicit update
        was_separate = getattr(self.christoffel, 'return_friction_separately', False)
        self.christoffel.return_friction_separately = True
        
        # Get topology_id
        topology_id = resolve_topology_id(self.christoffel, kwargs.get('topology'))
        
        try:
            for i in range(steps):
                effective_dt = self.dt * dt_scale
                h = 0.5 * effective_dt
                
                # 1. Kick (Implicit Friction)
                # PRODUCTION: The implicit update v_new = (v + h*a) / (1 + h*mu)
                # is equivalent to applying friction implicitly
                res = self.christoffel(curr_v, curr_x, force=force, **kwargs)
                if isinstance(res, tuple):
                    gamma, mu = res
                else:
                    gamma, mu = res, 0.0  # Fallback
                
                if i == 0:
                    # Capture Gamma for Noether
                    christ_out = gamma

                # Get friction from gate if needed
                Wf = kwargs.get('W_forget_stack', None)
                bf = kwargs.get('b_forget_stack', None)
                Wi = kwargs.get('W_input_stack', None)
                velocity_friction_scale = kwargs.get(
                    'velocity_friction_scale',
                    getattr(self.christoffel, 'velocity_friction_scale', VELOCITY_FRICTION_SCALE),
                )
                
                if Wf is not None and bf is not None:
                    if Wf.dim() == 3:
                        Wf = Wf[0]
                    if bf.dim() == 2:
                        bf = bf[0]
                    
                    # AUDIT FIX: Centralized boundary features
                    feat = get_boundary_features(curr_x, topology_id)
                    
                    gate = torch.matmul(feat, Wf.t()) + bf
                    if Wi is not None:
                        if Wi.dim() == 3:
                            Wi = Wi[0]
                        gate = gate + torch.matmul(force, Wi.t())
                    mu = torch.sigmoid(gate) * FRICTION_SCALE
                    if velocity_friction_scale > 0:
                        v_norm = torch.norm(curr_v, dim=-1, keepdim=True)
                        v_norm = v_norm / (curr_v.shape[-1] ** 0.5 + EPSILON_SMOOTH)
                        mu = mu * (1.0 + velocity_friction_scale * v_norm)

                # PRODUCTION: Implicit Update with EPSILON_STANDARD=1e-7
                # v_half = (curr_v + h*(F - Gamma)) / (1 + h*mu + eps)
                v_half = (curr_v + h * (force - gamma)) / (1.0 + h * mu + EPSILON_STANDARD)
                
                # 2. Drift (full step position)
                curr_x = curr_x + effective_dt * v_half
                
                # Apply Boundary (Torus) - PRODUCTION: Use smooth wrapping
                curr_x = apply_boundary_python(curr_x, topology_id)
                
                # 3. Kick (half step velocity at new pos)
                res_half = self.christoffel(v_half, curr_x, force=force, **kwargs)
                if isinstance(res_half, tuple):
                    gamma_half, mu_half = res_half
                else:
                    gamma_half, mu_half = res_half, 0.0
                
                if Wf is not None and bf is not None:
                    if Wf.dim() == 3:
                        Wf = Wf[0]
                    if bf.dim() == 2:
                        bf = bf[0]
                    
                    # AUDIT FIX: Centralized boundary features
                    feat = get_boundary_features(curr_x, topology_id)
                    
                    gate = torch.matmul(feat, Wf.t()) + bf
                    if Wi is not None:
                        if Wi.dim() == 3:
                            Wi = Wi[0]
                        gate = gate + torch.matmul(force, Wi.t())
                    mu_half = torch.sigmoid(gate) * FRICTION_SCALE
                    if velocity_friction_scale > 0:
                        v_norm = torch.norm(v_half, dim=-1, keepdim=True)
                        v_norm = v_norm / (v_half.shape[-1] ** 0.5 + EPSILON_SMOOTH)
                        mu_half = mu_half * (1.0 + velocity_friction_scale * v_norm)
                    
                # PRODUCTION: Implicit Update with EPSILON_STANDARD=1e-7
                curr_v = (v_half + h * (force - gamma_half)) / (1.0 + h * mu_half + EPSILON_STANDARD)
        finally:
            self.christoffel.return_friction_separately = was_separate
        
        if collect_christ:
            return curr_x, curr_v, christ_out
        return curr_x, curr_v
