
"""
Heun's Method (Improved Euler / RK2).
2nd order accuracy with only 2 evaluations per step.
Great balance between accuracy and speed.
"""
import torch
import torch.nn as nn

try:
    from gfn.cuda.ops import heun_fused, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

from gfn.geometry.boundaries import apply_boundary_python, resolve_topology_id

class HeunIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt
        
    def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
        # Try Professional Fused CUDA Kernel
        if CUDA_AVAILABLE and x.is_cuda and not collect_christ:
            try:
                U = getattr(self.christoffel, 'U', None)
                W = getattr(self.christoffel, 'W', None)
                if U is not None and W is not None:
                    topology = resolve_topology_id(self.christoffel, kwargs.get('topology'))
                    
                    R = getattr(self.christoffel, 'R', 2.0)

                    r = getattr(self.christoffel, 'r', 1.0)
                    
                    # Friction gate parameters
                    fg = getattr(self.christoffel, 'forget_gate', None)
                    W_forget = fg.weight if fg is not None else torch.empty(0, device=x.device)
                    b_forget = fg.bias if (fg is not None and fg.bias is not None) else torch.empty(0, device=x.device)
                    
                    ig = getattr(self.christoffel, 'input_gate', None)
                    W_input = ig.weight if ig is not None else torch.empty(0, device=x.device)
                    
                    # Active inference parameters
                    plasticity = getattr(self.christoffel, 'plasticity', 0.0)
                    sing_thresh = getattr(self.christoffel, 'semantic_certainty_threshold', 1.0)
                    sing_strength = getattr(self.christoffel, 'curvature_amplification_factor', 1.0)
                    
                    return heun_fused(
                        x, v, force, U, W, self.dt, dt_scale, steps=steps, topology=topology,
                        W_forget=W_forget, b_forget=b_forget,
                        plasticity=plasticity, sing_thresh=sing_thresh, sing_strength=sing_strength,
                        R=R, r=r, W_input=W_input
                    )
            except Exception:
                pass

        curr_x, curr_v = x, v
        christ_out = None
        
        for _ in range(steps):
            dt = self.dt * dt_scale
            
            def dynamics(current_x, current_v, is_first=False):
                # LEVEL 25: CLUTCH CONNECTION
                nonlocal christ_out
                c_out = self.christoffel(current_v, current_x, force=force, **kwargs)
                if is_first and christ_out is None:
                    christ_out = c_out
                
                acc = -c_out
                if force is not None:
                    acc = acc + force
                return acc
                
            # Determine Topology
            topo_id = resolve_topology_id(self.christoffel, kwargs.get('topology'))

            # k1
            dx1 = curr_v
            dv1 = dynamics(curr_x, curr_v, is_first=True)
            
            # Predictor step (Euler)
            v_pred = curr_v + dt * dv1
            x_pred = apply_boundary_python(curr_x + dt * dx1, topo_id)
            
            # k2 (using predicted velocity AND position)
            dx2 = v_pred
            dv2 = dynamics(x_pred, v_pred)
            
            # Corrector step
            curr_x = curr_x + (dt / 2.0) * (dx1 + dx2)
            curr_v = curr_v + (dt / 2.0) * (dv1 + dv2)
            
            # Apply Boundary (Torus)
            curr_x = apply_boundary_python(curr_x, topo_id)
        
        if collect_christ:
            return curr_x, curr_v, christ_out
        return curr_x, curr_v
