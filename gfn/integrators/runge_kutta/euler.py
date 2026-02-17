
"""
Euler Integrator (1st Order).
The simplest explicit integrator. 
Useful as a baseline to demonstrate the instability of low-order non-symplectic methods.
"""
import torch
import torch.nn as nn

try:
    from gfn.cuda.ops import euler_fused, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

from gfn.geometry.boundaries import apply_boundary_python, resolve_topology_id

class EulerIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt

    def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
        # Try Professional Fused CUDA Kernel
        if CUDA_AVAILABLE and x.is_cuda and not collect_christ:
            try:
                # We need U, W from Christoffel
                U = getattr(self.christoffel, 'U', None)
                W = getattr(self.christoffel, 'W', None)
                if U is not None and W is not None:
                    topology = resolve_topology_id(self.christoffel, kwargs.get('topology'))
                    
                    R = getattr(self.christoffel, 'R', 2.0)

                    r = getattr(self.christoffel, 'r', 1.0)
                    
                    return euler_fused(x, v, force, U, W, self.dt, dt_scale, steps=steps, topology=topology, R=R, r=r)
            except Exception as e:
                print(f"[GFN:WARN] CUDA euler_fused failed: {e}, falling back to PyTorch")
                # Fall through to PyTorch implementation

        curr_x, curr_v = x, v
        christ_out = None
        
        for i in range(steps):
            dt = self.dt * dt_scale
            
            c_out = self.christoffel(curr_v, curr_x, force=force, **kwargs)
            if i == 0:
                christ_out = c_out
                
            acc = -c_out
            if force is not None:
                acc = acc + force
                
            curr_x = curr_x + dt * curr_v
            curr_v = curr_v + dt * acc
            
            # Apply Boundary (Torus)
            topo_id = resolve_topology_id(self.christoffel, kwargs.get('topology'))
            curr_x = apply_boundary_python(curr_x, topo_id)
        
        if collect_christ:
            return curr_x, curr_v, christ_out
        return curr_x, curr_v
