import torch
import torch.nn as nn

try:
    from gfn.geometry.boundaries import apply_boundary_python
except ImportError:
    def apply_boundary_python(x, tid): return x

class PEFRLIntegrator(nn.Module):
    """
    PEFRL (Position Extended Forest-Ruth Like) Integrator (Paper 11).
    A 4th-order symplectic integrator optimized by Omelyan et al. (2002).
    
    Provides approx 100x smaller energy drift than Forest-Ruth/Yoshida 
    by minimizing the error constant for Hamiltonian systems.
    
    Algorithm: 4 Force evaluations per step, 5 Position drifts.
    """
    def __init__(self, christoffel, dt=0.1):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt
        
        # Exact PEFRL Coefficients (Omelyan 2002)
        self.xi = 0.1786178958448091
        self.lam = -0.2123418310626054
        self.chi = -0.06626458266981849
        
    def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
        dt = self.dt * dt_scale
        
        # Constants for staging
        XI = self.xi
        LAM = self.lam
        CHI = self.chi
        
        K1 = (1.0 - 2.0 * LAM) / 2.0
        D1 = 1.0 - 2.0 * (CHI + XI)
        
        # Topology selection
        topo_id = getattr(self.christoffel, 'topology_id', 0)
        if topo_id == 0 and hasattr(self.christoffel, 'is_torus') and self.christoffel.is_torus:
            topo_id = 1
            
        curr_x, curr_v = x, v
        christ_out = None
        
        for _ in range(steps):
            # Define internal acceleration evaluator
            def get_acc(tx, tv, is_first=False):
                nonlocal christ_out
                # Acceleration = -Gamma(v,v) + F_ext
                c_out = self.christoffel(tv, tx, force=force, **kwargs)
                if is_first and christ_out is None:
                    christ_out = c_out
                
                a = -c_out
                if force is not None:
                    a = a + force
                return a

            # 1. Drift xi
            curr_x = apply_boundary_python(curr_x + XI * dt * curr_v, topo_id)
            
            # 2. Kick K1 (Eval 1)
            curr_v = curr_v + K1 * dt * get_acc(curr_x, curr_v, is_first=True)
            
            # 3. Drift chi
            curr_x = apply_boundary_python(curr_x + CHI * dt * curr_v, topo_id)
            
            # 4. Kick lambda (Eval 2)
            curr_v = curr_v + LAM * dt * get_acc(curr_x, curr_v)
            
            # 5. Drift D1
            curr_x = apply_boundary_python(curr_x + D1 * dt * curr_v, topo_id)
            
            # 6. Kick lambda (Eval 3)
            curr_v = curr_v + LAM * dt * get_acc(curr_x, curr_v)
            
            # 7. Drift chi
            curr_x = apply_boundary_python(curr_x + CHI * dt * curr_v, topo_id)
            
            # 8. Kick K1 (Eval 4)
            curr_v = curr_v + K1 * dt * get_acc(curr_x, curr_v)
            
            # 9. Final Drift xi
            curr_x = apply_boundary_python(curr_x + XI * dt * curr_v, topo_id)
            
        if collect_christ:
            return curr_x, curr_v, christ_out
        return curr_x, curr_v
