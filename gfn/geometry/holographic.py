import torch
import torch.nn as nn
import torch.nn.functional as F

class AdSCFTChristoffel(nn.Module):
    """
    AdS/CFT Holographic Extensions (Paper 18).
    
    This module implements a conformal manifold geometry inspired by the 
    Bulk-Boundary correspondence. It lifts the boundary state 'x' to a 
    bulk coordinate '(x, z)' where 'z' is the radial (holographic) dimension.
    
    The metric is assumed to be conformally flat in the bulk:
        g_ij = (1 / z(x)^2) * delta_ij
    
    The resulting Christoffel symbols are:
        Gamma^k_ij = -1/z * (z_j delta^k_i + z_i delta^k_j - z_k delta_ij)
    """
    def __init__(self, base_christoffel, z_min=0.1, z_max=10.0):
        super().__init__()
        self.base_christoffel = base_christoffel
        self.dim = base_christoffel.dim
        
        # Radial Field Network: Map x to radial distance z
        # We want z > 0, so we use softplus or exp
        self.radial_net = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.SiLU(),
            nn.Linear(self.dim // 2, 1),
            nn.Softplus()
        )
        self.z_min = z_min
        self.z_max = z_max

    def get_z_and_grad(self, x):
        """
        Computes the radial coordinate z and its gradient w.r.t x.
        """
        x_req = x.detach().requires_grad_(True)
        z = self.radial_net(x_req) + self.z_min
        z = torch.clamp(z, max=self.z_max)
        
        # Compute grad(z)
        grad_z = torch.autograd.grad(
            z.sum(), x_req, 
            create_graph=self.training, 
            retain_graph=True
        )[0]
        
        return z, grad_z

    def forward(self, v, x, **kwargs):
        # 1. Base Geometry (Optional)
        # In a pure AdS/CFT setup, the base might be Euclidean, 
        # but we allow wrapping to combine effects.
        gamma_base = self.base_christoffel(v, x, **kwargs)
        
        # 2. Holographic Contribution
        # Gamma^k = Gamma^k_ij v^i v^j
        # Gamma^k = -1/z * ( (z_j v^j) v^k + (z_i v^i) v^k - z_k (v . v) )
        # Gamma^k = -1/z * ( 2 * (grad_z . v) * v^k - (v . v) * grad_z_k )
        
        z, grad_z = self.get_z_and_grad(x)
        
        v_dot_gradz = (v * grad_z).sum(dim=-1, keepdim=True) # [B, 1]
        v_sq = (v * v).sum(dim=-1, keepdim=True) # [B, 1]
        
        gamma_ads = -(1.0 / z) * (2.0 * v_dot_gradz * v - v_sq * grad_z)
        
        return gamma_base + gamma_ads
