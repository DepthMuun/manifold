import torch
import torch.nn as nn

class GeometricNoise(nn.Module):
    """
    Stochastic Differential Geometry (Paper 16).
    
    Implements Brownian motion on Riemannian Manifolds.
    To ensure the distribution remains on the manifold (Gibbs stationary state),
    we must add a Geometric Drift correction term.
    
    Formula:
        dv^i = ... + sigma * dW^i + (sigma^2 / 2) * Gamma^i_{jk} * g^{jk}
    
    This module computes both the random noise vector and the drift correction.
    """
    def __init__(self, dim, sigma=0.01):
        super().__init__()
        self.dim = dim
        self.log_sigma = nn.Parameter(torch.tensor(sigma).log())
        
    def get_sigma(self):
        return self.log_sigma.exp()

    def forward(self, x, v, christoffel_fn, dt=0.1, **kwargs):
        """
        Computes the stochastic impulse for a time step dt.
        
        Args:
            x: Current position [batch, dim]
            v: Current velocity [batch, dim]
            christoffel_fn: Function that returns Gamma [batch, dim] or (Gamma, mu)
            dt: Time step
            
        Returns:
            impulse: [batch, dim] to be added to velocity.
        """
        sigma = self.get_sigma()
        batch_size = x.shape[0]
        device = x.device
        
        # 1. Brownian Noise: sigma * sqrt(dt) * N(0, 1)
        # We use sqrt(dt) because variance scales with dt
        noise = sigma * torch.sqrt(torch.tensor(dt, device=device)) * torch.randn_like(v)
        
        # 2. Geometric Drift Correction (Paper 16)
        # Term: (sigma^2 / 2) * Gamma^i_{jk} * g^{jk} * dt
        # In our Low-Rank implementation, Gamma is already the contraction Gamma^i_{jk} v^j v^k.
        # However, the drift requires the contraction with the inverse metric g^{jk}.
        
        # In our framework, we don't always materialize g^{jk}.
        # Approximation: Since we are in a Riemannian GFN where the metric is derived from 
        # the Christoffel symbols via accumulation, we can approximate the drift 
        # using the Curvature itself.
        
        # More accurately, if we assume g^{jk} is close to Euclidean (identity) initially:
        # Drift_i approx (sigma^2 / 2) * sum_j Gamma^i_{jj}
        
        # We can estimate sum_j Gamma^i_{jj} by probing the christoffel_fn 
        # with basis vectors, or by using the fact that for LowRank, 
        # Gamma^i_{jk} v^j v^k = sum_r (v^T U_r)^2 W_r,i
        # Then Gamma^i_{jk} g^{jk} = sum_r (tr(U_r U_r^T g^{-1})) W_r,i
        
        # Since we want to be "surgical" and efficient:
        # We will use the velocity-dependent Gamma as a proxy for the mean curvature spread.
        # Or better: We bypass full g^{-1} by using the trace of the learned basis.
        
        # If the christoffel_fn is a LowRankChristoffel, it has .U and .W
        # Let's try to find them.
        base_geo = christoffel_fn
        # Handle wrappers (Thermo, Confusion, etc)
        while hasattr(base_geo, 'base_christoffel'):
            base_geo = base_geo.base_christoffel
            
        if hasattr(base_geo, 'U') and hasattr(base_geo, 'W'):
            # Proper Low-Rank Drift
            # U, W: [dim, rank]
            # Drift ~ (sigma^2 / 2) * sum_r ||U_r||^2 * W_r
            U_sq_norm = (base_geo.U ** 2).sum(dim=0) # [rank]
            drift = torch.matmul(U_sq_norm, base_geo.W.t()) # [dim]
            drift = (sigma**2 / 2.0) * drift * dt
        else:
            # Fallback: No drift correction or zero if not low-rank
            drift = torch.zeros_like(v)
            
        return noise + drift
