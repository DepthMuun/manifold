
import torch
import torch.nn as nn
import numpy as np
from gfn.geometry.lowrank import LowRankChristoffel

class GaugeChristoffel(nn.Module):
    r"""
    Combines Riemannian curvature with gauge-theoretic corrections.
    
    This module implements the Gauge Theory framework for semantic consistency
    described in Paper 06. It models semantic transformations as gauge symmetries
    and uses a learned connection A_mu to defining parallel transport.
    
    The total connection is given by:
    Gamma^g_uv = Gamma^R_uv + g * (D_u v - d_u v)
    
    where:
    - Gamma^R is the base Riemannian Christoffel symbol (usually Global or LowRank)
    - g is a learnable coupling constant
    - D_u v is the covariant derivative
    - d_u v is the partial derivative
    """
    def __init__(self, dim, gauge_dim, group='U1', rank=32, physics_config=None):
        super().__init__()
        self.dim = dim
        self.gauge_dim = gauge_dim
        self.group = group
        self.config = physics_config or {}
        
        # Base Riemannian Christoffel symbols
        # We reuse the existing LowRank implementation for the base geometry
        self.base_christoffel = LowRankChristoffel(dim, rank, physics_config)
        
        # Gauge connection network: x -> A_mu(x)
        # Maps hidden state x to the Lie algebra element A_mu
        # For U(1), this is a vector of dim -> gauge_dim (phase shifts)
        self.A_net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, gauge_dim * dim) # Output shape: [batch, dim * gauge_dim]
        )
        
        # Learnable gauge coupling strength
        self.gauge_coupling = nn.Parameter(torch.tensor(0.1))
        
    def compute_connection(self, x):
        """
        Computes the gauge connection field A_mu(x).
        
        Args:
            x (Tensor): [batch, dim] hidden state
            
        Returns:
            A (Tensor): [batch, dim, gauge_dim] connection coefficients
        """
        flat_A = self.A_net(x)
        return flat_A.view(-1, self.dim, self.gauge_dim)

    def compute_field_strength(self, x):
        """
        Computes the field strength tensor F_mn = d_m A_n - d_n A_m + [A_m, A_n]
        
        Args:
            x (Tensor): [batch, dim] hidden state
            
        Returns:
            F (Tensor): [batch, dim, dim, gauge_dim] curvature tensor
        """
        # We use autograd to compute the Jacobian of A with respect to x
        # Note: This can be expensive for large batches/dims
        
        # Wrapper for easy jacobian computation
        def get_A(x_in):
            return self.compute_connection(x_in)
            
        # Compute Jacobian: [batch, output_dim, input_dim]
        # output_dim is dim * gauge_dim, input_dim is dim
        # Reshaped to: [batch, dim, gauge_dim, dim]
        # Note: torch.autograd.functional.jacobian computes full Jacobian [batch, out, batch, in]
        # which is huge. For batch-wise independent, we create a wrapper or loop.
        # For efficiency here, we only support single item or small batches via loop
        
        batch_size = x.size(0)
        field_strengths = []
        
        for i in range(batch_size):
            # x[i] -> [dim], A(x[i]) -> [dim, gauge_dim]
            jac = torch.autograd.functional.jacobian(
                lambda x_i: self.compute_connection(x_i.unsqueeze(0)).squeeze(0),
                x[i],
                create_graph=True
            )
            # JAC shape: [dim (mu), gauge_dim (a), dim (nu)] = d_nu A^a_mu
            # We want F^a_munu = d_mu A^a_nu - d_nu A^a_mu (for Abelian U1)
            
            # Rearrange to [dim, dim, gauge_dim]
            d_A = jac.permute(2, 0, 1) # [nu, mu, a]
            
            if self.group == 'U1':
                # Abelian: F_munu = d_mu A_nu - d_nu A_mu
                F = d_A.permute(1, 0, 2) - d_A # [mu, nu, a] - [nu, mu, a]
            else:
                raise NotImplementedError("Non-Abelian groups require commutator term")
                
            field_strengths.append(F)
            
        return torch.stack(field_strengths)

    def parallel_transport(self, v, x):
        """
        Parallel transport velocity v along the gauge connection.
        
        Args:
            v (Tensor): [batch, dim] velocity vector
            x (Tensor): [batch, dim] position vector
            
        Returns:
            v_transported (Tensor): [batch, dim]
        """
        A = self.compute_connection(x) # [batch, dim, gauge_dim]
        
        if self.group == 'U1':
            # For U(1), the connection acts as a phase rotation
            # A_mu corresponds to the phase shift per unit length in direction mu
            # The total phase shift for vector v is A_mu * v^mu
            
            # Project v onto connection to get total phase shift
            # [batch, 1, dim] @ [batch, dim, gauge_dim] -> [batch, 1, gauge_dim]
            phase_shift = torch.bmm(v.unsqueeze(1), A).squeeze(1)
            
            # In complex input, this would be exp(i * phase).
            # For real networks, we can model this as a rotation in 2D subspaces
            # or simply as a modulation.
            # Paper 06 suggests: return v * phase.real for real valued networks
            # treating the gauge dimension as latent phase factors.
            
            # Simple modulation model for real-valued networks:
            modulation = torch.cos(phase_shift.mean(dim=-1, keepdim=True))
            return v * modulation
        else:
            raise NotImplementedError("Only U(1) gauge is currently implemented")

    def forward(self, v, x=None, force=None, **kwargs):
        """
        Computes the total Gauge-Corrected Christoffel symbol.
        
        Gamma^g = Gamma^base + g * (D v - d v)
        
        Since D v - d v = A v (approx), the correction represents the
        force exerted by the gauge field.
        """
        # Base Riemannian term
        gamma_base = self.base_christoffel(v, x, force, **kwargs)
        
        if x is None:
            return gamma_base
            
        # Gauge correction term
        # Transport v slightly and see difference
        v_transported = self.parallel_transport(v, x)
        
        # The correction is proportional to the difference induced by transport
        gamma_gauge = self.gauge_coupling * (v_transported - v)
        
        return gamma_base + gamma_gauge
        
def gauge_invariant_loss(logits, logits_transformed):
    """
    Computes the gauge invariance penalty.
    L_gauge = MSE(f(x), f(g*x))
    """
    return torch.nn.functional.mse_loss(logits, logits_transformed)
