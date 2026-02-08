import torch
import torch.nn as nn
from ..constants import FRICTION_SCALE, EPSILON_STRONG
from .lowrank import LowRankChristoffel

class HyperChristoffel(LowRankChristoffel):
    """
    Hyper-Christoffel: Context-Dependent Geometry.
    
    Architecture:
    Gamma(v, v | x) = W(x) * (U(x)^T v)^2
    
    Efficient Implementation (Gated Modulation):
    U(x) = U_static * diag(Gate_u(x))
    W(x) = W_static * diag(Gate_w(x))
    
    Where Gate(x) outputs a [rank] vector in [0, 2], scaling the importance 
    of each geometric basis vector dynamically.
    """
    def __init__(self, dim, rank=16, physics_config=None):
        super().__init__(dim, rank, physics_config)
        
        # HyperNetworks: State x -> Modulation Gates [rank]
        # Light-weight: just a linear projection + activation
        self.gate_u = nn.Linear(dim, rank)
        self.gate_w = nn.Linear(dim, rank)
        
        # Initialize gates to be near identity (output ~1.0)
        # Sigmoid(0) = 0.5 -> * 2 = 1.0
        nn.init.zeros_(self.gate_u.weight)
        nn.init.zeros_(self.gate_u.bias)
        nn.init.zeros_(self.gate_w.weight)
        nn.init.zeros_(self.gate_w.bias)
        
    def forward(self, v, x=None, force=None, **kwargs):
        if x is None:
            return super().forward(v, None, force=force, **kwargs)
            
        # 1. Compute Context Gates
        # Range: [0, 2] - allowing to silence (0) or amplify (2) specific basis vectors
        g_u = torch.sigmoid(self.gate_u(x)) * 2.0 # [batch, rank]
        g_w = torch.sigmoid(self.gate_w(x)) * 2.0 # [batch, rank]
        
        # 2. Modulate Static Basis
        # U: [dim, rank]
        # g_u: [batch, rank]
        # Effective U: U * g_u (broadcast) -> effectively specific U for each batch item!
        # U_dynamic = U (1, dim, rank) * g_u (batch, 1, rank)
        
        # PyTorch optimization: Don't materialize full U_dynamic [batch, dim, rank] (too big)
        # Instead, modulate projection:
        # proj = v @ U -> [batch, rank]
        # proj_dynamic = proj * g_u
        
        # Weights U, W are [dim, rank]
        # v: [batch, dim]
        
        # a) Project momentum onto static basis
        proj_static = torch.matmul(v, self.U) # [batch, rank]
        
        # b) Modulate projection by Context (Hyper-U)
        proj_dynamic = proj_static * g_u # [batch, rank]
        
        # c) Soft-Saturation (to prevent energy explosion)
        # Instead of pure quadratic sq_dynamic = proj_dynamic * proj_dynamic
        sq_dynamic = (proj_dynamic * proj_dynamic) / (1.0 + torch.abs(proj_dynamic))
        
        # d) Modulate Reconstruction by Context (Hyper-W)
        sq_modulated = sq_dynamic * g_w # [batch, rank]
        
        # e) Reconstruct force
        # out = sq_modulated @ W.T
        gamma = torch.matmul(sq_modulated, self.W.t()) # [batch, dim]
        
        if self.is_torus:
            x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        else:
            x_in = x

        Wf = kwargs.get('W_forget_stack', None)
        Wi = kwargs.get('W_input_stack', None)
        bf = kwargs.get('b_forget_stack', None)

        if Wf is not None and bf is not None:
            if Wf.dim() == 3:
                Wf = Wf[0]
            if Wi is not None and Wi.dim() == 3:
                Wi = Wi[0]
            if bf.dim() == 2:
                bf = bf[0]
            gate_activ = torch.matmul(x_in, Wf.t()) + bf
            if Wi is not None and force is not None:
                gate_activ = gate_activ + torch.matmul(force, Wi.t())
        else:
            gate_activ = self.forget_gate(x_in)
            if force is not None:
                gate_activ = gate_activ + self.input_gate(force)

        mu_base = torch.sigmoid(gate_activ) * FRICTION_SCALE
        velocity_magnitude = torch.norm(v, dim=-1, keepdim=True)
        velocity_magnitude = velocity_magnitude / (self.dim ** 0.5 + EPSILON_STRONG)
        mu = mu_base * (1.0 + self.velocity_friction_scale * velocity_magnitude)

        if getattr(self, 'return_friction_separately', False):
            gamma_normalized = self._normalize_christoffel_structure(gamma)
            return gamma_normalized, mu

        gamma = gamma + mu * v
        gamma = self._normalize_christoffel_structure(gamma)
        return self.clamp_val * torch.tanh(gamma / self.clamp_val)
