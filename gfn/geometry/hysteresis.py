import torch
import torch.nn as nn

class HysteresisChristoffel(nn.Module):
    """
    Hysteresis Semantic Memory (Paper 19).
    
    Implements trajectory-dependent Christoffel symbols:
        Γ_hyst = Γ_base + δΓ(h)
    
    where h is the accumulated "Hysteresis State" representing the 
    history of the agent's path on the manifold.
    
    This turns the manifold into a "Memory Manifold" where past trajectories
    deform the local curvature, creating "social paths" or "semantic habits".
    """
    def __init__(self, base_christoffel, dim, rank=16):
        super().__init__()
        self.base_christoffel = base_christoffel
        self.dim = dim
        self.rank = rank
        
        # Deformation map: h -> δΓ
        # We use a low-rank approximation for the deformation to be efficient
        self.U_hyst = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.W_hyst = nn.Parameter(torch.randn(dim, rank) * 0.01)
        
    def forward(self, v, x=None, memory_state=None, **kwargs):
        """
        Compute symbols with hysteresis deformation.
        
        Args:
            v: Velocity [Batch, Dim]
            x: Position [Batch, Dim]
            memory_state: Accumulated hysteresis state h [Batch, Dim]
        """
        # 1. Base Curvature
        gamma = self.base_christoffel(v, x, **kwargs)
        
        # 2. Add Hysteresis Deformation: δΓ(h, v)
        # We model this as a learned force field modulated by memory
        if memory_state is not None:
            # δΓ = (U_hyst @ W_hyst.T) @ h
            # This is a linear memory-driven force correction
            # To be strictly trajectory dependent, we could make it bilinear with v
            
            # Simple version: Memory acts as a "Ghost Force" that deforms symbols
            # δΓ^k = (W_hyst @ (U_hyst.T @ h))
            delta = torch.matmul(memory_state, self.U_hyst) # [Batch, Rank]
            delta = torch.matmul(delta, self.W_hyst.t())    # [Batch, Dim]
            
            # Bilinear interaction with velocity (Paper 19 suggests dependency on path)
            # δΓ(v, v) -> we want the correction to scale with movement
            # Scaling it by ||v|| ensures it doesn't manifest at zero velocity
            v_norm = torch.norm(v, dim=-1, keepdim=True)
            delta = delta * v_norm
            
            gamma = gamma + delta
            
        return gamma
