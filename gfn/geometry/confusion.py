import torch
import torch.nn as nn
from ..constants import CURVATURE_CLAMP

class ConfusionChristoffel(nn.Module):
    """
    Physicality of Confusion (Paper 08).
    
    Implements a metric deformation based on the "Confusion" of the agent.
    Confusion is proxied by the magnitude of the driving force F (which represents
    the gradient of the negative log-likelihood or error signal).
    
    Theory:
    When the agent is "confused" (high error/force), the manifold should "expand" 
    locally to increase the distance to the goal, forcing the agent to spend more 
    "time" (integration steps) processing that region.
    
    Metric Deformation:
        g_new = g_base + lambda * (F @ F.T)
        
    Christoffel Effect:
        The effective Christoffel symbols are scaled by (1 + sensitivity * ||F||^2).
    """
    def __init__(self, base_christoffel, sensitivity=1.0):
        super().__init__()
        self.base_christoffel = base_christoffel
        self.sensitivity = sensitivity
        self.dim = getattr(base_christoffel, 'dim', None)
        
        # Expose attributes of base geometry
        if hasattr(base_christoffel, 'is_torus'):
            self.is_torus = base_christoffel.is_torus
        if hasattr(base_christoffel, 'topology_id'):
            self.topology_id = base_christoffel.topology_id
            
    def forward(self, v, x=None, force=None, **kwargs):
        # 1. Compute Base Geometry
        # We pass through all arguments
        gamma = self.base_christoffel(v, x, force=force, **kwargs)
        
        # 2. Compute Confusion
        # Confusion ~ Magnitude of Force (Error Gradient)
        if force is not None:
            # Force magnitude squared: ||F||^2
            confusion = (force ** 2).mean(dim=-1, keepdim=True)
            
            # 3. Deform Geometry
            # Scaling factor: 1 + lambda * confusion
            # If force is high, gamma increases -> curvature increases -> "gravity" increases
            # This generally slows down the flow (braking effect) if gamma opposes velocity, 
            # or deviates it strongly.
            scale = 1.0 + self.sensitivity * confusion
            
            gamma = gamma * scale
            
            # Optional: Return friction separately if base supported it
            if isinstance(gamma, tuple):
                # (gamma, mu)
                return gamma[0] * scale, gamma[1] * scale
                
        return gamma
