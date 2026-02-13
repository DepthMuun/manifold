import torch
import torch.nn as nn
from ..geometry.confusion import ConfusionChristoffel

class CuriosityNoise(nn.Module):
    """
    Curiosity-Driven Entropy Noise (Paper 26).
    
    Proactively injects noise into the velocity state based on the local 
    "Confusion" (surprise/uncertainty) of the agent.
    
    Physics Principle:
    - In regions of high uncertainty (high force F), we increase "Temperature" T.
    - High T leads to more stochastic exploration (Brownian drift).
    - This prevents the model from getting stuck in local minima of the manifold.
    
    Formula:
        v_new = v + lambda_c * Confusion(F) * eps
        where eps ~ N(0, I)
    """
    def __init__(self, dim, base_std=0.01, sensitivity=1.0):
        super().__init__()
        self.dim = dim
        self.base_std = base_std
        self.sensitivity = sensitivity
        
    def forward(self, v, force=None, training=True):
        """
        Inject curiosity noise.
        
        Args:
            v: Velocity tensor [Batch, Dim]
            force: Current force F [Batch, Dim] (proxy for confusion)
            training: Whether to apply noise (active inference can still use it in eval)
            
        Returns:
            v_noisy: Velocity with injected curiosity jitter
        """
        if not training or force is None:
            return v
            
        # 1. Compute Confusion Proxy (Force Magnitude)
        # Higher force = higher confusion = higher noise
        confusion = (force ** 2).mean(dim=-1, keepdim=True) # [Batch, 1]
        
        # 2. Compute Dynamic Scale
        # Scale = base_std * (1 + sensitivity * confusion)
        scale = self.base_std * (1.0 + self.sensitivity * confusion)
        
        # 3. Generate Noise
        noise = torch.randn_like(v) * scale
        
        return v + noise
