import torch
import torch.nn as nn
import torch.nn.functional as F

class ThermodynamicGating(nn.Module):
    """
    Thermodynamic Gating (Paper 03).
    
    Modulates the time-step 'dt' based on the Hamiltonian Energy (H) of the state.
    
    Physics Principle:
    - High Energy (H) corresponds to regions of high dynamic instability or curvature.
    - In these regions, we should slow down time (reduce dt) to integrate carefully.
    - In Low Energy regions (equilibrium), we can speed up time (increase dt).
    
    Formula:
        H = K(v) + U(x)
        K(v) = 0.5 * ||v||^2  (Kinetic)
        U(x) = 0.5 * ||x||^2  (Potential - Harmonic Oscillator ansatz)
        
        gate = sigmoid( (H_ref - H) / Temperature )
        
    If H > H_ref: gate < 0.5 (Slow down)
    If H < H_ref: gate > 0.5 (Speed up)
    """
    def __init__(self, dim, temperature=1.0, ref_energy=1.0, sensitivity=1.0):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.ref_energy = ref_energy
        self.sensitivity = sensitivity
        
        # Learnable reference energy and temperature?
        # Paper 03 suggests these can be fixed or learned. 
        # We make them learnable for flexibility.
        self.log_temp = nn.Parameter(torch.tensor(0.0)) # Init T=1.0
        self.ref_H = nn.Parameter(torch.tensor(ref_energy))
        
    def forward(self, x, v):
        """
        Args:
            x: Position [Batch, Dim]
            v: Velocity [Batch, Dim]
            
        Returns:
            gate: Scaling factor [Batch, 1] in range (0, 1)
        """
        # 1. Kinetic Energy (Euclidean approximation for gating efficiency)
        # K = 0.5 * v^2
        K = 0.5 * (v ** 2).sum(dim=-1, keepdim=True)
        
        # 2. Potential Energy (Harmonic approximation)
        # U = 0.5 * x^2
        U = 0.5 * (x ** 2).sum(dim=-1, keepdim=True)
        
        # 3. Total Hamiltonian
        H = K + U
        
        # 4. Thermodynamic Gating
        # T = exp(log_temp)
        T = self.log_temp.exp()
        
        # If H is HIGH -> (Ref - H) is NEGATIVE -> Sigmoid -> 0 -> Small dt
        # If H is LOW  -> (Ref - H) is POSITIVE -> Sigmoid -> 1 -> Large dt
        logits = (self.ref_H - H) / (T * self.sensitivity)
        
        gate = torch.sigmoid(logits)
        
        return gate
