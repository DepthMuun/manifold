import torch
import unittest
import sys
import os
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfn.integrators.symplectic.pefrl import PEFRLIntegrator

class HarmonicChristoffel(torch.nn.Module):
    """
    Simple Harmonicoscillator: Force = -k*x
    Here we simulate it as Christoffel(v,v) = x (k=1)
    Energy H = 1/2 v^2 + 1/2 x^2
    """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, v, x, **kwargs):
        # We want a = -x
        # a = -Gamma(v,v) -> Gamma(v,v) = x
        return x

class TestPEFRLIntegrator(unittest.TestCase):
    def test_energy_conservation(self):
        dim = 10
        dt = 0.1
        steps = 100
        
        christ = HarmonicChristoffel(dim)
        integ = PEFRLIntegrator(christ, dt=dt)
        
        # Initial state: Circle in phase space
        x = torch.zeros(1, dim)
        v = torch.ones(1, dim)
        
        def energy(qx, pv):
            return 0.5 * torch.sum(qx**2 + pv**2)
        
        e_init = energy(x, v)
        
        curr_x, curr_v = x, v
        try:
            for _ in range(steps):
                curr_x, curr_v = integ(curr_x, curr_v)
        except Exception as e:
            import traceback
            print(f"FAILED during integration: {e}")
            traceback.print_exc()
            raise e
            
        e_final = energy(curr_x, curr_v)
        
        drift = torch.abs(e_final - e_init) / e_init
        
        print(f"PEFRL Energy Drift after {steps} steps: {drift.item():.2e}")
        
        # 4th order should be very precise
        self.assertLess(drift.item(), 1e-5, "PEFRL should conserve energy well on harmonic oscillator")

    def test_symplectic_jacobian(self):
        # Symplectic maps must have Jacobian determinant 1 (Liouville)
        dim = 2
        dt = 0.5
        christ = HarmonicChristoffel(dim)
        integ = PEFRLIntegrator(christ, dt=dt)
        
        x = torch.randn(1, dim, requires_grad=True)
        v = torch.randn(1, dim, requires_grad=True)
        
        next_x, next_v = integ(x, v)
        
        # Compute Jacobian [4, 4]
        outputs = torch.cat([next_x, next_v], dim=-1).squeeze(0)
        
        jacobian = []
        for i in range(dim * 2):
            # Differentiate w.r.t original tensors
            grad_x = torch.autograd.grad(outputs[i], x, retain_graph=True, allow_unused=True)[0]
            grad_v = torch.autograd.grad(outputs[i], v, retain_graph=True, allow_unused=True)[0]
            
            if grad_x is None: grad_x = torch.zeros_like(x)
            if grad_v is None: grad_v = torch.zeros_like(v)
            
            jacobian.append(torch.cat([grad_x, grad_v], dim=-1))
        
        jacobian = torch.cat(jacobian, dim=0)
        det = torch.det(jacobian)
        
        print(f"PEFRL Jacobian Determinant: {det.item():.6f}")
        self.assertAlmostEqual(det.item(), 1.0, places=4, msg="PEFRL must be symplectic (det J = 1)")

if __name__ == '__main__':
    unittest.main()
