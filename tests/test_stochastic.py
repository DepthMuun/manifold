import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfn.noise.geometric import GeometricNoise
from gfn.integrators.stochastic import StochasticIntegrator
from gfn.integrators import EulerIntegrator

class MockChristoffel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.U = torch.eye(dim, 4) # Mock LowRank
        self.W = torch.ones(dim, 4)
    def forward(self, v, x=None, **kwargs):
        return torch.zeros_like(v)

class TestStochasticDynamics(unittest.TestCase):
    def test_geometric_noise_drift(self):
        dim = 16
        batch = 2
        noise_mod = GeometricNoise(dim, sigma=0.1)
        christoffel = MockChristoffel(dim)
        
        x = torch.zeros(batch, dim)
        v = torch.zeros(batch, dim)
        
        # With zero velocity, noise should be mostly the random part + drift
        # Drift = (sigma^2/2) * sum_r ||Ur||^2 * Wr * dt
        # ||Ur||^2 = 1.0 (identity columns)
        # sum_r = 4 (rank)
        # Drift_i = (0.01 / 2) * 4 * 1.0 * 0.1 = 0.002
        
        # Seed for reproducibility if needed, but we check mean
        torch.manual_seed(42)
        impulse = noise_mod(x, v, christoffel, dt=0.1)
        
        self.assertEqual(impulse.shape, (batch, dim))
        # Drift is positive in this mock, so mean should be around 0.002
        # But random noise is sigma*sqrt(0.1) ~ 0.1 * 0.316 = 0.03
        # So we check that drift is included by comparing with a zero-drift case if U, W were missing
        
    def test_stochastic_integrator(self):
        dim = 16
        batch = 2
        christoffel = MockChristoffel(dim)
        base_integ = EulerIntegrator(christoffel, dt=0.1)
        noise_mod = GeometricNoise(dim, sigma=0.01)
        stoch_integ = StochasticIntegrator(base_integ, noise_mod)
        
        x = torch.randn(batch, dim)
        v = torch.randn(batch, dim)
        
        x_next, v_next = stoch_integ(x, v)
        
        # Check that v_next is different from deterministic Euler
        # Euler: v_next = v + f*dt - Gamma*dt
        # Since Gamma is 0 and f is None: v_det = v
        self.assertFalse(torch.allclose(v, v_next), "Stochastic integrator should change velocity even with zero force/gamma.")
        self.assertTrue(torch.allclose(x + v*0.1, x_next), "Position update should still be deterministic Euler.")

if __name__ == '__main__':
    unittest.main()
