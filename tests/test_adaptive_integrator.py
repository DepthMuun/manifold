import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfn.integrators.adaptive import AdaptiveIntegrator
from gfn.integrators.symplectic.euler import EulerIntegrator
from gfn.layers.base import MLayer

class MockChristoffel(torch.nn.Module):
    def forward(self, v, x, **kwargs):
        # Simple harmonic oscillator-like force: -x
        return x

class TestAdaptiveIntegrator(unittest.TestCase):
    def test_adaptive_execution_flow(self):
        dim = 16
        batch = 2
        
        # Base integrator
        base = EulerIntegrator(MockChristoffel(), dt=0.1)
        
        # Adaptive wrapper
        adaptive = AdaptiveIntegrator(base, tolerance=1e-5, max_depth=2)
        
        x = torch.randn(batch, dim)
        v = torch.randn(batch, dim)
        
        # Force high error by using high velocity? 
        # Actually random inputs usually trigger some subdivision if tolerance is tight.
        
        # Run forward
        x_out, v_out = adaptive(x, v)
        
        self.assertEqual(x_out.shape, (batch, dim))
        self.assertEqual(v_out.shape, (batch, dim))
        
    def test_mlayer_integration(self):
        dim = 16
        heads = 4
        config = {
            'active_inference': {
                'adaptive_resolution': {
                    'enabled': True,
                    'tolerance': 1e-4,
                    'max_depth': 2
                }
            }
        }
        
        layer = MLayer(dim, heads=heads, physics_config=config)
        
        # Check if integrator is wrapped
        self.assertIsInstance(layer.integrators[0], AdaptiveIntegrator)
        
        x = torch.randn(2, dim)
        v = torch.randn(2, dim)
        
        x_out, v_out, _, _ = layer(x, v)
        self.assertEqual(x_out.shape, (2, dim))

if __name__ == '__main__':
    unittest.main()
