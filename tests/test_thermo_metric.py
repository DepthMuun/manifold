import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfn.geometry.thermo import ThermodynamicChristoffel
from gfn.layers.base import MLayer

class MockChristoffel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, v, x, force=None, **kwargs):
        # Return constant 1s
        return torch.ones_like(v)

class TestThermodynamicMetric(unittest.TestCase):
    def test_thermo_modulation(self):
        dim = 16
        batch = 4
        # T=1.0, alpha=1.0 for easy math
        thermo_geo = ThermodynamicChristoffel(MockChristoffel(dim), temperature=1.0, alpha=1.0)
        
        x = torch.randn(batch, dim)
        v = torch.randn(batch, dim)
        
        # Case 1: Low Energy (Force ~ 0) -> Modulation ~ exp(-0) = 1.0
        force_low = torch.zeros(batch, dim)
        out_low = thermo_geo(v, x, force=force_low)
        self.assertTrue(torch.allclose(out_low, torch.ones_like(out_low), atol=1e-4))
        
        # Case 2: High Energy (Force large) -> Modulation < 1.0 (Flatter geometry)
        # Energy = 100^2 = 10000
        # Modulation = exp(-1.0 * 10000 / 1.0) ~ 0
        force_high = torch.ones(batch, dim) * 100.0
        out_high = thermo_geo(v, x, force=force_high)
        
        # Should be very small
        self.assertTrue(torch.all(out_high < 0.1), "High energy should flatten geometry (small gamma)")
        
    def test_mlayer_integration(self):
        dim = 16
        heads = 4
        config = {
            'active_inference': {
                'thermodynamic_geometry': {
                    'enabled': True,
                    'temperature': 0.5,
                    'alpha': 0.1
                }
            }
        }
        
        layer = MLayer(dim, heads=heads, physics_config=config)
        
        # Check wrapping
        # The first wrapper should be Thermo (inner), then Confusion (if enabled). Here only Thermo.
        self.assertIsInstance(layer.christoffels[0], ThermodynamicChristoffel)
        self.assertEqual(layer.christoffels[0].alpha, 0.1)
        
        # Forward pass
        x = torch.randn(2, dim)
        v = torch.randn(2, dim)
        f = torch.randn(2, dim)
        
        x_out, v_out, _, _ = layer(x, v, force=f)
        self.assertEqual(x_out.shape, (2, dim))

if __name__ == '__main__':
    unittest.main()
