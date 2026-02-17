import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfn.layers.base import MLayer
from gfn.layers.thermo import ThermodynamicGating

class TestThermodynamicGating(unittest.TestCase):
    def test_thermo_gating_module(self):
        dim = 16
        batch = 4
        gating = ThermodynamicGating(dim, temperature=1.0, ref_energy=5.0)
        
        x = torch.randn(batch, dim)
        v = torch.randn(batch, dim)
        
        gate = gating(x, v)
        self.assertEqual(gate.shape, (batch, 1))
        self.assertTrue(torch.all(gate >= 0.0))
        self.assertTrue(torch.all(gate <= 1.0))
        
        # Test physics: Higher energy should yield lower gate (slower time)
        # Case 1: Low energy
        x_low = torch.zeros(batch, dim)
        v_low = torch.zeros(batch, dim) # K=0, U=0 -> H=0. H < Ref (5.0). Gate > 0.5
        gate_low = gating(x_low, v_low)
        self.assertTrue(torch.all(gate_low > 0.5), f"Low energy should be fast! Got {gate_low}")
        
        # Case 2: High energy
        x_high = torch.randn(batch, dim) * 10
        v_high = torch.randn(batch, dim) * 10 # H >> 5.0. Gate < 0.5
        gate_high = gating(x_high, v_high)
        self.assertTrue(torch.all(gate_high < 0.5), f"High energy should be slow! Got {gate_high}")

    def test_mlayer_integration(self):
        dim = 16
        heads = 4
        config = {
            'active_inference': {
                'dynamic_time': {
                    'enabled': True,
                    'type': 'thermo'
                }
            }
        }
        
        layer = MLayer(dim, heads=heads, physics_config=config)
        
        # Check if correct module was initialized
        self.assertIsInstance(layer.gatings[0], ThermodynamicGating)
        
        x = torch.randn(2, dim)
        v = torch.randn(2, dim)
        
        # Run forward pass
        x_out, v_out, context, _ = layer(x, v)
        
        self.assertEqual(x_out.shape, (2, dim))
        self.assertEqual(v_out.shape, (2, dim))

if __name__ == '__main__':
    unittest.main()
