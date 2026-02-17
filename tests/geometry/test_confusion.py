import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfn.geometry.confusion import ConfusionChristoffel
from gfn.layers.base import MLayer

class MockChristoffel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, v, x, force=None, **kwargs):
        # Return constant 1s
        return torch.ones_like(v)

class TestConfusionMetric(unittest.TestCase):
    def test_confusion_scaling(self):
        dim = 16
        batch = 2
        sensitivity = 2.0
        
        base = MockChristoffel(dim)
        confusion_geo = ConfusionChristoffel(base, sensitivity=sensitivity)
        
        x = torch.randn(batch, dim)
        v = torch.randn(batch, dim)
        
        # Case 1: Zero force -> No scaling (1.0)
        force_zero = torch.zeros(batch, dim)
        out_zero = confusion_geo(v, x, force=force_zero)
        # Expected: base output (1.0) * (1 + 2.0 * 0) = 1.0
        self.assertTrue(torch.allclose(out_zero, torch.ones_like(out_zero)))
        
        # Case 2: Unit force -> Scaling
        force_unit = torch.ones(batch, dim)
        # Confusion = mean(1^2) = 1.0
        # Scale = 1 + 2.0 * 1.0 = 3.0
        out_unit = confusion_geo(v, x, force=force_unit)
        self.assertTrue(torch.allclose(out_unit, 3.0 * torch.ones_like(out_unit)))
        
    def test_mlayer_integration(self):
        dim = 16
        heads = 4
        config = {
            'active_inference': {
                'confusion_metric': {
                    'enabled': True,
                    'sensitivity': 0.5
                }
            }
        }
        
        layer = MLayer(dim, heads=heads, physics_config=config)
        
        # Check wrapping
        head_dim = dim // heads
        self.assertIsInstance(layer.christoffels[0], ConfusionChristoffel)
        self.assertEqual(layer.christoffels[0].sensitivity, 0.5)
        
        # Forward pass
        x = torch.randn(2, dim)
        v = torch.randn(2, dim)
        f = torch.randn(2, dim)
        
        x_out, v_out, _, _ = layer(x, v, force=f)
        self.assertEqual(x_out.shape, (2, dim))

if __name__ == '__main__':
    unittest.main()
