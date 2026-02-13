import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfn.geometry.holographic import AdSCFTChristoffel

class MockChristoffel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, v, x=None, **kwargs):
        return torch.zeros_like(v)

class TestHolographicGeometry(unittest.TestCase):
    def test_conformal_gamma(self):
        dim = 16
        batch = 2
        holo_geo = AdSCFTChristoffel(MockChristoffel(dim))
        
        x = torch.randn(batch, dim)
        v = torch.randn(batch, dim)
        
        gamma = holo_geo(v, x)
        
        self.assertEqual(gamma.shape, (batch, dim))
        
        # Verify that gamma is not zero (dynamic radial field)
        self.assertFalse(torch.allclose(gamma, torch.zeros_like(gamma)), "Holographic gamma should be non-zero for random x, v")
        
    def test_radial_clamping(self):
        dim = 8
        holo_geo = AdSCFTChristoffel(MockChristoffel(dim), z_min=0.1, z_max=1.0)
        
        # Test large input to trigger clamping
        x = torch.ones(1, dim) * 1000.0
        z, _ = holo_geo.get_z_and_grad(x)
        
        self.assertLessEqual(z.item(), 1.0 + 1e-5)
        self.assertGreaterEqual(z.item(), 0.1)

if __name__ == '__main__':
    unittest.main()
