import torch
import torch.nn as nn
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfn.geometry.ricci import RicciFlowChristoffel
from gfn.geometry import ReactiveChristoffel

class TestRicciFlow(unittest.TestCase):
    def test_ricci_step(self):
        dim = 16
        rank = 4
        # We need a LowRank base to test the parameter modification
        base_geo = ReactiveChristoffel(dim, rank)
        # Initialize with random values so mul_ actually changes something
        nn.init.normal_(base_geo.W, std=0.1)
        nn.init.normal_(base_geo.U, std=0.1)
        
        ricci_geo = RicciFlowChristoffel(base_geo, lr=0.1)
        
        # Capture initial weights
        w_init = base_geo.W.clone()
        u_init = base_geo.U.clone()
        
        # Perform Ricci Flow step
        x_batch = torch.randn(4, dim)
        res = ricci_geo.ricci_flow_step(x_batch)
        
        self.assertTrue(res["ricci_smoothing"])
        
        # Verify parameters changed
        self.assertFalse(torch.allclose(w_init, base_geo.W), "W should be modified by Ricci Flow step.")
        self.assertFalse(torch.allclose(u_init, base_geo.U), "U should be modified by Ricci Flow step.")
        
    def test_forward_transparency(self):
        dim = 16
        rank = 4
        base_geo = ReactiveChristoffel(dim, rank)
        ricci_geo = RicciFlowChristoffel(base_geo)
        
        v = torch.randn(2, dim)
        x = torch.randn(2, dim)
        
        out_base = base_geo(v, x)
        out_ricci = ricci_geo(v, x)
        
        self.assertTrue(torch.allclose(out_base, out_ricci), "Ricci wrapper should be transparent in forward pass.")

if __name__ == '__main__':
    unittest.main()
