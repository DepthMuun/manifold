import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfn.layers.base import MLayer
from gfn.losses.noether import noether_loss

class TestNoetherRegularization(unittest.TestCase):
    def test_noether_loss_zero_for_identical_heads(self):
        # Heads with shared manifold instances should have zero discrepancy
        dim = 16
        heads = 4
        isomeric_groups = [[0, 1], [2, 3]]
        
        # MLayer handles instance sharing internally if isomeric_groups matches
        layer = MLayer(dim, heads=heads, physics_config={'symmetries': {'isomeric_groups': isomeric_groups}})
        
        head_dim = dim // heads
        x = torch.randn(8, head_dim).repeat(1, heads)
        v = torch.randn(8, head_dim).repeat(1, heads)
        
        # Run with collect_christ=True
        _, _, _, christ_outputs = layer(x, v, collect_christ=True)
        
        # Check that heads 0/1 share the same instance
        self.assertTrue(layer.christoffels[0] is layer.christoffels[1])
        
        # Compute loss
        loss = noether_loss(christ_outputs, isomeric_groups=isomeric_groups, lambda_n=1.0)
        
        print(f"Noether Loss (Identical Heads): {loss.item():.2e}")
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_noether_loss_penalizes_divergence(self):
        # Independent heads should have non-zero loss initially
        dim = 8
        heads = 2
        isomeric_groups = [[0, 1]]
        
        # Create layer WITHOUT internal sharing (independent christoffels)
        # We manually simulate this by just passing different christoffel outputs
        
        c0 = torch.randn(4, 4)
        c1 = c0 + 0.1 * torch.randn(4, 4) # Perturb
        
        christ_outputs = [c0, c1]
        
        loss = noether_loss(christ_outputs, isomeric_groups=isomeric_groups, lambda_n=1.0)
        
        print(f"Noether Loss (Divergent Heads): {loss.item():.2e}")
        self.assertGreater(loss.item(), 0.0)

    def test_gradient_flow(self):
        # Verify that gradients flow through the loss back to the layer params
        dim = 8
        heads = 2
        isomeric_groups = [[0, 1]]
        
        # Force independent manifolds for testing
        layer = MLayer(dim, heads=heads)
        
        x = torch.randn(2, dim)
        v = torch.randn(2, dim)
        
        # Ensure we have gradients
        for p in layer.parameters(): p.requires_grad = True
        
        _, _, _, christ_outputs = layer(x, v, collect_christ=True)
        loss = noether_loss(christ_outputs, isomeric_groups=isomeric_groups, lambda_n=1.0)
        
        loss.backward()
        
        # Check if any weights got gradients
        has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in layer.parameters())
        self.assertTrue(has_grad, "Noether loss should propagate gradients to layer parameters")

if __name__ == '__main__':
    unittest.main()
