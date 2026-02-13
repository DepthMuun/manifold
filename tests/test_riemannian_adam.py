import torch
import unittest
import sys
import os
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfn.optimizers.riemannian_adam import RiemannianAdam
print(f"DEBUG: RiemannianAdam source: {RiemannianAdam.__module__} from {os.path.abspath(sys.modules[RiemannianAdam.__module__].__file__)}")

class MockChristoffel(torch.nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim
    def forward(self, v, x, **kwargs):
        # Return a simple velocity-dependent curvature
        return 0.1 * v

class TestRiemannianAdam(unittest.TestCase):
    def test_orthogonal_preservation_cayley(self):
        # Orthogonal matrix should remain orthogonal after Cayley update
        dim = 4
        p = torch.eye(dim, requires_grad=True)
        
        optimizer = RiemannianAdam([p], lr=0.1, retraction='cayley')
        
        # Set artificial gradient
        p.grad = torch.randn(dim, dim)
        
        # Check skew-symmetry of V internally (we can't access V easily, but we can check grad)
        grad_skewed = optimizer._project_tangent(p.data, p.grad.data, 'cayley')
        grad_skew_error = torch.norm(grad_skewed + grad_skewed.t())
        print(f"DEBUG: Grad Skew Error: {grad_skew_error.item():.2e}")
        
        optimizer.step()
        
        # Check orthogonality: P @ P.T = I
        identity = torch.eye(dim)
        ortho_error = torch.norm(p.data @ p.data.t() - identity)
        
        print(f"Cayley Orthogonality Error: {ortho_error.item():.2e}")
        self.assertLess(ortho_error.item(), 1e-4, "Cayley retraction should preserve orthogonality")

    def test_torus_wrapping_and_transport(self):
        # Torus parameters should stay in [-pi, pi]
        dim = 10
        p = torch.nn.Parameter(torch.full((1, dim), 3.1)) # Near pi
        
        # Mock christoffel for transport
        christ = MockChristoffel(dim)
        optimizer = RiemannianAdam([{'params': [p], 'christoffel': christ}], lr=0.5, retraction='torus')
        
        # Gradient that pushes p beyond pi
        p.grad = torch.full((1, dim), -1.0) # p = p - lr * (-1) = 3.1 + 0.5 = 3.6 > pi
        optimizer.step()
        
        # Value should be wrapped
        val = p.data[0, 0].item()
        print(f"Wrapped Torus Value (target 3.6 - 2pi): {val:.4f}")
        
        self.assertLessEqual(val, math.pi)
        self.assertGreaterEqual(val, -math.pi)
        self.assertAlmostEqual(val, 3.6 - 2*math.pi, places=4)
        
        # Check that exp_avg was transported (it shouldn't be zero/destroyed)
        state = optimizer.state[p]
        self.assertIn('exp_avg', state)
        self.assertEqual(state['exp_avg'].shape, p.shape)

    def test_sphere_projection(self):
        # Gradient parallel to p should be projected out in 'normalize' mode
        dim = 5
        p = torch.randn(1, dim, requires_grad=True)
        p.data = p.data / p.data.norm() # On sphere
        
        optimizer = RiemannianAdam([p], lr=0.1, retraction='normalize')
        
        # Gradient strictly parallel to p
        p.grad = p.data.clone()
        
        # Project tangent should make this gradient zero
        projected_grad = optimizer._project_tangent(p.data, p.grad.data, 'normalize')
        
        proj_norm = projected_grad.norm().item()
        print(f"DEBUG: p_norm={p.data.norm().item()}, grad_norm={p.grad.data.norm().item()}")
        print(f"DEBUG: projected_grad_norm={proj_norm}")
        print(f"Sphere Gradient Projection Norm: {proj_norm:.2e}")
        self.assertLess(proj_norm, 1e-6, "Gradient parallel to p should be projected out on sphere")

if __name__ == '__main__':
    unittest.main()
