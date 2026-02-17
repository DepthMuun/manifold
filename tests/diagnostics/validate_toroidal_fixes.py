
import torch
import unittest
import math
import sys
import os
import psutil
import gc
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

# Ensure we can import from gfn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock gfn.cuda.ops BEFORE importing modules that depend on it
sys.modules['gfn.cuda.ops'] = MagicMock(CUDA_AVAILABLE=False)

from gfn.geometry.boundaries import apply_boundary_python
from gfn.integrators.symplectic.leapfrog import LeapfrogIntegrator, apply_boundary_python as integrator_boundary_fn
from gfn.model.fusion import CUDAFusionManager

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

class TestToroidalFixes(unittest.TestCase):
    
    def setUp(self):
        self.start_mem = get_process_memory()
        print(f"\n[Test Setup] Start Memory: {self.start_mem:.2f} MB")

    def tearDown(self):
        gc.collect()
        end_mem = get_process_memory()
        print(f"[Test Teardown] End Memory: {end_mem:.2f} MB (Delta: {end_mem - self.start_mem:.2f} MB)")
    
    def test_boundary_consistency(self):
        """Test that apply_boundary_python behaves correctly and is consistent."""
        print("\n[Test] Verifying Boundary Consistency...")
        
        # Test 1: Wrapping range [0, 2pi)
        x = torch.tensor([
            0.1,                # Within range
            2 * math.pi + 0.1,  # Wrap positive
            -0.1,               # Wrap negative
            4 * math.pi + 0.1   # Wrap multiple positive
        ])
        
        # Use topology_id=1 (Torus)
        wrapped = apply_boundary_python(x, topology_id=1)
        
        expected = torch.tensor([0.1, 0.1, 2 * math.pi - 0.1, 0.1])
        
        # Allow small numerical error due to float precision
        self.assertTrue(torch.allclose(wrapped, expected, atol=1e-5), 
                        f"Wrapping failed. Got {wrapped}, expected {expected}")
        
        # Test 2: Smoothness (Atan2 gradients)
        x_grad = torch.tensor([2 * math.pi], requires_grad=True)
        y = apply_boundary_python(x_grad, topology_id=1)
        y.backward()
        
        # Gradient should be 1.0 (continuous flow)
        self.assertTrue(torch.allclose(x_grad.grad, torch.tensor([1.0])),
                        f"Gradient check failed. Got {x_grad.grad}")
        
        # Test 3: Euclidean (topology_id=0) should be identity
        x_eucl = torch.tensor([100.0])
        wrapped_eucl = apply_boundary_python(x_eucl, topology_id=0)
        self.assertTrue(torch.allclose(wrapped_eucl, x_eucl), "Euclidean identity failed")
        
        print("[Pass] Boundary logic is correct.")

    def test_leapfrog_integration(self):
        """Test that LeapfrogIntegrator uses the correct boundary logic."""
        print("\n[Test] Verifying Leapfrog Integrator...")
        
        # Mock Christoffel
        christoffel = MagicMock()
        christoffel.return_value = (torch.zeros(1, 2), 0.0) # Gamma=0, Friction=0
        christoffel.topology_id = 1
        christoffel.is_torus = True
        
        integrator = LeapfrogIntegrator(christoffel, dt=0.1)
        
        # Initial state near boundary
        x = torch.tensor([[2 * math.pi - 0.05, 0.0]])
        v = torch.tensor([[1.0, 0.0]]) # Moving right
        
        # Step
        # x_new approx x + dt*v = 2pi - 0.05 + 0.1 = 2pi + 0.05
        # Should wrap to 0.05
        x_out, v_out = integrator(x, v)
        
        expected_x = 0.05
        self.assertTrue(abs(x_out[0, 0].item() - expected_x) < 1e-3,
                        f"Leapfrog wrapping failed. Got {x_out[0, 0].item()}")
        
        print("[Pass] Leapfrog integrator wraps correctly.")

    def test_fusion_manager_routing(self):
        """Test that CUDAFusionManager routes toroidal topology correctly."""
        print("\n[Test] Verifying CUDA Fusion Routing...")
        
        # Mock Manifold model
        model = MagicMock()
        model.heads = 2
        model.dim = 4
        model.integrator_type = 'leapfrog'
        model.training = False
        model.use_scan = False
        model.depth = 1
        
        # Set topology to torus
        model.physics_config = {'topology': {'type': 'torus'}}
        
        # Use SimpleNamespace to avoid MagicMock creating 'macro_manifold' automatically
        layer = SimpleNamespace()
        layer.dt_params = torch.tensor([0.1, 0.1])
        layer.base_dt = 0.05 # Add base_dt
        
        # Mock Christoffels with required attributes (gates)
        # They need to be accessed by attribute
        h1 = MagicMock()
        h2 = MagicMock()
        # Set forget_gate and input_gate as MagicMocks with weights
        h1.forget_gate = MagicMock()
        h1.forget_gate.weight = torch.randn(2, 2)
        h1.forget_gate.bias = torch.zeros(2)
        h1.forget_gate.out_features = 2
        h1.forget_gate.in_features = 2
        h1.input_gate = MagicMock()
        h1.input_gate.weight = torch.randn(2, 2)
        
        # Configure mocks to NOT have base_christoffel to avoid infinite unwrapping loop in fusion.py
        del h1.base_christoffel
        
        h1.V = MagicMock()
        h1.V.weight = torch.randn(1, 2)
        h1.V.bias = torch.zeros(1)
        
        # Duplicate for h2
        h2.forget_gate = MagicMock()
        h2.forget_gate.weight = torch.randn(2, 2)
        h2.forget_gate.bias = torch.zeros(2)
        h2.forget_gate.out_features = 2
        h2.forget_gate.in_features = 2
        h2.input_gate = MagicMock()
        h2.input_gate.weight = torch.randn(2, 2)
        
        del h2.base_christoffel
        
        h2.V = MagicMock()
        h2.V.weight = torch.randn(1, 2)
        h2.V.bias = torch.zeros(1)

        layer.christoffels = [h1, h2] # 2 heads
        
        # Add required attributes for prepare_parameters with REAL TENSORS
        layer.out_proj_x = MagicMock()
        layer.out_proj_x.weight = torch.randn(4, 4)
        layer.out_proj_x.bias = torch.zeros(4)
        layer.out_proj_v = MagicMock()
        layer.out_proj_v.weight = torch.randn(4, 4)
        layer.out_proj_v.bias = torch.zeros(4)
        layer.mixed_norm_x = MagicMock()
        layer.mixed_norm_x.weight = torch.ones(4)
        layer.mixed_norm_x.bias = torch.zeros(4)
        layer.mixed_norm_v = MagicMock()
        layer.mixed_norm_v.weight = torch.ones(4)
        layer.mixed_norm_v.bias = torch.zeros(4)
        layer.gatings = [] 
        layer.head_dim = 2 # dim 4 // heads 2
        
        # Mock layers list
        model.layers = [layer]
        # Mock parameters() iterator to return a tensor on CPU
        model.parameters = lambda: iter([torch.tensor([0.0])]) 
        
        manager = CUDAFusionManager(model)
        
        # Mock CUDA availability
        with patch.dict('sys.modules', {'gfn.cuda.ops': MagicMock(CUDA_AVAILABLE=True)}):
            # Force reload of fusion module or just rely on dynamic check in can_fuse?
            # CUDAFusionManager checks import inside can_fuse, but prepare_parameters doesn't check it.
            # However, prepare_parameters might use device.
            
            # Check prepare_parameters
            params = manager.prepare_parameters()
            
            self.assertIsNotNone(params, "prepare_parameters returned None")
            
            # Check 1: is_torus flag
            self.assertTrue(params['is_torus'], "is_torus flag should be True")
            self.assertEqual(params['topology_id'], 1, "topology_id should be 1")
            
            # Check 2: Dummy U/W tensors (zeros)
            # U_stack should be zeros
            self.assertTrue(torch.all(params['U_stack'] == 0), "U_stack should be zeros for Torus routing")
            self.assertTrue(torch.all(params['W_stack'] == 0), "W_stack should be zeros for Torus routing")
            
            print("[Pass] Fusion Manager prepares dummy weights for Torus.")

if __name__ == '__main__':
    unittest.main()
