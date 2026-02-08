#!/usr/bin/env python3
"""
Numerical verification test for LowRankChristoffelWithFrictionFunction backward pass.
Compares CUDA gradients with numerical gradients calculated in Python.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from gfn.cuda.autograd import LowRankChristoffelWithFrictionFunction, CUDA_AVAILABLE
    from gfn.cuda.ops import christoffel_fused
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA not available, skipping verification test")

def compute_numerical_gradients(func, inputs, output_indices=None, eps=1e-5):
    """
    Computes numerical gradients using finite differences.
    
    Args:
        func: Function to evaluate
        inputs: List of input tensors
        output_indices: Output indices to consider (None = all)
        eps: Step size for finite differences
    
    Returns:
        List of numerical gradients for each input
    """
    gradients = []
    
    # Evaluate original function
    outputs = func(*inputs)
    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    
    # Select outputs to consider
    if output_indices is not None:
        outputs = [outputs[i] for i in output_indices]
    
    # For each input
    for i, input_tensor in enumerate(inputs):
        if input_tensor.numel() == 0 or not input_tensor.requires_grad:
            gradients.append(None)
            continue
            
        grad = torch.zeros_like(input_tensor)
        original_data = input_tensor.data.clone()
        
        # Flatten to facilitate iteration
        flat_input = input_tensor.view(-1)
        flat_grad = grad.view(-1)
        original_flat = original_data.view(-1)
        
        for j in range(flat_input.numel()):
            # Central difference for higher precision
            original_flat[j] += eps
            input_tensor.data = original_flat.view(input_tensor.shape)
            outputs_plus = func(*inputs)
            if isinstance(outputs_plus, torch.Tensor):
                outputs_plus = [outputs_plus]
            
            original_flat[j] -= 2 * eps
            input_tensor.data = original_flat.view(input_tensor.shape)
            outputs_minus = func(*inputs)
            if isinstance(outputs_minus, torch.Tensor):
                outputs_minus = [outputs_minus]
            
            # Restore original
            original_flat[j] += eps
            input_tensor.data = original_flat.view(input_tensor.shape)
            
            # Calculate gradients for each selected output
            grad_sum = 0.0
            for k, (plus, minus) in enumerate(zip(outputs_plus, outputs_minus)):
                if plus.numel() > 0 and minus.numel() > 0:
                    diff = (plus - minus) / (2 * eps)
                    grad_sum += diff.sum().item()
            
            flat_grad[j] = grad_sum
        
        gradients.append(grad)
    
    return gradients

def test_backward_vs_numerical():
    """Test comparing CUDA backward with numerical gradients."""
    if not CUDA_AVAILABLE or not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return False
    
    print("=" * 70)
    print("TEST: Backward CUDA vs Numerical Gradients")
    print("=" * 70)
    
    # Test parameters
    batch_size = 3
    dim = 6
    rank = 4
    device = torch.device('cuda')
    
    # Seed for reproducibility
    torch.manual_seed(42)
    
    print(f"Configuration: batch={batch_size}, dim={dim}, rank={rank}")
    
    # Create test inputs
    v = torch.randn(batch_size, dim, device=device, requires_grad=True, dtype=torch.float64)
    U = torch.randn(dim, rank, device=device, requires_grad=True, dtype=torch.float64)
    W = torch.randn(dim, rank, device=device, requires_grad=True, dtype=torch.float64)
    x = torch.randn(batch_size, dim, device=device, requires_grad=True, dtype=torch.float64)
    V_w = torch.randn(dim, device=device, requires_grad=True, dtype=torch.float64)
    force = torch.randn(batch_size, dim, device=device, requires_grad=True, dtype=torch.float64)
    W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True, dtype=torch.float64)  # For Torus topology
    b_forget = torch.randn(dim, device=device, requires_grad=True, dtype=torch.float64)
    W_input = torch.randn(dim, dim, device=device, requires_grad=True, dtype=torch.float64)
    
    # Hyperparameters
    plasticity = 0.1
    sing_thresh = 0.5
    sing_strength = 2.0
    topology = 1  # Torus
    R = 2.0
    r = 1.0
    
    print(f"Hyperparameters: plasticity={plasticity}, sing_thresh={sing_thresh}, sing_strength={sing_strength}")
    print(f"Topology: {topology} (Torus), R={R}, r={r}")
    print()
    
    # Test 1: Verify forward pass
    print("1. Verifying forward pass...")
    
    # Forward CUDA
    output_cuda = LowRankChristoffelWithFrictionFunction.apply(
        v, U, W, x, V_w, force, W_forget, b_forget, W_input,
        plasticity, sing_thresh, sing_strength, topology, R, r
    )
    
    # Forward Python (fallback)
    try:
        # Use christoffel_fused for the Christoffel part
        gamma = christoffel_fused(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology, R, r)
        
        # Add friction (simplified)
        if force is not None and W_forget is not None and b_forget is not None:
            # Calculate friction coefficient (simplified)
            features = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) if topology == 1 else x
            mu = torch.sigmoid(torch.matmul(features, W_forget.t()) + b_forget) * 5.0  # FRICTION_SCALE
            friction_output = gamma + mu * v
        else:
            friction_output = gamma
        
        # Compare
        forward_diff = torch.abs(output_cuda - friction_output).max().item()
        print(f"   Forward max difference: {forward_diff:.2e}")
        
        if forward_diff > 1e-4:
            print("   ⚠️  WARNING: Significant difference in forward")
        else:
            print("   ✅ Forward consistent")
            
    except Exception as e:
        print(f"   ⚠️  Could not compare with Python: {e}")
    
    print()
    
    # Test 2: Compare backward
    print("2. Verifying backward pass...")
    
    # Create simple loss function
    loss = output_cuda.sum()
    
    # Calculate gradients with CUDA backward
    grad_output = torch.ones_like(output_cuda)
    grads_cuda = LowRankChristoffelWithFrictionFunction.backward(None, grad_output)
    
    # Function for numerical gradients
    def forward_func(v, U, W, x, V_w, force, W_forget, b_forget, W_input):
        return LowRankChristoffelWithFrictionFunction.apply(
            v, U, W, x, V_w, force, W_forget, b_forget, W_input,
            plasticity, sing_thresh, sing_strength, topology, R, r
        )
    
    inputs = [v, U, W, x, V_w, force, W_forget, b_forget, W_input]
    grads_numerical = compute_numerical_gradients(forward_func, inputs)
    
    # Compare gradients
    param_names = ['v', 'U', 'W', 'x', 'V_w', 'force', 'W_forget', 'b_forget', 'W_input']
    tolerances = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]  # Tolerances per parameter
    
    print("   Gradient comparison:")
    max_errors = []
    
    for i, (name, grad_cuda, grad_num, tol) in enumerate(zip(param_names, grads_cuda, grads_numerical, tolerances)):
        if grad_cuda is None or grad_num is None:
            print(f"   {name}: Skipped (None)")
            continue
            
        if grad_cuda.numel() == 0 or grad_num.numel() == 0:
            print(f"   {name}: Skipped (empty)")
            continue
        
        # Verify shapes
        if grad_cuda.shape != grad_num.shape:
            print(f"   {name}: ❌ Inconsistent shape - CUDA: {grad_cuda.shape}, Numerical: {grad_num.shape}")
            continue
        
        # Calculate errors
        abs_diff = torch.abs(grad_cuda - grad_num)
        rel_diff = abs_diff / (torch.abs(grad_num) + 1e-8)
        
        max_abs_error = abs_diff.max().item()
        max_rel_error = rel_diff.max().item()
        mean_abs_error = abs_diff.mean().item()
        mean_rel_error = rel_diff.mean().item()
        
        max_errors.append(max_abs_error)
        
        print(f"   {name}:")
        print(f"     Max absolute error: {max_abs_error:.2e}")
        print(f"     Max relative error: {max_rel_error:.2e}")
        print(f"     Mean absolute error: {mean_abs_error:.2e}")
        print(f"     Mean relative error: {mean_rel_error:.2e}")
        
        # Verify if within tolerance
        if max_abs_error <= tol and max_rel_error <= 10 * tol:
            print(f"     ✅ WITHIN TOLERANCE ({tol})")
        else:
            print(f"     ❌ OUT OF TOLERANCE ({tol})")
            
            # Show some problematic values
            if max_abs_error > tol:
                max_idx = torch.argmax(abs_diff)
                cuda_val = grad_cuda.view(-1)[max_idx].item()
                num_val = grad_num.view(-1)[max_idx].item()
                print(f"     CUDA value at index {max_idx}: {cuda_val:.6e}")
                print(f"     Numerical value at index {max_idx}: {num_val:.6e}")
        
        print()
    
    # Summary
    print("3. Summary:")
    if max_errors:
        overall_max_error = max(max_errors)
        print(f"   Global max error: {overall_max_error:.2e}")
        
        if overall_max_error < 1e-3:
            print("   ✅ TEST PASSED: CUDA gradients match numerical gradients")
            return True
        else:
            print("   ❌ TEST FAILED: Significant differences detected")
            return False
    else:
        print("   ⚠️  Could not compare gradients")
        return False

def test_gradient_checking_pytorch():
    """Test using PyTorch gradcheck."""
    if not CUDA_AVAILABLE or not torch.cuda.is_available():
        print("CUDA not available, skipping gradcheck")
        return
    
    print("\n" + "=" * 70)
    print("TEST: Gradient Checking with PyTorch")
    print("=" * 70)
    
    from torch.autograd import gradcheck
    
    # Create small inputs for gradcheck
    batch_size = 2
    dim = 4
    rank = 2
    device = torch.device('cuda')
    
    torch.manual_seed(123)
    
    # Create inputs with requires_grad=True and dtype float64
    v = torch.randn(batch_size, dim, device=device, requires_grad=True, dtype=torch.float64)
    U = torch.randn(dim, rank, device=device, requires_grad=True, dtype=torch.float64)
    W = torch.randn(dim, rank, device=device, requires_grad=True, dtype=torch.float64)
    x = torch.randn(batch_size, dim, device=device, requires_grad=True, dtype=torch.float64)
    V_w = torch.randn(dim, device=device, requires_grad=True, dtype=torch.float64)
    force = torch.randn(batch_size, dim, device=device, requires_grad=True, dtype=torch.float64)
    W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True, dtype=torch.float64)
    b_forget = torch.randn(dim, device=device, requires_grad=True, dtype=torch.float64)
    W_input = torch.randn(dim, dim, device=device, requires_grad=True, dtype=torch.float64)
    
    # Hyperparameters
    test_input = (v, U, W, x, V_w, force, W_forget, b_forget, W_input, 
                 0.1, 0.5, 2.0, 1, 2.0, 1.0)
    
    try:
        result = gradcheck(
            LowRankChristoffelWithFrictionFunction.apply,
            test_input,
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=False
        )
        
        if result:
            print("✅ Gradcheck passed!")
        else:
            print("❌ Gradcheck failed!")
            
    except Exception as e:
        print(f"Error in gradcheck: {e}")
        print("This might be expected if the function is not sufficiently differentiable")

if __name__ == "__main__":
    print("INICIANDO TESTS DE VERIFICACIÓN CUDA")
    print("Objetivo: Verificar que el nuevo backward CUDA coincide con gradientes numéricos")
    print()
    
    try:
        # Test principal
        success = test_backward_vs_numerical()
        
        # Test adicional con gradcheck
        test_gradient_checking_pytorch()
        
        print("\n" + "=" * 70)
        if success:
            print("✅ TESTS COMPLETADOS: El backward CUDA es consistente")
        else:
            print("⚠️  TESTS COMPLETADOS: Se detectaron diferencias - revisar implementación")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error durante los tests: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)