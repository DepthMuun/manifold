import torch
import torch.nn as nn
import os
import sys

# Add local directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import autograd

def test_leapfrog_gradients():
    print("\n" + "="*80)
    print("Verifying Leapfrog Adjoint Gradients")
    print("="*80)
    
    device = torch.device("cuda")
    dtype = torch.float64 # Use double precision for gradcheck
    
    B, D, R = 4, 8, 4
    steps = 5
    dt = 0.1
    
    # Inputs requiring gradients
    x = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    force = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    U = torch.randn(D, R, device=device, dtype=dtype, requires_grad=True)
    W = torch.randn(D, R, device=device, dtype=dtype, requires_grad=True)
    
    # Optional parameters for friction
    Wf = torch.randn(D, D, device=device, dtype=dtype, requires_grad=True)
    bf = torch.randn(D, device=device, dtype=dtype, requires_grad=True)
    
    plasticity = 0.5
    topology = 0 # Euclidean
    Radi = 2.0
    radi = 1.0

    def integrator_func(x_in, v_in, force_in, U_in, W_in, Wf_in, bf_in):
        x_out, v_out = autograd.leapfrog_fused_autograd(
            x_in, v_in, force_in, U_in, W_in,
            dt, 1.0, steps, topology,
            Wf_in, bf_in, plasticity, Radi, radi
        )
        # Use a scalar loss from both outputs
        return (x_out**2).sum() + (v_out**2).sum()

    print(f"Running gradcheck (Double Precision, Steps={steps})...")
    try:
        torch.autograd.gradcheck(
            integrator_func, (x, v, force, U, W, Wf, bf),
            eps=1e-6, atol=1e-4, rtol=1e-3
        )
        print("✓ PASS: Leapfrog Adjoint gradients match numerical gradients!")
    except Exception as e:
        print(f"✗ FAIL: Leapfrog Gradient mismatch!")
        print(e)

def test_heun_gradients():
    print("\n" + "="*80)
    print("Verifying Heun (RK2) Adjoint Gradients")
    print("="*80)
    
    device = torch.device("cuda")
    dtype = torch.float64
    
    B, D, R = 4, 8, 4
    steps = 3
    dt = 0.1
    
    x = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    force = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    U = torch.randn(D, R, device=device, dtype=dtype, requires_grad=True)
    W = torch.randn(D, R, device=device, dtype=dtype, requires_grad=True)
    
    topology = 0
    Radi = 2.0
    radi = 1.0

    def integrator_func(x_in, v_in, force_in, U_in, W_in):
        x_out, v_out = autograd.heun_fused_autograd(
            x_in, v_in, force_in, U_in, W_in,
            dt, 1.0, steps, topology, Radi, radi
        )
        return (x_out**2).sum() + (v_out**2).sum()

    print(f"Running gradcheck (Double Precision, Steps={steps})...")
    try:
        torch.autograd.gradcheck(
            integrator_func, (x, v, force, U, W),
            eps=1e-6, atol=1e-4, rtol=1e-3
        )
        print("✓ PASS: Heun Adjoint gradients match numerical gradients!")
    except Exception as e:
        print(f"✗ FAIL: Heun Gradient mismatch!")
        print(e)

if __name__ == "__main__":
    test_leapfrog_gradients()
    test_heun_gradients()
