import numpy as np
import torch
import gfn.math.geometry.riemannian as pure_geo
import gfn.math.geometry.torch_geometry as torch_geo

def verify_hyperbolic_parity():
    print("Verifying Hyperbolic Christoffel Parity...")
    dim = 4
    x_np = np.random.randn(dim).astype(np.float32)
    v_np = np.random.randn(dim).astype(np.float32)
    
    # Pure NumPy
    gamma_np = pure_geo.hyperbolic_christoffel(x_np, v_np)
    
    # PyTorch
    x_torch = torch.from_numpy(x_np)
    v_torch = torch.from_numpy(v_np)
    gamma_torch = torch_geo.hyperbolic_christoffel_torch(x_torch, v_torch)
    
    # Compare
    diff = np.abs(gamma_np - gamma_torch.numpy()).max()
    print(f"Max difference: {diff}")
    assert diff < 1e-6, "Parity check failed!"
    print("SUCCESS: Pure vs Torch parity confirmed.")

if __name__ == "__main__":
    verify_hyperbolic_parity()
