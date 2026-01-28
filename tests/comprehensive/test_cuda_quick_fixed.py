#!/usr/bin/env python3
"""
Quick CUDA Test - Fixed Version
===============================

Test rápido para verificar que los kernels CUDA están funcionando correctamente.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_cuda_kernels():
    """Test rápido de kernels CUDA"""
    print("="*60)
    print("QUICK CUDA KERNEL TEST")
    print("="*60)
    
    try:
        from gfn.cuda.ops import CUDA_AVAILABLE, recurrent_manifold_fused
        print(f"✓ CUDA module imported successfully")
        print(f"  CUDA Available: {CUDA_AVAILABLE}")
        
        if not CUDA_AVAILABLE:
            print("⚠️  CUDA not available - cannot test kernels")
            return False
            
    except Exception as e:
        print(f"✗ Failed to import CUDA module: {e}")
        return False
    
    # Test básico de kernel
    try:
        device = torch.device('cuda')
        
        # Crear tensores pequeños
        batch_size = 2
        seq_len = 4
        dim = 8
        num_heads = 1
        
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        v = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        f = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        U_stack = torch.randn(num_heads, dim, dim // num_heads, device=device, dtype=torch.float32)
        W_stack = torch.randn(num_heads, dim // num_heads, dim, device=device, dtype=torch.float32)
        dt_scales = torch.ones(num_heads, device=device, dtype=torch.float32)
        forget_rates = torch.ones(num_heads, device=device, dtype=torch.float32)
        
        print(f"\nInput shapes:")
        print(f"  x: {x.shape}")
        print(f"  v: {v.shape}")
        print(f"  f: {f.shape}")
        print(f"  U_stack: {U_stack.shape}")
        print(f"  W_stack: {W_stack.shape}")
        
        # Ejecutar kernel
        result = recurrent_manifold_fused(
            x, v, f, U_stack, W_stack, 0.1, dt_scales, forget_rates,
            num_heads=num_heads, topology=1, R=2.0, r=1.0
        )
        
        if result is None:
            print("⚠️  CUDA kernel returned None - using Python fallback")
            return False
        
        print(f"✓ CUDA kernel executed successfully")
        print(f"  Result type: {type(result)}")
        
        # El kernel puede devolver una tupla (x_new, v_new) o un tensor
        if isinstance(result, tuple):
            print(f"  Result components: {len(result)}")
            for i, component in enumerate(result):
                print(f"    Component {i}: shape={component.shape}, device={component.device}")
                if torch.isnan(component).any():
                    print(f"✗ Component {i} contains NaN values")
                    return False
                if torch.isinf(component).any():
                    print(f"✗ Component {i} contains Inf values")
                    return False
        else:
            print(f"  Result shape: {result.shape}")
            print(f"  Result device: {result.device}")
            
            # Verificar validez
            if torch.isnan(result).any():
                print("✗ CUDA result contains NaN values")
                return False
            
            if torch.isinf(result).any():
                print("✗ CUDA result contains Inf values")
                return False
        
        print("✓ CUDA result is valid")
        return True
        
    except Exception as e:
        print(f"✗ CUDA kernel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_integration():
    """Test de integración con modelo"""
    print("\n" + "="*60)
    print("MODEL INTEGRATION TEST")
    print("="*60)
    
    try:
        from gfn.model import Manifold
        print("✓ Model imported successfully")
        
        # Crear modelo simple
        model = Manifold(
            vocab_size=2,
            dim=8,
            depth=1,
            heads=1,
            integrator_type='heun',
            holographic=True
        )
        print("✓ Model created successfully")
        
        # Test forward pass
        batch_size = 2
        seq_len = 4
        
        # Crear input simple
        input_ids = torch.randint(0, 2, (batch_size, seq_len))
        print(f"  Input shape: {input_ids.shape}")
        
        # Forward pass - el modelo devuelve solo logits
        logits = model(input_ids)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ejecutar todos los tests"""
    print("CUDA KERNEL VALIDATION")
    print("="*60)
    
    # Test 1: CUDA kernels
    cuda_ok = test_cuda_kernels()
    
    # Test 2: Model integration
    model_ok = test_model_integration()
    
    # Resumen
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"CUDA Kernels: {'✅ PASS' if cuda_ok else '❌ FAIL'}")
    print(f"Model Integration: {'✅ PASS' if model_ok else '❌ FAIL'}")
    
    if cuda_ok and model_ok:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())