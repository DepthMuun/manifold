#!/usr/bin/env python3
"""
CUDA Kernel Status Test
=======================

Test para verificar el estado de los kernels CUDA compilados.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Verificar estado de CUDA kernels"""
    print("="*60)
    print("CUDA KERNEL STATUS VERIFICATION")
    print("="*60)
    
    # 1. Verificar disponibilidad de CUDA
    print("\n1. CUDA Availability Check:")
    try:
        from gfn.cuda.ops import CUDA_AVAILABLE
        print(f"   ✓ CUDA Available: {CUDA_AVAILABLE}")
        
        if not CUDA_AVAILABLE:
            print("   ⚠️  CUDA not available - kernels not loaded")
            return False
            
    except Exception as e:
        print(f"   ✗ Failed to check CUDA: {e}")
        return False
    
    # 2. Verificar que el módulo CUDA está cargado
    print("\n2. CUDA Module Loading Check:")
    try:
        from gfn.cuda.ops import gfn_cuda
        if gfn_cuda is not None:
            print(f"   ✓ gfn_cuda module loaded: {gfn_cuda}")
            print(f"   ✓ Module attributes: {[attr for attr in dir(gfn_cuda) if not attr.startswith('_')][:10]}")
        else:
            print("   ⚠️  gfn_cuda module is None")
            return False
            
    except Exception as e:
        print(f"   ✗ Failed to access gfn_cuda: {e}")
        return False
    
    # 3. Verificar kernels disponibles
    print("\n3. CUDA Kernels Available:")
    try:
        cuda_functions = [
            'recurrent_manifold_fused',
            'christoffel_fused',
            'recurrent_manifold_fused_autograd',
            'christoffel_fused_autograd'
        ]
        
        from gfn.cuda import autograd
        available_functions = []
        
        for func_name in cuda_functions:
            if hasattr(autograd, func_name):
                available_functions.append(func_name)
                print(f"   ✓ {func_name} available")
            else:
                print(f"   ✗ {func_name} not available")
        
        if available_functions:
            print(f"   ✓ Total CUDA functions: {len(available_functions)}")
        else:
            print("   ⚠️  No CUDA functions found")
            return False
            
    except Exception as e:
        print(f"   ✗ Failed to check CUDA functions: {e}")
        return False
    
    # 4. Test simple de kernel (sin verificación de resultados)
    print("\n4. CUDA Kernel Execution Test:")
    try:
        device = torch.device('cuda')
        
        # Crear tensores muy pequeños
        batch_size = 1
        seq_len = 2
        dim = 4
        num_heads = 1
        
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        v = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        f = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        U_stack = torch.randn(num_heads, dim, dim // num_heads, device=device, dtype=torch.float32)
        W_stack = torch.randn(num_heads, dim // num_heads, dim, device=device, dtype=torch.float32)
        dt_scales = torch.ones(num_heads, device=device, dtype=torch.float32)
        forget_rates = torch.ones(num_heads, device=device, dtype=torch.float32)
        
        print(f"   Input shapes: x={x.shape}, v={v.shape}, f={f.shape}")
        
        # Ejecutar kernel con manejo de errores
        from gfn.cuda.ops import recurrent_manifold_fused
        
        result = recurrent_manifold_fused(
            x, v, f, U_stack, W_stack, 0.1, dt_scales, forget_rates,
            num_heads=num_heads, topology=1, R=2.0, r=1.0
        )
        
        if result is not None:
            print(f"   ✓ Kernel executed (returned {type(result).__name__})")
            if isinstance(result, tuple):
                print(f"   ✓ Returned tuple with {len(result)} components")
                for i, comp in enumerate(result):
                    print(f"     Component {i}: {comp.shape}")
            else:
                print(f"   ✓ Returned tensor: {result.shape}")
            
            return True
        else:
            print("   ⚠️  Kernel returned None (Python fallback)")
            return False
            
    except Exception as e:
        print(f"   ✗ Kernel execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✅ CUDA KERNELS VERIFICATION COMPLETE")
    print("="*60)
    print("✓ CUDA is available and kernels are loaded")
    print("✓ gfn_cuda module is properly imported")
    print("✓ CUDA functions are accessible")
    print("✓ Kernels execute successfully")
    print("\n🎉 Your CUDA compilation is working correctly!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)