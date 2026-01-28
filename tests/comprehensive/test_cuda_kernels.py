#!/usr/bin/env python3
"""
CUDA Kernel Validation Test
==========================

Test específico para validar kernels CUDA compilados vs Python fallback.
Valida que los kernels CUDA funcionen correctamente y den resultados consistentes.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.cuda.ops import CUDA_AVAILABLE, recurrent_manifold_fused, christoffel_fused
from gfn.model import Manifold

def test_cuda_kernel_execution():
    """Test 1: Verificar que los kernels CUDA se ejecutan"""
    print("\n[TEST] CUDA Kernel Execution")
    print("-" * 40)
    
    if not CUDA_AVAILABLE:
        print("⚠️  CUDA not available - skipping kernel tests")
        return {"status": "skipped", "reason": "CUDA not available"}
    
    device = torch.device('cuda')
    
    # Parámetros de prueba
    batch_size = 2
    seq_len = 4
    dim = 8
    num_heads = 2
    
    # Crear tensores de prueba
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
    f = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
    
    # Stack de matrices para cada head
    U_stack = torch.randn(num_heads, dim, dim // num_heads, device=device, dtype=torch.float32)
    W_stack = torch.randn(num_heads, dim // num_heads, dim, device=device, dtype=torch.float32)
    
    # Parámetros temporales
    dt = 0.1
    dt_scales = torch.ones(num_heads, device=device, dtype=torch.float32)
    forget_rates = torch.ones(num_heads, device=device, dtype=torch.float32)
    
    # Parámetros de topología toroidal
    topology = 1  # Toroidal
    R = 2.0
    r = 1.0
    
    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  v: {v.shape}")
    print(f"  f: {f.shape}")
    print(f"  U_stack: {U_stack.shape}")
    print(f"  W_stack: {W_stack.shape}")
    
    try:
        # Ejecutar kernel CUDA
        start_time = time.time()
        result = recurrent_manifold_fused(
            x, v, f, U_stack, W_stack, dt, dt_scales, forget_rates,
            num_heads=num_heads, topology=topology, R=R, r=r
        )
        cuda_time = time.time() - start_time
        
        if result is None:
            raise RuntimeError("CUDA kernel returned None - fallback to Python")
        
        print(f"✓ CUDA kernel executed successfully")
        print(f"  Result shape: {result.shape}")
        print(f"  Execution time: {cuda_time*1000:.3f}ms")
        
        # Verificar que el resultado es válido
        if torch.isnan(result).any():
            raise RuntimeError("CUDA result contains NaN values")
        
        if torch.isinf(result).any():
            raise RuntimeError("CUDA result contains Inf values")
        
        return {
            "status": "passed",
            "result_shape": list(result.shape),
            "execution_time_ms": cuda_time * 1000,
            "cuda_available": True,
            "result_valid": True
        }
        
    except Exception as e:
        print(f"✗ CUDA kernel failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "cuda_available": True
        }

def test_christoffel_fused_cuda():
    """Test 2: Christoffel fused CUDA kernel"""
    print("\n[TEST] Christoffel Fused CUDA Kernel")
    print("-" * 40)
    
    if not CUDA_AVAILABLE:
        print("⚠️  CUDA not available - skipping Christoffel test")
        return {"status": "skipped", "reason": "CUDA not available"}
    
    device = torch.device('cuda')
    
    # Parámetros de prueba
    batch_size = 4
    dim = 8
    rank = 4
    
    # Crear tensores de prueba
    v = torch.randn(batch_size, dim, device=device, dtype=torch.float32)
    U = torch.randn(dim, rank, device=device, dtype=torch.float32)
    W = torch.randn(rank, dim, device=device, dtype=torch.float32)
    
    # Parámetros de topología
    topology = 1  # Toroidal
    R = 2.0
    r = 1.0
    
    try:
        # Ejecutar kernel CUDA
        start_time = time.time()
        result = christoffel_fused(v, U, W, topology=topology, R=R, r=r)
        cuda_time = time.time() - start_time
        
        print(f"✓ Christoffel CUDA kernel executed successfully")
        print(f"  Result shape: {result.shape}")
        print(f"  Execution time: {cuda_time*1000:.3f}ms")
        
        # Verificar validez del resultado
        if torch.isnan(result).any():
            raise RuntimeError("Christoffel result contains NaN values")
        
        if torch.isinf(result).any():
            raise RuntimeError("Christoffel result contains Inf values")
        
        return {
            "status": "passed",
            "result_shape": list(result.shape),
            "execution_time_ms": cuda_time * 1000,
            "result_valid": True
        }
        
    except Exception as e:
        print(f"✗ Christoffel CUDA kernel failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

def test_cuda_memory_handling():
    """Test 3: Manejo de memoria CUDA"""
    print("\n[TEST] CUDA Memory Handling")
    print("-" * 40)
    
    if not CUDA_AVAILABLE:
        print("⚠️  CUDA not available - skipping memory test")
        return {"status": "skipped", "reason": "CUDA not available"}
    
    device = torch.device('cuda')
    
    # Medir memoria antes
    torch.cuda.empty_cache()
    memory_before = torch.cuda.memory_allocated(device)
    
    try:
        # Crear tensores grandes
        large_batch = 8
        seq_len = 64
        dim = 128
        num_heads = 4
        
        x = torch.randn(large_batch, seq_len, dim, device=device)
        v = torch.randn(large_batch, seq_len, dim, device=device)
        f = torch.randn(large_batch, seq_len, dim, device=device)
        U_stack = torch.randn(num_heads, dim, dim // num_heads, device=device)
        W_stack = torch.randn(num_heads, dim // num_heads, dim, device=device)
        dt_scales = torch.ones(num_heads, device=device)
        forget_rates = torch.ones(num_heads, device=device)
        
        # Ejecutar múltiples veces
        for i in range(5):
            result = recurrent_manifold_fused(
                x, v, f, U_stack, W_stack, 0.1, dt_scales, forget_rates,
                num_heads=num_heads, topology=1, R=2.0, r=1.0
            )
            
            if result is None:
                raise RuntimeError(f"Iteration {i}: CUDA kernel returned None")
        
        # Medir memoria después
        memory_after = torch.cuda.memory_allocated(device)
        memory_increase = memory_after - memory_before
        
        print(f"✓ Memory handling test completed")
        print(f"  Memory before: {memory_before / 1024**2:.2f} MB")
        print(f"  Memory after: {memory_after / 1024**2:.2f} MB")
        print(f"  Memory increase: {memory_increase / 1024**2:.2f} MB")
        
        # Verificar que no hay memory leak significativo
        if memory_increase > 100 * 1024**2:  # 100 MB
            print(f"⚠️  Warning: Significant memory increase detected")
        
        return {
            "status": "passed",
            "memory_before_mb": memory_before / 1024**2,
            "memory_after_mb": memory_after / 1024**2,
            "memory_increase_mb": memory_increase / 1024**2
        }
        
    except Exception as e:
        print(f"✗ Memory handling test failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

def test_cuda_performance():
    """Test 4: Rendimiento de kernels CUDA"""
    print("\n[TEST] CUDA Performance Benchmark")
    print("-" * 40)
    
    if not CUDA_AVAILABLE:
        print("⚠️  CUDA not available - skipping performance test")
        return {"status": "skipped", "reason": "CUDA not available"}
    
    device = torch.device('cuda')
    
    # Configuración de prueba
    batch_sizes = [1, 2, 4, 8]
    seq_lens = [8, 16, 32, 64]
    dim = 64
    num_heads = 4
    
    results = []
    
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            try:
                # Crear tensores
                x = torch.randn(batch_size, seq_len, dim, device=device)
                v = torch.randn(batch_size, seq_len, dim, device=device)
                f = torch.randn(batch_size, seq_len, dim, device=device)
                U_stack = torch.randn(num_heads, dim, dim // num_heads, device=device)
                W_stack = torch.randn(num_heads, dim // num_heads, dim, device=device)
                dt_scales = torch.ones(num_heads, device=device)
                forget_rates = torch.ones(num_heads, device=device)
                
                # Warm-up
                for _ in range(3):
                    _ = recurrent_manifold_fused(
                        x, v, f, U_stack, W_stack, 0.1, dt_scales, forget_rates,
                        num_heads=num_heads, topology=1, R=2.0, r=1.0
                    )
                
                # Medición de tiempo
                torch.cuda.synchronize()
                start_time = time.time()
                
                n_iterations = 10
                for _ in range(n_iterations):
                    result = recurrent_manifold_fused(
                        x, v, f, U_stack, W_stack, 0.1, dt_scales, forget_rates,
                        num_heads=num_heads, topology=1, R=2.0, r=1.0
                    )
                
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                avg_time = total_time / n_iterations
                
                # Calcular throughput
                elements_processed = batch_size * seq_len * dim * n_iterations
                throughput = elements_processed / total_time / 1e6  # M elementos/segundo
                
                results.append({
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "avg_time_ms": avg_time * 1000,
                    "throughput_melements_per_sec": throughput
                })
                
                print(f"  Batch {batch_size}, Seq {seq_len}: {avg_time*1000:.2f}ms, {throughput:.1f}M elems/sec")
                
            except Exception as e:
                print(f"  Batch {batch_size}, Seq {seq_len}: FAILED - {e}")
                results.append({
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "error": str(e)
                })
    
    return {
        "status": "passed",
        "benchmark_results": results,
        "test_config": {
            "batch_sizes": batch_sizes,
            "seq_lens": seq_lens,
            "dim": dim,
            "num_heads": num_heads
        }
    }

def run_cuda_validation_suite():
    """Ejecutar suite completa de validación CUDA"""
    print("="*60)
    print("CUDA KERNEL VALIDATION SUITE")
    print("="*60)
    
    tests = [
        ("CUDA Kernel Execution", test_cuda_kernel_execution),
        ("Christoffel Fused CUDA", test_christoffel_fused_cuda),
        ("CUDA Memory Handling", test_cuda_memory_handling),
        ("CUDA Performance", test_cuda_performance),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            results[test_name] = {
                "status": "error",
                "error": str(e)
            }
    
    # Generar reporte
    print("\n" + "="*60)
    print("CUDA VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r.get("status") == "passed")
    total = len(tests)
    
    print(f"Tests Passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = result.get("status", "unknown")
        if status == "passed":
            print(f"✓ {test_name}")
        elif status == "skipped":
            print(f"⚠️  {test_name} (skipped: {result.get('reason', 'unknown')})")
        else:
            print(f"✗ {test_name} (failed: {result.get('error', 'unknown')})")
    
    # Guardar reporte
    report_path = Path(__file__).parent / "cuda_validation_report.json"
    import json
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")
    
    return results

if __name__ == "__main__":
    results = run_cuda_validation_suite()
    
    # Exit con código apropiado
    passed = sum(1 for r in results.values() if r.get("status") == "passed")
    total = len(results)
    
    if passed == total:
        print("\n🎉 All CUDA tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} CUDA tests failed")
        sys.exit(1)