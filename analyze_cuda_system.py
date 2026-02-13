#!/usr/bin/env python3
"""
Análisis completo del sistema CUDA - identificar problemas en kernels, bindings y configuración
"""

import torch
import sys
import os
import importlib

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def analyze_cuda_system():
    print("=== ANÁLISIS COMPLETO DEL SISTEMA CUDA ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 1. Verificar disponibilidad de módulos CUDA
    print("1. VERIFICACIÓN DE MÓDULOS CUDA:")
    try:
        from gfn.cuda import ops
        print("✓ gfn.cuda.ops importado exitosamente")
        
        # Verificar CUDA_AVAILABLE
        print(f"  CUDA_AVAILABLE: {ops.CUDA_AVAILABLE}")
        print(f"  gfn_cuda module: {ops.gfn_cuda}")
        
        if ops.CUDA_AVAILABLE:
            print("✓ Módulo CUDA compilado disponible")
            # Listar funciones disponibles
            cuda_funcs = [attr for attr in dir(ops.gfn_cuda) if not attr.startswith('_')]
            print(f"  Funciones CUDA disponibles: {cuda_funcs}")
        else:
            print("✗ Módulo CUDA no disponible - necesita compilación")
            
    except Exception as e:
        print(f"✗ Error importando ops: {e}")
    
    print()
    
    # 2. Verificar kernels específicos
    print("2. VERIFICACIÓN DE KERNELS ESPECÍFICOS:")
    
    # Verificar recurrent_manifold_fused
    try:
        if ops.CUDA_AVAILABLE and hasattr(ops.gfn_cuda, 'recurrent_manifold_fused'):
            print("✓ recurrent_manifold_fused CUDA disponible")
        else:
            print("✗ recurrent_manifold_fused CUDA no disponible")
    except:
        print("✗ recurrent_manifold_fused CUDA no disponible")
    
    # Verificar toroidal_leapfrog_fused
    try:
        if ops.CUDA_AVAILABLE and hasattr(ops.gfn_cuda, 'toroidal_leapfrog_fused'):
            print("✓ toroidal_leapfrog_fused CUDA disponible")
        else:
            print("✗ toroidal_leapfrog_fused CUDA no disponible")
    except:
        print("✗ toroidal_leapfrog_fused CUDA no disponible")
    
    print()
    
    # 3. Verificar autograd
    print("3. VERIFICACIÓN DE AUTOGRAD:")
    try:
        from gfn.cuda import autograd
        print("✓ gfn.cuda.autograd importado exitosamente")
        
        # Verificar funciones de autograd
        autograd_funcs = [attr for attr in dir(autograd) if not attr.startswith('_')]
        toroidal_funcs = [f for f in autograd_funcs if 'toroidal' in f.lower()]
        recurrent_funcs = [f for f in autograd_funcs if 'recurrent' in f.lower()]
        
        print(f"  Funciones toroidales en autograd: {toroidal_funcs}")
        print(f"  Funciones recurrentes en autograd: {recurrent_funcs}")
        
    except Exception as e:
        print(f"✗ Error importando autograd: {e}")
    
    print()
    
    # 4. Verificar problema de indexación en autograd
    print("4. ANÁLISIS DEL PROBLEMA DE INDEXACIÓN:")
    try:
        from gfn.cuda import autograd
        
        # Crear un test simple para ver el problema
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, T, D, H = 1, 5, 4, 2
        num_layers = 2
        
        x = torch.zeros(B, D, device=device)
        v = torch.ones(B, D, device=device) * 2.0
        f = torch.zeros(B, T, D, device=device)
        
        # Crear tensores con shapes correctos
        U_stack = torch.ones(num_layers * H * D, 2, device=device) * 0.001
        W_stack = torch.ones(num_layers * H * 2, D, device=device) * 0.001
        dt_scales = torch.ones(num_layers, device=device)  # Shape correcto: [num_layers]
        
        print(f"  num_layers: {num_layers}")
        print(f"  dt_scales shape: {dt_scales.shape}")
        print(f"  dt_scales values: {dt_scales}")
        
        # Verificar el rango de layer_idx
        for layer_idx in range(num_layers):
            print(f"  layer_idx={layer_idx}, dt_scales[{layer_idx}] = {dt_scales[layer_idx]}")
        
        # El problema ocurre cuando layer_idx >= dt_scales.size(0)
        if num_layers > dt_scales.size(0):
            print(f"✗ PROBLEMA: num_layers ({num_layers}) > dt_scales.size(0) ({dt_scales.size(0)})")
        else:
            print(f"✓ Indexación válida: num_layers ({num_layers}) <= dt_scales.size(0) ({dt_scales.size(0)})")
            
    except Exception as e:
        print(f"✗ Error en análisis de indexación: {e}")
    
    print()
    
    # 5. Verificar configuración de compilación
    print("5. VERIFICACIÓN DE CONFIGURACIÓN:")
    cuda_dir = os.path.join(os.path.dirname(__file__), 'gfn', 'cuda')
    setup_file = os.path.join(cuda_dir, 'setup.py')
    
    if os.path.exists(setup_file):
        print(f"✓ setup.py encontrado en: {setup_file}")
        
        # Verificar si hay archivos compilados
        build_dir = os.path.join(cuda_dir, 'build')
        if os.path.exists(build_dir):
            print(f"✓ Directorio build encontrado: {build_dir}")
            
            # Buscar archivos .pyd (Windows) o .so (Linux)
            import glob
            pyd_files = glob.glob(os.path.join(build_dir, '**', '*.pyd'), recursive=True)
            so_files = glob.glob(os.path.join(build_dir, '**', '*.so'), recursive=True)
            
            if pyd_files or so_files:
                print(f"✓ Archivos compilados encontrados:")
                for f in pyd_files + so_files:
                    print(f"    {f}")
            else:
                print("✗ No se encontraron archivos compilados")
        else:
            print("✗ Directorio build no encontrado")
    else:
        print(f"✗ setup.py no encontrado en: {cuda_dir}")
    
    print()
    
    # 6. Recomendaciones
    print("6. RECOMENDACIONES:")
    if not ops.CUDA_AVAILABLE:
        print("  - Compilar CUDA: cd gfn/cuda && python setup.py build_ext --inplace")
    else:
        print("  - CUDA está compilado, verificar implementaciones específicas")
    
    # Verificar si los kernels toroidales están implementados
    toroidal_kernel = os.path.join(cuda_dir, 'src', 'integrators', 'toroidal', 'toroidal_christoffel_fused.cu')
    if os.path.exists(toroidal_kernel):
        print(f"  ✓ Kernel toroidal encontrado: {toroidal_kernel}")
    else:
        print(f"  ✗ Kernel toroidal no encontrado: {toroidal_kernel}")
    
    print()
    print("=== RESUMEN DE PROBLEMAS ENCONTRADOS ===")
    print("1. IndexError en autograd.py: layer_idx puede exceder dt_scales.size(0)")
    print("2. Posible falta de bindings para kernels toroidales")
    print("3. Verificar si toroidal_leapfrog_fused está correctamente registrado en ops.py")

if __name__ == "__main__":
    analyze_cuda_system()