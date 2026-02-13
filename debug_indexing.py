#!/usr/bin/env python3
"""
Debug del problema de indexación en autograd.py
"""

import torch
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfn.cuda.autograd import recurrent_manifold_fused_autograd

def debug_indexing_issue():
    print("=== DEBUGGING INDEXING ISSUE ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reproducir el problema exacto
    B, T, D, H = 1, 10, 4, 2
    num_layers = 2
    dt = 0.1
    
    print(f"B={B}, T={T}, D={D}, H={H}, num_layers={num_layers}")
    
    # Inputs
    x = torch.zeros(B, D, device=device, requires_grad=True)
    v = torch.ones(B, D, device=device, requires_grad=True) * 5.0
    f = torch.zeros(B, T, D, device=device, requires_grad=True)
    
    # Parameters
    rank = 2
    U_stack = torch.ones(num_layers * H * D, rank, device=device) * 0.001
    W_stack = torch.ones(num_layers * H * rank, D, device=device) * 0.001
    
    # El problema: dt_scales shape
    dt_scales = torch.ones(num_layers, device=device)  # Shape: [2]
    forget_rates = torch.zeros(num_layers, device=device)
    
    print(f"U_stack shape: {U_stack.shape}")
    print(f"W_stack shape: {W_stack.shape}")
    print(f"dt_scales shape: {dt_scales.shape}")
    print(f"dt_scales values: {dt_scales}")
    
    # Verificar el cálculo de num_layers desde U_stack
    calculated_num_layers = U_stack.shape[0] // H
    print(f"calculated_num_layers: {calculated_num_layers}")
    print(f"expected num_layers: {num_layers}")
    
    # Verificar el rango real del loop
    print(f"Layer indices that will be accessed: {list(range(num_layers))}")
    print(f"dt_scales indices available: {list(range(dt_scales.size(0)))}")
    
    # El error debe estar en algún otro lugar...
    # Quizás el problema es que num_layers no es lo que esperamos
    
    # Vamos a inspeccionar el código más cuidadosamente
    print("\n=== INSPECCIONANDO CÓDIGO ===")
    
    # Revisar el reshape de U_stack
    head_dim = D // H
    U_reshaped = U_stack.view(num_layers, H, head_dim, -1)
    W_reshaped = W_stack.view(num_layers, H, head_dim, -1).permute(0, 1, 3, 2)
    
    print(f"head_dim: {head_dim}")
    print(f"U_reshaped shape: {U_reshaped.shape}")
    print(f"W_reshaped shape: {W_reshaped.shape}")
    
    # Ahora vemos si el loop realmente accede a índices válidos
    for layer_idx in range(num_layers):
        print(f"layer_idx={layer_idx}: accediendo dt_scales[{layer_idx}] = {dt_scales[layer_idx]}")
    
    print("\n=== INTENTANDO EJECUCIÓN ===")
    
    try:
        result = recurrent_manifold_fused_autograd(
            x=x, v=v, f=f,
            U_stack=U_stack, W_stack=W_stack,
            dt=dt, dt_scales=dt_scales, forget_rates=forget_rates,
            num_heads=H, topology=0,
            plasticity=0.0, sing_thresh=0.5, sing_strength=2.0,
            mix_x=None, mix_v=None, Wf=None, Wi=None, bf=None, Wp=None, bp=None
        )
        print("✓ Éxito!")
        print(f"Result shapes: x={result[0].shape}, v={result[1].shape}, seq={result[2].shape}")
        
    except Exception as e:
        print(f"✗ Falló: {e}")
        import traceback
        traceback.print_exc()
        
        # Inspeccionar el stack trace para ver exactamente dónde falla
        print("\n=== ANÁLISIS DEL ERROR ===")
        # El error debe estar en la línea 647 de autograd.py
        # Vamos a ver qué valores tiene en ese momento
        
        # Crear un tensor con más elementos para ver si es eso
        print("Intentando con dt_scales más grande...")
        dt_scales_large = torch.ones(num_layers + 5, device=device)
        
        try:
            result = recurrent_manifold_fused_autograd(
                x=x, v=v, f=f,
                U_stack=U_stack, W_stack=W_stack,
                dt=dt, dt_scales=dt_scales_large, forget_rates=forget_rates,
                num_heads=H, topology=0,
                plasticity=0.0, sing_thresh=0.5, sing_strength=2.0,
                mix_x=None, mix_v=None, Wf=None, Wi=None, bf=None, Wp=None, bp=None
            )
            print("✓ Funcionó con dt_scales más grande!")
            print("El problema es que el código está accediendo a un índice que no existe")
            
        except Exception as e2:
            print(f"✗ Aún falla con dt_scales más grande: {e2}")

if __name__ == "__main__":
    debug_indexing_issue()