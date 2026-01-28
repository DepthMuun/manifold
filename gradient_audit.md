---
title: "Auditoría de gradientes explosivos (Python/CUDA)"
project: "gfn2.6.3c/manifold"
date: "2026-01-28"
scope: "Entrenamiento parity ablation + recurrent_manifold_fused"
---

## Resumen ejecutivo (estado actual)

### Síntomas observados

- `tests/benchmarks/core/test_loss_optimizer_ablation.py`: NaN en loss muy temprano (step 3–4) en varias configuraciones.
- Normas de gradiente extremadamente grandes en configuraciones con Hamiltonian loss.

### Hipótesis principales (prioridad)

1. Inestabilidad numérica/energética en la dinámica (dt alto, impulso alto, singularities/plasticity fuertes) que lleva a estados/velocidades enormes y luego NaN.
2. Pérdida Hamiltoniana aplicada a `v_seq`/`x_seq` sin normalización/escala consistente, amplificando gradientes.
3. Kernel CUDA produce valores NaN/Inf por overflow (activations/exp/sigmoid o acumulaciones) o por falta de clamps en rangos.

### Estrategia de auditoría

Para cada archivo relevante, anotar:
- Qué hace en la ruta de entrenamiento.
- Puntos de riesgo para NaN/Inf o gradientes explosivos.
- Divergencias entre CPU/Python vs CUDA (si aplica).
- Acciones/cambios recomendados (mínimos y luego estructurales).

## Resultados por archivo

### [losses.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/losses.py)

- Estado: revisado parcialmente (hamiltonian_loss + geodesic_regularization).
- Hallazgos:
  - `hamiltonian_loss` originalmente asumía `velocities` como lista; en el ablation se pasa un tensor `[B, T, D]`. Eso distorsiona el cálculo (itera en batch), y puede amplificar gradientes de forma no deseada.
  - Comparación `lambda_h == 0.0` falla si `lambda_h` llega como tensor (ambigüedad / error).
  - `geodesic_regularization` tiene un camino “heurístico” con división fija `/1000.0`; puede desescalar de forma inconsistente entre setups.
- Cambios aplicados:
  - Normalización de inputs: si `velocities/states/forces` son tensores, se convierten a listas por timestep (`unbind(dim=1)`).
  - `lambda_h` tensor -> float vía `.item()`.
- Riesgo remanente:
  - La energía `0.5 * sum(g * v^2)` no tiene clamp/escala; si `v` crece por dinámica, la pérdida y gradientes crecen cuadráticamente.

### [test_loss_optimizer_ablation.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/tests/benchmarks/core/test_loss_optimizer_ablation.py)

- Estado: revisado parcialmente (config + loop + pérdidas).
- Hallazgos:
  - Configuración original era altamente inestable: `base_dt=0.4`, `impulse_scale=80`, `sing_strength=20`, `plasticity=0.2`, `dynamic_time=True`.
  - Incluso sin Hamiltonian/geodesic, la dinámica puede “explosionar” y producir NaNs en pocos pasos (lo observado en AdamW_NoAux).
  - `first_head_metric` asumía `model.layers[0].christoffels` pero el layer concreto puede ser `FractalMLayer` sin ese atributo; eso rompía la suite.
- Cambios aplicados:
  - `first_head_metric` ahora es opcional y solo se usa si existe.
  - Ajustes de estabilidad para poder diagnosticar gradientes sin NaNs inmediatos:
    - `learning_rate`: 5e-4
    - `base_dt`: 0.05
    - `impulse_scale`: 10.0
    - `sing_strength`: 3.0
    - `reactive_curvature.plasticity`: 0.05
    - `dynamic_time.enabled`: False
    - `clip_grad_norm_`: 0.5

### [ops.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/cuda/ops.py)

- Estado: revisado parcialmente (RecurrentManifoldFused autograd).
- Hallazgos:
  - `recurrent_manifold_fused(...)` devuelve `None` en CPU; esto fuerza el entrenamiento a depender de CUDA (lo cual está alineado con el objetivo, pero complica tests).
  - `RecurrentManifoldFused.forward` clona `x/v` porque el kernel muta in-place; esto es correcto para autograd, pero el estado final devuelto (`x_state/v_state`) corresponde a los clones mutados.
  - `backward` siempre llama `gfn_cuda.recurrent_manifold_backward`. Si el kernel backward tiene cualquier desbalance de escalas, el gradiente explotará sin que Python pueda “rescatarlo”.

### [cuda_kernels.cpp](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/cuda/cuda_kernels.cpp)

- Estado: revisado parcialmente (bindings recurrent_manifold_fused/backward).
- Hallazgos:
  - El error previo `grad_v_seq` no declarado se explica por mismatch de firma/args; ya aparece corregido en el archivo actual (param existe y se usa).
  - Warning `NOMINMAX` redefinido es benigno (se define por cmdline).

## Pendiente de inspección (siguiente pasada)

- Kernels CUDA:
  - `gfn/cuda/src/integrators/recurrent/recurrent_manifold_fused.cu`
  - `gfn/cuda/src/integrators/recurrent/recurrent_manifold_backward.cu`
  - `gfn/cuda/include/gradients.cuh`
  - `gfn/cuda/include/forces.cuh` (y helpers de friction/singularity/plasticity)
### [model.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/model.py)

- Estado: revisado (forward pass, stacking de pesos, escalas dt, parámetros toroidales).
- Hallazgos críticos:
  - **Weight stacking sin normalización (líneas 250-300)**: Los tensores `U_stack`, `W_stack`, `W_f_stack`, `W_i_stack`, etc. se apilan directamente sin verificar magnitudes. Pesos grandes se amplifican en la dinámica.
  - **dt_scales sin clamp (línea 323)**: `dt_scales = torch.nn.functional.softplus(layer0.dt_params)` puede producir valores muy grandes si `dt_params` no está acotado, causando pasos de integración inestables.
  - **Parámetros toroidales sin validación (líneas 332-333)**: `R_val = topo_cfg.get('R', 2.0)` y `r_val = topo_cfg.get('r', 1.0)` se pasan directamente al kernel CUDA sin verificar que `R > r > 0`. Valores inválidos pueden causar división por cero o NaN en `compute_christoffel_torus`.
  - **Manejo de tensores dummy**: Cuando faltan parámetros (ej. `W_potential_list`), se crean tensores de ceros sin verificar dimensiones compatibles.
  - **Falta de validación de entrada**: No se verifica que `x`, `v` estén en rangos válidos antes del kernel.
- Riesgo de gradientes explosivos: Alto - La combinación de pesos sin normalizar + dt_scales sin clamp + parámetros toroidales inválidos puede llevar a estados/velocidades que crecen sin control.
- Recomendaciones:
  - Agregar normalización de pesos antes de apilar (ej. `torch.nn.functional.normalize`).
  - Clamp `dt_scales` a rango seguro (ej. `[1e-4, 0.1]`).
  - Validar `R > r > 0` y lanzar error claro si no se cumple.
  - Agregar verificación de rangos para `x`, `v` antes de llamar al kernel.

### [gradients.cuh](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/cuda/include/gradients.cuh)

- Estado: revisado (compute_friction_backward, compute_christoffel_torus_backward, compute_plasticity_backward).
- Hallazgos críticos:
  - **compute_friction_backward**: Usa `sig * (1.0f - sig) * 5.0f` que puede amplificar gradientes si `sig` está en rangos extremos. El factor 5.0 es arbitrario y no está acotado.
  - **compute_christoffel_torus_backward**: Contiene división por `(R + r * cos_th + 1e-6f)` que puede ser inestable si `R ≈ r` y `cos_th ≈ -1`. También usa `scale_M * 0.05f` sin verificar que `scale_M` no sea excesivamente grande.
  - **compute_plasticity_backward**: La fórmula `g_M * dM_dE * 2.0f * v[i]` puede amplificar gradientes cuadráticamente con la velocidad. Si `v` es grande, los gradientes crecen proporcionalmente sin límite.
  - **Uso de atomicAdd sin control**: Múltiples hilos pueden acumular gradientes grandes en la misma posición, causando crecimiento exponencial.
  - **Falta de clamps en funciones trigonométricas**: `sinf`, `cosf` de ángulos grandes pueden causar pérdida de precisión.
- Riesgo de gradientes explosivos: Muy Alto - Las funciones de gradiente contienen múltiples fuentes de amplificación sin límites.
- Recomendaciones:
  - Agregar clamps a todos los factores de escala (ej. `scale_M` en `[0.1, 10.0]`).
  - Limitar el rango de velocidades antes de calcular gradientes (ej. `|v| < 10.0`).
  - Usar precisión doble para acumuladores críticos.
  - Agregar verificación de rangos para parámetros de entrada al kernel.

### [forces.cuh](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/cuda/include/forces.cuh)

- Estado: revisado (compute_friction_coeff, compute_plasticity_scale, compute_singularity_scale, apply_friction_damping, compute_christoffel_torus).
- Hallazgos críticos:
  - **compute_friction_coeff (líneas 100-149)**: Usa `sigmoidf_device(gate_activation) * 5.0f` para calcular coeficientes de fricción. El factor 5.0 es arbitrario y no está acotado, puede amplificar gradientes significativamente.
  - **compute_plasticity_scale (líneas 50-99)**: Modula la plasticidad basándose en energía cinética `1.0f + plasticity_alpha * tanhf(E_mean)`. Si `plasticity_alpha` es grande (>1.0) y la energía es alta, puede causar crecimiento exponencial de la escala M.
  - **compute_singularity_scale (líneas 300-349)**: Calcula escalamiento de singularidad usando `sigmoidf_device(stiffness * (potential - threshold))` con `stiffness = 20.0f`. Este valor alto puede crear transiciones muy abruptas y amplificar gradientes.
  - **apply_friction_damping (líneas 150-199)**: Aplica amortiguamiento exponencial `v[i] *= expf(-mu[i] * dt)`. Si `mu[i] * dt` es grande, puede causar underflow o valores extremadamente pequeños.
  - **compute_christoffel_torus (líneas 200-249)**: Contiene división por `(R + r * cos_th)` con protección mínima `1e-6f`. Si `R ≈ r` y `cos_th ≈ -1`, la división puede ser inestable.
  - **Falta de validación de rangos**: Ninguna función verifica que los valores de entrada estén en rangos seguros antes de aplicar operaciones no lineales.
  - **Factores mágicos sin justificación**: Los valores 5.0, 20.0, 0.05f aparecen sin documentación sobre sus límites o justificación.
- Riesgo de gradientes explosivos: Muy Alto - Múltiples funciones contienen amplificación sin límites y factores arbitrarios.
- Recomendaciones:
  - Reemplazar factores mágicos con parámetros configurables con rangos seguros.
  - Agregar clamps a todos los valores de salida (ej. `mu_out[i] = clamp(sigmoid(...) * max_friction, 0.1f, 2.0f)`).
  - Limitar el rango de `plasticity_alpha` a valores < 1.0 para evitar crecimiento exponencial.
  - Agregar validación de rangos para parámetros de entrada (R, r, dt, etc.).
  - Documentar los límites teóricos de cada función y agregar aserciones en debug mode.

### [optim.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/optim.py)

- Estado: revisado (RiemannianAdam y ManifoldSGD).
- Hallazgos críticos:
  - **Weight decay condicional para torus (línea 77)**: El código excluye weight decay para parámetros con `retraction == 'torus'`, lo cual es correcto para mantener la topología toroidal.
  - **Retracción toroidal sin validación de rangos (líneas 150-153)**: `p.data.copy_(torch.remainder(phase, 2.0 * math.pi))` no verifica que `phase` esté en rangos razonables antes de aplicar remainder, puede causar pérdida de precisión si `phase` es muy grande.
  - **Cayley retraction sin manejo de errores robusto (líneas 163-170)**: El try-catch para SVD solo imprime un pass silencioso, no hay logging ni manejo de la condición de error, puede causar comportamiento inconsistente.
  - **Max_norm sin clamp dinámico (línea 87)**: `max_norm = 10.0` es fijo y no se adapta según la magnitud de los gradientes o la escala del problema.
  - **Bias correction sin límite de steps (líneas 104-105)**: `bias_correction1 = 1 - beta1 ** state['step']` puede causar división por cero o valores muy pequeños si `state['step']` es grande.
  - **Learning rate sin validación (línea 72)**: `lr` se convierte a float sin verificar que esté en rangos positivos razonables.
  - **Falta de gradient clipping**: No hay mecanismo para limitar la magnitud de los gradientes antes de aplicar la actualización.
- Riesgo de gradientes explosivos: Medio - Aunque el optimizador tiene mecanismos de retracción, la falta de validación de rangos y gradient clipping puede permitir que gradientes grandes pasen sin control.
- Recomendaciones:
  - Agregar validación de rangos para `phase` en retracción toroidal (ej. `phase = clamp(phase, -1000*math.pi, 1000*math.pi)`).
  - Implementar gradient clipping antes de aplicar la actualización (ej. `torch.nn.utils.clip_grad_norm_`).
  - Agregar logging y manejo robusto de errores en SVD de Cayley retraction.
  - Hacer `max_norm` adaptable o agregar validación de rangos.
  - Agregar límite máximo para `state['step']` o usar epsilon adicional en bias correction.
  - Validar que `lr > 0` y esté en rangos razonables (ej. `[1e-6, 1.0]`).

## Resumen de riesgos identificados

| Archivo | Riesgo de Gradientes Explosivos | Factores Principales |
|---------|--------------------------------|----------------------|
| [model.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/model.py) | **Alto** | Weight stacking sin normalización, dt_scales sin clamp, parámetros toroidales sin validación |
| [gradients.cuh](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/cuda/include/gradients.cuh) | **Muy Alto** | Amplificación arbitraria (5.0f), división inestable, gradientes cuadráticos con velocidad |
| [forces.cuh](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/cuda/include/forces.cuh) | **Muy Alto** | Factores mágicos sin límites (5.0f, 20.0f), plasticidad exponencial, fricción sin clamp |
| [optim.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/optim.py) | **Medio** | Falta gradient clipping, validación de rangos insuficiente, manejo de errores débil |
| [losses.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/losses.py) | **Medio-Alto** | Hamiltonian loss sin normalización, energía cinética sin clamp |

## Recomendaciones prioritarias para estabilizar el entrenamiento

1. **Implementar clamps inmediatos** en todas las funciones de CUDA:
n   - Limitar `scale_M` a `[0.1, 10.0]`
   - Limitar `mu[i]` a `[0.1, 2.0]`
   - Limitar `plasticity_alpha` a `< 1.0`

2. **Agregar normalización de pesos** en `model.py` antes de apilar
3. **Implementar gradient clipping** en el optimizador
4. **Validar parámetros toroidales** (`R > r > 0`) con aserciones
5. **Agregar clamps a dt_scales** en `[1e-4, 0.1]`
6. **Limitar rangos de velocidad** antes de calcular gradientes

## Próximos pasos sugeridos

1. Implementar los clamps y validaciones identificados
2. Ejecutar `test_loss_optimizer_ablation.py` con los cambios
3. Verificar convergencia en `benchmark_language_streaming.py`
4. Analizar resultados del ablation test para determinar optimal loss configuration

