# Plan de Implementación CUDA (GFN) — Progreso y Validación

- Fecha: 2026-02-10
- Hardware: NVIDIA GeForce GTX 1650
- Alcance: Entrenamiento 100% CUDA, kernels fusionados, paridad matemática, validación

## Objetivos
- Acelerar entrenamiento usando CUDA en toda la ruta (sin bucles Python).
- Mantener paridad matemática con la implementación Python (toroidal, singularidades, fricción).
- Proveer forward y backward analíticos donde aplique.

## Roadmap
- Fase 1: Entrenamiento 100% CUDA
  - Usar integrador leapfrog_fused_autograd en train por cabeza/capa con dt_scales y fricción.
  - Desbloquear ruta toroidal en train usando integrador Leapfrog con topology=torus.
- Fase 2: Kernel recurrente fusionado completo
  - Implementar recurrent_manifold_fused.cu con Γ/µ/topología y dt_scales por capa/cabeza.
  - Exponer bindings y usar en train/eval desde fusion.
- Fase 3: Backward toroidal dedicado
  - Añadir backward analítico para kernel toroidal y habilitar su uso en train.
- Fase 4: Paridad matemática
  - Unificar CURVATURE_CLAMP, EPSILON_STANDARD, singularidades/plasticidad en forward/backward.
  - Validar fricción backward y clamping suave.
- Fase 5: Integradores secundarios
  - Heun (RK2) paridad en torus con backward consistente.
- Fase 6: Validación
  - Gradcheck CUDA y benchmarks de producción (tiempo/it, throughput, convergencia).
- Fase 7: Routing y limpieza
  - Selección automática de kernels fusionados; fallback solo si CUDA no disponible.

## Progreso Actual
- Ruta de entrenamiento CUDA en gestor de fusión:
  - Actualizado para usar integrador Leapfrog en train por cabeza/capa [fusion.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/model/fusion.py#L378-L440).
  - Desbloqueada la rama toroidal en train (se gestiona vía integrador CUDA) [fusion.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/model/fusion.py#L350-L377).
- Autograd recurrente:
  - Sustituido placeholder con versión vectorizada por cabeza/capa [autograd.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/autograd.py#L606-L679).
- Bindings CUDA:
  - dynamic_gating_fused disponible y expuesto [cuda_kernels.cpp](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/cuda_kernels.cpp#L208-L216).
- Guardias:
  - Evitar uso del kernel toroidal dedicado con requires_grad en train [ops.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/ops.py#L334-L398).
- Compilación:
  - Extensión CUDA recompilada correctamente (leapfrog/heun forward/backward, recurrent mínimo).

## Tests y Observaciones
- Benchmark producción:
  - Training Manifold-GFN-PRODUCTION ~36–39s/it en GTX1650 tras activación de ruta CUDA en train.
  - La reducción mayor vendrá con el kernel recurrente fusionado completo (menos lanzamientos).
- Consistencia:
  - Paridad en clamps/plasticidad fricción confirmada en lowrank fricción backward [lowrank_christoffel_friction_backward.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/geometry/lowrank_christoffel_friction_backward.cu#L111-L185).
- Compilación:
  - compile_fast.bat OK; objetos generados para heun/leapfrog y recurrent mínimo.

## Próximas Acciones
- Implementar recurrent_manifold_fused.cu con:
  - Γ low‑rank y toroidal; µ con features sin/cos en torus.
  - dt_scales por capa/cabeza; topology id y R,r.
  - Binding en [cuda_kernels.cpp](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/cuda_kernels.cpp) y uso en [ops.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/ops.py).
- Añadir backward toroidal dedicado y habilitar train con kernel toroidal.
- Gradcheck CUDA (leapfrog/torus/recurrente) y mini‑benchmarks de tiempo/it.

## Trazabilidad
- Gestor de fusión actualizado [fusion.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/model/fusion.py#L350-L440)
- Autograd Leapfrog con backward CUDA [autograd.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/autograd.py#L328-L495)
- Autograd recurrente vectorizado [autograd.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/autograd.py#L606-L679)
- Kernel Leapfrog con soporte toroidal [leapfrog_fused.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/integrators/symplectic/leapfrog_fused.cu)
- Kernel recurrente actual (mínimo) [recurrent_manifold_fused.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/integrators/recurrent_manifold_fused.cu)

## Métricas a Registrar
- Tiempo por iteración (train) y throughput.
- Éxito de gradcheck en doble precisión.
- Convergencia/accuracy en benchmark paridad.

