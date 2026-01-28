## Auditoría de incoherencias (toroide, pérdidas geométricas, Hamiltoniano, modelo)

Nota operativa: el indexador del proyecto no está listo en este entorno, así que el barrido se hizo con búsquedas por patrón y lectura directa de archivos.

### Qué sí está coherente (base)

- La métrica toroidal y el “wrapping” periódico están implementados de forma consistente en Python:
  - Métrica y fuerza de Christoffel: [toroidal.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/geometry/toroidal.py#L55-L148)
  - Wrapping y distancia mínima en S1: [boundaries.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/geometry/boundaries.py#L4-L27)
  - Pérdida toroidal basada en distancia angular mínima: [losses.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/losses.py#L148-L158)

### Incoherencias críticas (comportamiento distinto según ruta)

- Mezcla de cabezas (head mixing) tiene 2 implementaciones CUDA incompatibles entre sí y con la ruta Python del toroide:
  - Python (toroide) mezcla con entrada periódica `[sin(x), cos(x), v]` y reproyecta `x` a `[0, 2π)`: [base.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/layers/base.py#L308-L336)
  - CUDA “device” usada dentro del recurrente fused sí implementa toroide con `W_x` de tamaño `3*dim`: [forces.cuh](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/cuda/include/forces.cuh#L10-L33)
  - CUDA kernel “standalone” en `src/layers/head_mixing.cu` NO contempla topología, asume `W_x` de tamaño `dim*dim` y mezcla `x` linealmente (sin `sin/cos`, sin `v`): [head_mixing.cu](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/cuda/src/layers/head_mixing.cu#L14-L66)
  - Resultado: dependiendo de si el mixing ocurre dentro del kernel recurrente o por el camino Python/torch, el toroide puede perder periodicidad (o directamente no usar el “mixer” correcto).

- `head_mixing_fused` se invoca desde Python pero no existe en la API Python:
  - Llamada desde MLayer: [base.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/layers/base.py#L293-L303)
  - No hay definición en `gfn/cuda/ops.py` (solo existe `recurrent_manifold_fused`): [ops.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/cuda/ops.py#L224-L236)
  - No hay export PyBind de `launch_head_mixing_fused` en el módulo CUDA (los bindings no exponen head mixing): [cuda_kernels.cpp](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/cuda/cuda_kernels.cpp#L295-L334)
  - Resultado: el `try` falla, se usa fallback torch (que está bien funcionalmente), pero hay “code path” muerto e inconsistencia entre intención y realidad.

### Incoherencias en configuración (lo que se configura no es lo que se usa)

- Parámetros de “active inference”/singularidades se leen de rutas distintas según el camino:
  - Toroide (Python): usa `active_inference.reactive_curvature.plasticity` y `active_inference.singularities.*`: [toroidal.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/geometry/toroidal.py#L44-L55), [toroidal.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/geometry/toroidal.py#L130-L142)
  - Recurrent fused (modelo): toma `active_inference['plasticity']` y `physics_config['singularities']` (top-level), no las rutas usadas por toroide: [model.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/model.py#L298-L305)
  - Resultado: con configs “normales” (como en benchmarks), el kernel fused puede estar ejecutando con `plasticity=0.0` y singularidades desactivadas aunque estén habilitadas en la config del toroide.

### Incoherencias de gradiente (entrenamiento)

- Backward CUDA del mixing no corresponde al forward (se ignora RMSNorm en el backward):
  - Forward aplica `rmsnorm_device` al final del mixing: [forces.cuh](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/cuda/include/forces.cuh#L30-L33)
  - Backward declara explícitamente que omite RMSNorm (“Assuming Identity”): [gradients.cuh](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/cuda/include/gradients.cuh#L39-L57)
  - Resultado: gradientes aproximados/inconsistentes cuando el camino fused se usa en entrenamiento.

- En el autograd wrapper fused se corta el gradiente de `dt_scales` (y por extensión de `dt_params`):
  - En `RecurrentManifoldFusedFn.backward`, el retorno de `dt_scales` es `None`: [autograd.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/cuda/autograd.py#L399-L418)
  - Resultado: si se usa `recurrent_manifold_fused_autograd` durante training, `dt_params` (que controla `dt_scales`) no recibe gradiente por esta ruta, y la dinámica temporal aprendible queda congelada respecto a esa contribución.

### Incoherencias “de intención” (código vs comentario / rutas especiales)

- El comentario dice “Torus uses Python loop to keep gradients stable”, pero el código permite fused también en toroide (`can_fuse` no depende de `is_torus`): [model.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/model.py#L201-L210)

- En modo fractal, el kernel fused puede perder mezcla de cabezas porque `mix_x/mix_v` se extrae de `self.layers[0]` y no de `macro_manifold`:
  - Fractal envuelve `MLayer` en `macro_manifold`: [fractal.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/layers/fractal.py#L25-L43)
  - En fused, `mix_x` se toma solo si `self.layers[0]` tiene `out_proj_x`, lo cual no pasa con `FractalMLayer`: [model.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/model.py#L290-L296)
  - Resultado: con fractal habilitado, el camino fused puede comportarse distinto al camino Python aunque todo lo demás sea igual.

### Observación sobre pérdidas geométricas (escalado)

- `geodesic_regularization` tiene un “modo fused” con escalado heurístico `/1000.0` cuando recibe un tensor 1D ya agregado: [losses.py](file:///d:/ASAS/projects/tests/gfn2.6.3c/manifold/gfn/losses.py#L60-L82)
- Resultado: el peso efectivo de la regularización geométrica cambia según el camino (lista estándar vs tensor agregado), incluso manteniendo `lambda_g` fijo.

