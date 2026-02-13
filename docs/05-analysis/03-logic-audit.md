# Auditoría lógica y matemática del núcleo Manifold

## Resumen

Este documento presenta una auditoría lógica y matemática enfocada en el núcleo MLayer y su integración geométrica. Se analizan las ecuaciones implementadas, la coherencia de la lógica entre versiones (working vs old) y el comportamiento esperado de la topología tórica. El objetivo es validar consistencia teórica, no ejecutar pruebas.

## 1. Alcance y fuentes

- Núcleo MLayer y mezcla multi‑head: gfn/layers/base.py (working y old).
- Integración geométrica y operaciones CUDA: gfn/cuda/ops.py (working).
- Lógica de benchmark y criterios de convergencia: tests/benchmarks/viz/vis_gfn_superiority.py (working y old).

## 2. Modelo matemático base (núcleo MLayer)

El estado evoluciona con un sistema geodésico con fricción y fuerza externa:

- dx/dt = v
- dv/dt = F(x,u) − Γ(x)(v,v) − μ(x,u) ⊙ v

Donde:
- Γ(x)(v,v) es la contracción cuadrática en v de los símbolos de Christoffel.
- μ(x,u) es una compuerta disipativa dependiente del estado y/o fuerza.

La actualización discreta usa un integrador simpléctico (Leapfrog o variantes), con dt escalado por cabeza:

- dt_base = softplus(dt_params)
- dt_eff = dt_base ⊙ g(x)

La compuerta temporal g(x) se obtiene con RiemannianGating y está acotada en (0,1).

## 3. Compensación de tiempo y estabilidad (working vs old)

### 3.1 Escalado dinámico

En working, dt_base se limita por un rango estable:

- dt_base ← clamp(dt_base, dt_min, dt_max)

Esto evita pasos excesivos cuando softplus(dt_params) crece sin límite. En old, no existe esta cota, por lo que dt_base puede crecer y amplificar la dinámica.

Implicación lógica:
- Working introduce un control de estabilidad numérica explícito que no está presente en old.
- La ecuación implementada difiere por un operador de recorte que cambia el campo de flujo efectivo.

## 4. Fricción (compuerta termodinámica)

La fricción sigue el esquema:

- μ(x,u) = σ(W_f · φ(x) + W_i · u + b_f) · s

Donde:
- φ(x) = [sin(x), cos(x)] en topología tórica
- φ(x) = x en topología euclidiana
- s es la escala de fricción definida en el núcleo CUDA

La implementación descompone la matriz de fricción en W_f (estado) y W_i (fuerza). Esta lógica es equivalente en working y old, pero en working se prepara explícitamente el “clutch stacking” para kernels y el mezclado usa v_mix = tanh(v/100).

Implicación lógica:
- La ecuación de fricción es la misma en ambas versiones, pero la canalización en working está alineada con el kernel y con el mezclado periódico.

## 5. Topología tórica y conservación de fase

### 5.1 Mezcla periódica

En topología tórica, la mezcla multi‑head usa:

- mixer_in = [sin(x), cos(x), tanh(v/100)]
- x_next = W_x · mixer_in

### 5.2 Proyección a fase

Working aplica un wrapping suave:

- x_next ← atan2(sin(x_next), cos(x_next))

Old no realiza esta proyección tras la mezcla. En CUDA, el integrador Leapfrog usa wrap por módulo 2π en la fase interna, pero el post‑mixing en Python no re‑impone fase en old.

Implicación lógica:
- Working preserva continuidad angular tras la mezcla, alineando la representación con la topología.
- Old puede introducir discontinuidades de fase al mezclar cabezas en salida.

## 6. Geometrías disponibles (working vs old)

Working incorpora módulos de curvatura adicionales:

- HierarchicalChristoffel
- AdaptiveRankChristoffel

Old solo expone Reactive/Hyper/Toroidal/Euc/Hyp/Spherical. Esto no cambia la forma de la ecuación, pero sí cambia Γ(x) y por tanto el campo geométrico.

Implicación lógica:
- La ecuación es formalmente igual, pero el operador Γ es más expresivo en working.

## 7. Integración y lógica CUDA

En working, el Leapfrog en CUDA aplica:

- Kick‑Drift‑Kick con fricción implícita: v ← (v + h(F−Γ)) / (1 + h μ)
- Wrap tórico: x ← x mod 2π

La fricción en CUDA usa φ(x) = [sin(x), cos(x)] en topología tórica y agrega término de fuerza si W_i está presente.

Implicación lógica:
- La integración es coherente con el modelo de fricción y con la topología compacta.

## 8. Lógica del benchmark (sin ejecutar)

### 8.1 Criterios de convergencia

En working y old, el criterio real es:

- acc_threshold = 0.98
- loss_threshold = 0.2
- min_steps = 100
- patience = 20

La lógica de “hits” exige satisfacer simultáneamente acc y loss por “patience” pasos después de min_steps.

### 8.2 Desfase entre comentarios y configuración

En working existen discrepancias entre comentarios y valores reales:

- El comentario menciona base_dt = 0.05 y ajustes de fricción, pero la configuración define base_dt = 0.4.
- El encabezado impreso menciona singularity_strength y LEAPFROG_SUBSTEPS, pero la configuración efectiva mantiene valores distintos o no especificados.

Implicación lógica:
- El código de benchmark está correctamente definido, pero su documentación interna no refleja la configuración efectiva.

## 9. Hallazgos principales

- Working introduce un clamp explícito en dt_base que cambia el flujo temporal y mejora estabilidad.
- Working aplica wrapping suave atan2 tras la mezcla en torus; old no, lo que puede producir discontinuidades.
- Working habilita curvatura jerárquica/adaptativa; old no.
- La lógica de fricción y compuertas es coherente entre versiones, con diferencias de canalización y mezcla.
- El benchmark tiene coherencia interna, pero la documentación embebida no coincide con los parámetros reales.

## 10. Conclusión

La lógica matemática central es consistente con el modelo geodésico con fricción y topología tórica. Las diferencias entre working y old se concentran en estabilización temporal, preservación de fase y variedad geométrica disponible. No se ejecutaron pruebas ni se alteró el código.

## 11. Paridad CUDA vs Python (envoltura y constantes)

- Envoltura toroidal en integradores:
  - CUDA (fused Leapfrog y Heun) usa atan2(sin, cos) y luego ajusta a [0, 2π) en [device_utils.cuh](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/common/device_utils.cuh#L18-L26) y [heun_fused.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/integrators/runge_kutta/heun_fused.cu#L92-L121).
  - Python (Leapfrog puro) usa la misma envoltura suave con atan2: ver smooth_boundary_wrap y su uso en [leapfrog.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/integrators/symplectic/leapfrog.py#L58-L83) y [leapfrog.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/integrators/symplectic/leapfrog.py#L173-L178).
  - Implicación: la envoltura base es consistente; el ajuste a [0, 2π) introduce un punto de quiebre común a ambos caminos.

- Constantes de estabilidad y fricción:
  - Epsilon: Python define EPSILON_STANDARD/STRONG/SMOOTH = 1e-7 en [constants.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/constants.py#L93-L105) y CUDA usa 1e-7 en [core.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/core.py#L129-L133). Paridad confirmada.
  - Escala de fricción: 0.02 en Python [constants.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/constants.py#L74-L83) y en CUDA [core.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/core.py#L124-L128). Paridad confirmada.
  - Curvature clamp: Python usa 3.0 en [constants.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/constants.py#L56-L59) y CUDA usa 3.0 en [core.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/core.py#L132-L135). Paridad confirmada.

- Parámetros de singularidades en llamadas internas:
  - En el kernel fused Leapfrog, la llamada a Christoffel usa sing_thresh=1.0 y sing_strength=1.0 (efectivamente desactivado) dentro de [leapfrog_fused.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/integrators/symplectic/leapfrog_fused.cu#L110-L114) y [leapfrog_fused.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/integrators/symplectic/leapfrog_fused.cu#L142-L146).
  - La interfaz pública Python de christoffel_fused usa por defecto threshold=0.5 y strength=2.0: ver [ops.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/ops.py#L531-L539).
  - Implicación: en trayectorias integradas con fused Leapfrog, la amplificación por singularidades no opera; en cálculos independientes de Γ vía christoffel_fused podría operar según configuración. Diferencia controlada por diseño, pero no estricta paridad.

- Fricción por compuerta:
  - CUDA compute_friction usa características Fourier en torus y combina fuerza si hay W_input: ver [christoffel_impl.cuh](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/geometry/christoffel_impl.cuh#L148-L187).
  - Python Leapfrog (puro y fused wrapper) usa φ(x) = [sin, cos] en torus: ver [leapfrog.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/integrators/symplectic/leapfrog.py#L164-L168) y [ops.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/ops.py#L462-L469,L482-L489). Paridad lógica mantenida.

Conclusión de paridad:
- La envoltura base y las constantes de fricción/epsilons están alineadas; la principal divergencia se concentra en parámetros internos de singularidades en fused Leapfrog (desactivados en kernel) y en la ruta toroidal dedicada (ver secciones siguientes).

## 12. Inventario CUDA y rutas de ejecución reales

### 12.1 Módulos expuestos por bindings

- El módulo CUDA expone kernels de geometría e integradores en [cuda_kernels.cpp](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/cuda_kernels.cpp#L1-L199).
- Expuestos: lowrank_christoffel_fused, christoffel_backward_fused, lowrank_christoffel_with_friction, lowrank_christoffel_friction_backward, leapfrog_fused, leapfrog_backward_fused, heun_fused, heun_backward_fused, toroidal_leapfrog_fused, recurrent_manifold_fused.

### 12.2 Rutas Python activas

- La ruta pública para Leapfrog usa autograd en [autograd.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/autograd.py#L539-L603) y cae a Python si CUDA falla.
- La ruta para recurrent_manifold_fused intenta CUDA y cae a una implementación simplificada en [ops.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/ops.py#L647-L687).
- No hay wrapper público para heun_fused en ops.py ni en autograd.py, pese a existir bindings CUDA: ver ausencia en [ops.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/ops.py#L538-L610).

### 12.3 Enrutamiento toroidal

- La ruta toroidal dedicada está implementada en [toroidal_christoffel_fused.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/integrators/toroidal/toroidal_christoffel_fused.cu#L136-L265) y expuesta por [ops.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/ops.py#L328-L401).
- En el modelo, el enrutamiento toroidal está en [fusion.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/model/fusion.py#L330-L370). Si esta ruta se bloquea en entrenamiento, la ejecución cae a Python.

## 13. Hallazgos críticos de lógica y matemáticas

### 13.1 Kernel toroidal dedicado sin gradiente

- toroidal_leapfrog_fused es solo forward y no tiene autograd/backward asociado.
- En [ops.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/ops.py#L328-L401) se invoca el kernel directamente sin torch.autograd.Function, por lo que la ruta toroidal CUDA no propaga gradientes.
- Implicación: entrenamiento con ruta toroidal CUDA puede estancarse o no converger si los gradientes dependen de esta integración.

### 13.2 Kernel toroidal: fricción simplificada y parámetros ignorados

- En [toroidal_christoffel_fused.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/integrators/toroidal/toroidal_christoffel_fused.cu#L205-L257) la fricción es constante (DEFAULT_FRICTION), no usa Wf/Wi ni fuerza.
- El kernel no usa plasticity, sing_thresh, sing_strength ni compuertas de fricción, lo cual rompe paridad con la integración Python.
- La interfaz expuesta en [cuda_kernels.cpp](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/cuda_kernels.cpp#L69-L73) no acepta Wf/Wi ni parámetros de histeresis.

### 13.3 Kernel recurrent_manifold_fused: integración Euler y parámetros ignorados

- [recurrent_manifold_fused.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/integrators/recurrent_manifold_fused.cu#L1-L120) implementa Euler explícito, ignora U_stack/W_stack/num_heads y no usa Christoffel.
- En [ops.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/ops.py#L647-L687) la ruta CUDA pasa dt_scale promedio, sin usar dt_scales por capa ni forget_rates.
- Implicación: la ruta CUDA no es equivalente a la lógica de integración por capas del fallback Python.

### 13.4 Heun CUDA: singularidades/plasticidad anuladas

- [heun_fused.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/integrators/runge_kutta/heun_fused.cu#L76-L109) llama a christoffel_device con plasticity=0.0 y sing_thresh/sing_strength=1.0, lo que desactiva esas modulaciones.
- La fricción se calcula con W_forget/b_forget, pero W_input se pasa como nullptr, lo cual omite el término de fuerza.

### 13.5 Backward lowrank_christoffel alineado con el forward

- En [christoffel_impl.cuh](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/geometry/christoffel_impl.cuh#L80-L127) el forward normaliza energía por rank y aplica factor 0.1 a plasticity.
- En [lowrank_christoffel_backward.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/geometry/lowrank_christoffel_backward.cu#L41-L109) el backward replica normalización por rank, EPSILON_STANDARD y el factor 0.1 en plasticity.
- La derivada del soft_clamp se aplica con t = gamma/CURVATURE_CLAMP en [lowrank_christoffel_backward.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/geometry/lowrank_christoffel_backward.cu#L76-L80), consistente con la saturación en el forward.
- Implicación: paridad matemática entre forward/backward para la ruta lowrank_christoffel, salvo diferencias en rutas con fricción (ver sección 13.6).

### 13.6 Backward con fricción mantiene desalineaciones

- [lowrank_christoffel_friction_backward.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/geometry/lowrank_christoffel_friction_backward.cu#L53-L71) aplica la derivada de soft_clamp sobre output (Γ+μ·v), pero el clamp sólo se aplica a Γ en el forward, lo que introduce un factor extra sobre μ·v.
- El backward de fricción usa EPSILON=1e-4 y no normaliza energía por rank en [lowrank_christoffel_friction_backward.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/geometry/lowrank_christoffel_friction_backward.cu#L115-L130), divergente del forward de [christoffel_impl.cuh](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/geometry/christoffel_impl.cuh#L85-L99).
- El slope de singularidad usa 10.0 fijo en [lowrank_christoffel_friction_backward.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/geometry/lowrank_christoffel_friction_backward.cu#L132-L139), distinto a SINGULARITY_GATE_SLOPE en [core.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/core.py#L134-L136).
- Implicación: gradientes de fricción y singularidades en la ruta “with friction” no son estrictamente consistentes con el forward CUDA ni con la versión Python.

### 13.7 Límites de dimensión y buffers locales

- Heun y backward con fricción usan buffers locales de tamaño fijo (64, 128). Ver [heun_fused.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/integrators/runge_kutta/heun_fused.cu#L38-L45) y [lowrank_christoffel_friction_backward.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/geometry/lowrank_christoffel_friction_backward.cu#L53-L76).
- El kernel toroidal usa buffers fijos de 256 en [toroidal_christoffel_fused.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/integrators/toroidal/toroidal_christoffel_fused.cu#L178-L186).
- Si dim > 64 (o 2*dim > 128 en torus) hay riesgo de overflow silencioso; si dim > 256 en torus, también.
### 13.8 Gating dinámico: ruta CUDA no visible en bindings

- En [ops.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/ops.py#L630-L642) se intenta invocar gfn_cuda.dynamic_gating_fused, pero no existe un binding en [cuda_kernels.cpp](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/cuda_kernels.cpp#L121-L199).
- Implicación: incluso con CUDA disponible, dynamic_gating_fused cae siempre al fallback PyTorch, por lo que la ruta “fused” no se ejecuta.

## 14. Brechas de paridad entre CUDA y Python

- Ruta toroidal CUDA: sin backward, sin plasticidad, sin singularidades, fricción constante.
- Ruta recurrent CUDA: Euler explícito sin Christoffel ni fricción, ignora U/W por cabeza y dt_scales.
- Backward CUDA de Christoffel: paridad OK en lowrank_christoffel; desalineación persiste en lowrank_christoffel_friction.
- Heun CUDA: no está expuesto en Python, y su forward no respeta plasticidad/singularidades ni fricción por fuerza.
- Gating dinámico: la ruta CUDA no está ligada en bindings, por lo que siempre usa fallback.

## 15. Impacto esperado en entrenamiento y convergencia

- La ausencia de gradiente en toroidal_leapfrog_fused puede explicar estancamiento de entrenamiento CUDA.
- La discrepancia de backward en lowrank_christoffel_friction puede generar gradientes inconsistentes y falta de convergencia frente a la versión Python.
- La ruta recurrent CUDA ignora dinámica geométrica, lo que introduce un modelo distinto al esperado por la matemática del manifold.
