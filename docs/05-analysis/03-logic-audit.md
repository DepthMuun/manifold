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
  - CUDA (fused Leapfrog) aplica envoltura periódica por módulo 2π: ver LeapfrogOperation forward en [ops.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/ops.py#L477-L480) y utilidad CUDA en [device_utils.cuh](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/common/device_utils.cuh#L18-L26).
  - Python (Leapfrog puro) usa envoltura suave con atan2, que preserva gradientes: ver smooth_boundary_wrap en [leapfrog.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/integrators/symplectic/leapfrog.py#L58-L83) y su uso en el drift [leapfrog.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/integrators/symplectic/leapfrog.py#L173-L178).
  - Implicación: cerca del corte de fase, fused CUDA puede producir discontinuidades de gradiente diferentes a Python. La salida final de mezcla multi‑head sí proyecta con atan2 en ambos caminos: ver [ops.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/ops.py#L618-L621) y [layers/base.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/layers/base.py#L351-L352).

- Constantes de estabilidad y fricción:
  - Epsilon: Python define EPSILON_STANDARD/STRONG/SMOOTH = 1e-7 en [constants.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/constants.py#L93-L105) y CUDA usa 1e-7 en [core.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/core.py#L129-L133). Paridad confirmada.
  - Escala de fricción: 0.02 en Python [constants.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/constants.py#L74-L83) y en CUDA [core.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/core.py#L124-L128). Paridad confirmada.
  - Curvature clamp: Python usa 3.0 en [constants.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/constants.py#L56-L59); CUDA usa 2.5 en [core.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/core.py#L134-L137). Diferencia documentada; afecta la saturación de Γ(v,v) en CUDA vs Python.

- Parámetros de singularidades en llamadas internas:
  - En el kernel fused Leapfrog, la llamada a Christoffel usa sing_thresh=1.0 y sing_strength=1.0 (efectivamente desactivado) dentro de [leapfrog_fused.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/integrators/symplectic/leapfrog_fused.cu#L110-L114) y [leapfrog_fused.cu](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/integrators/symplectic/leapfrog_fused.cu#L142-L146).
  - La interfaz pública Python de christoffel_fused usa por defecto threshold=0.5 y strength=2.0: ver [ops.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/ops.py#L531-L539).
  - Implicación: en trayectorias integradas con fused Leapfrog, la amplificación por singularidades no opera; en cálculos independientes de Γ vía christoffel_fused podría operar según configuración. Diferencia controlada por diseño, pero no estricta paridad.

- Fricción por compuerta:
  - CUDA compute_friction usa características Fourier en torus y combina fuerza si hay W_input: ver [christoffel_impl.cuh](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/src/geometry/christoffel_impl.cuh#L148-L187).
  - Python Leapfrog (puro y fused wrapper) usa φ(x) = [sin, cos] en torus: ver [leapfrog.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/integrators/symplectic/leapfrog.py#L164-L168) y [ops.py](file:///d:/ASAS/manifold_mini/manifold_working/gfn/cuda/ops.py#L462-L469,L482-L489). Paridad lógica mantenida.

Conclusión de paridad:
- La principal divergencia práctica está en la envoltura de posición durante la integración: módulo 2π (CUDA fused) vs atan2 suave (Python puro). Las constantes clave de fricción y epsilons están sincronizadas; curvature clamp difiere (2.5 vs 3.0) y la activación de singularidades en fused Leapfrog está anulada por parámetros internos.
