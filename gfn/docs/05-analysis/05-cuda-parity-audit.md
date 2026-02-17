# Auditoría de Paridad CUDA vs Python y Hoja de Ruta

Fecha: 15 de Febrero de 2026
Objetivo: Garantizar equivalencia numérica estricta entre la implementación de referencia en Python y los kernels CUDA optimizados.

## 1. Mapeo de Módulos

| Módulo Python (`gfn/`) | Archivo CUDA (`gfn/cuda/src/`) | Estado | Notas |
|------------------------|--------------------------------|--------|-------|
| `core/manifold.py` | `integrators/unified_mlayer.cu` | ⚠️ Parcial | Faltan integradores Euler/RK4. |
| `geometry/lowrank.py` | `geometry/geometry_library.cuh` | ❌ Discrepancia | Falta lógica de Plasticidad y Singularidad. |
| `geometry/toroidal.py` | `geometry/geometry_library.cuh` | ✅ Implementado | Verificado mapeo de coordenadas y métrica. |
| `integrators/symplectic/leapfrog.py` | `integrators/universal_integrator.cuh` | ⚠️ Parcial | Fricción dependiente de velocidad deshabilitada (hardcoded 0.0). |
| `integrators/runge_kutta/heun.py` | `integrators/universal_integrator.cuh` | ✅ Implementado | Lógica predictor-corrector correcta. |
| `integrators/runge_kutta/euler.py` | *No implementado* | ❌ Faltante | Kernel universal tiene enum EULER pero no implementación. |
| `layers/unified_mlayer.py` | `integrators/unified_mlayer.cu` | ✅ Implementado | Soporta Hysteresis, Gates, Holographic, Thermo. |

## 2. Auditoría de Ecuaciones y Discrepancias Matemáticas

### A. Símbolos de Christoffel (Low-Rank)
**Python (`ChristoffelOperation.forward`):**
```python
M = 1.0 + plasticity * 0.1 * tanh(E)
soft_m = sigmoid(slope * (gate - thresh))
M = M * (1.0 + (sing_strength - 1.0) * soft_m)
gamma = (h * h @ W.t) * scale * M  # <--- Modulación M aplicada
```
**CUDA (`geometry_library.cuh`):**
```cpp
// Phase 2: Active Inference (Plasticity & Singularities)
// (To be modularized...) -> VACÍO
*gamma_val = sum_gamma; // <--- Sin modulación M
```
**Acción Requerida:** Implementar Phase 2 en `geometry_library.cuh` para replicar el cálculo de `M`.

### B. Fricción Dependiente de la Velocidad
**Python:**
```python
mu = mu * (1.0 + velocity_friction_scale * v_norm)
```
**CUDA (`universal_integrator.cuh`):**
```cpp
// En universal_mlayer_kernel:
compute_friction_distributed(..., static_cast<scalar_t>(0.0), ...); // v_norm hardcoded a 0.0
```
**Acción Requerida:** Calcular `v_norm` dentro del kernel y pasarlo a `compute_friction_distributed`.

### C. Integrador Euler
**Python:**
```python
x += dt * v
v += dt * (f - gamma - friction * v)
```
**CUDA:**
Bloque `else if (Method == IntegrationMethod::EULER)` inexistente.
**Acción Requerida:** Implementar rama EULER en `universal_mlayer_kernel`.

## 3. Sincronización de Parámetros y Constantes

| Constante | Python (`CudaConstants`) | CUDA (`types.cuh`) | Estado |
|-----------|--------------------------|--------------------|--------|
| `EPSILON_STANDARD` | 1e-7 | 1e-7 | ✅ OK |
| `FRICTION_SCALE` | 0.02 | 0.02 | ✅ OK |
| `CURVATURE_CLAMP` | 3.0 | 3.0 | ✅ OK |
| `SINGULARITY_GATE_SLOPE` | 0.5 | 0.5 | ✅ OK |

## 4. Hoja de Ruta de Implementación

### Fase 1: Corrección Geométrica (Prioridad Alta)
1.  **Christoffel:** Implementar lógica de Plasticidad (`plasticity`) en `geometry_library.cuh`.
2.  **Christoffel:** Implementar lógica de Singularidad (`sing_thresh`, `sing_strength`) en `geometry_library.cuh`.
3.  **Validación:** Crear test unitario `tests/architecture/test_cuda_parity_christoffel.py` que compare outputs de Python vs CUDA con `plasticity > 0`.

### Fase 2: Corrección Física (Prioridad Media)
1.  **Fricción:** En `universal_integrator.cuh`, calcular norma de velocidad y pasarla a `compute_friction_distributed`.
2.  **Validación:** Test `tests/architecture/test_cuda_parity_friction.py` con `velocity_friction_scale > 0`.

### Fase 3: Completitud de Integradores (Prioridad Baja)
1.  **Euler:** Añadir rama EULER en `universal_integrator.cuh`.
2.  **Validación:** Test de integración simple comparando trayectorias Euler.

## 5. Estrategia de Verificación
Se implementará un script maestro `scripts/verify_cuda_parity.py` que:
1.  Instancie cada componente (Geometría, Fricción, Integrador) en Python y CUDA con los mismos pesos aleatorios.
2.  Ejecute forward pass con inputs idénticos.
3.  Reporte error máximo absoluto (debe ser `< 1e-6` para float32).
