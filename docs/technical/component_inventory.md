# GFN Component Inventory (Matrix Readiness)

Este documento lista todos los componentes detectados en el core de GFN que deben entrar en la "Matrix" de benchmarks para evaluar su ROI técnico.

## 1. Geometrías (Geometries)
- **Euclidean**: Espacio plano (Baseline).
- **Toroidal**: Periódica $[-\pi, \pi]$ (XOR/Ciclos).
- **Low-Rank**: Aproximación eficiente $O(Rank^2)$.
- **Adaptive**: Métrica que escala con el gradiente.
- **Reactive**: Métrica que reacciona a la velocidad del flujo.
- **Hyperbolic**: Espacio de curvatura negativa (Jerarquías).
- **Holographic**: Almacenamiento asociativo.
- **Spherical**: Curvatura positiva.
- **Hierarchical**: Estructuras anidadas (Tree-like).

## 2. Integradores (Integrators)
- **Symplectic**: Leapfrog, Yoshida (4th order), Verlet.
- **Runge-Kutta**: RK4, Heun.

## 3. Dinámicas (Dynamics)
- **Direct**: $x_{t+1} = f(x_t, v_t, F)$.
- **Residual**: El modelo predice $\Delta v$.
- **Mix**: Combinación de Direct y Residual.
- **Gated**: Fuerza modulada por un gate físico.
- **Stochastic**: Incluye ruido browniano en la trayectoria.

## 4. Componentes Físicos (Gates & Plugins)
- **Riemannian Gating**: Adaptive $dt$ basado en curvatura.
- **Thermodynamic Layer**: Balance entropía/energía.
- **Friction Gate**: Disipación de energía para estabilidad.
- **Singularity Gate**: Evita colapsos en puntos críticos de la métrica.
- **Hysteresis Module**: Memoria de "fuerzas fantasma" (Ghost Forces).

## 5. Mixers & Aggregators (Model Core)
- **FlowMixer**: Mezclado geométrico de heads.
- **GeodesicAttention**: Atención modulada por distancia.
- **HierarchicalAggregator**: Reducción de estados en geometrías anidadas.
- **HamiltonianPooling**: Pooling que preserva el volumen del espacio de fase.

## 6. Readouts
- **Categorical**: Proyección a vocabulario.
- **Toroidal**: Enmascaramiento $(\sin, \cos)$ para periodicidad.
- **Identity**: Salida directa de coordenadas.
- **Holographic**: Readout por patrones de interferencia.
