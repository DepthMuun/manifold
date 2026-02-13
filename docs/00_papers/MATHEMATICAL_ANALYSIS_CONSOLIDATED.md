# Análisis Matemático Consolidado — Corpus Completo de Geodesic Flow Networks

**Autor del análisis:** Asistente de Ingeniería  
**Fecha:** Febrero 2026  
**Fuente:** 31 archivos `.md` en `docs/00_papers/`

---

## Tabla de Contenidos

1. [GFN_PAPER — Documento Fundacional](#gfn_paper)
2. [Paper 00 — Foundational Hypothesis](#paper-00)
3. [Paper 01 — Hyper-Torus](#paper-01)
4. [Paper 02 — Reactive Geometry](#paper-02)
5. [Paper 03 — Thermodynamic Gating](#paper-03)
6. [Paper 04 — Symplectic Attention](#paper-04)
7. [Paper 05 — Holographic Latent Space](#paper-05)
8. [Paper 06a — Gauge Theory Semantic Consistency](#paper-06a)
9. [Paper 06b — Semantic Event Horizons](#paper-06b)
10. [Paper 07 — Recursive Manifold Resolvers](#paper-07)
11. [Paper 08 — Physicality of Confusion](#paper-08)
12. [Paper 10 — Implicit Neural Fields](#paper-10)
13. [Paper 11 — Exotic Symplectic Integrators](#paper-11)
14. [Paper 12 — Parallel Manifold Scans](#paper-12)
15. [Paper 13 — Geometric Coupling Flows](#paper-13)
16. [Paper 14 — Piecewise Riemannian Varieties](#paper-14)
17. [Paper 15 — Thermodynamic Geometry](#paper-15)
18. [Paper 16 — Stochastic Differential Geometry](#paper-16)
19. [Paper 17 — Ricci Flow Adaptive Geometry](#paper-17)
20. [Paper 18 — AdS/CFT Holographic Extensions](#paper-18)
21. [Paper 19 — Hysteresis Semantic Memory](#paper-19)
22. [Paper 21 — Adaptive Rank Geometry](#paper-21)
23. [Paper 22 — Hierarchical Multi-Scale Geometry](#paper-22)
24. [Paper 23 — Riemannian Adam Optimizer](#paper-23)
25. [Paper 24 — Fractal Manifold Tunneling](#paper-24)
26. [Paper 25 — Hamiltonian Pooling](#paper-25)
27. [Paper 26 — Curiosity-Driven Entropy Exploration](#paper-26)
28. [Paper 27 — Parallel Manifold Scan Layer](#paper-27)
29. [Paper 28 — Noether Symmetry Regularization](#paper-28)
30. [Documentos Auxiliares](#auxiliares)

---

<a id="gfn_paper"></a>
## GFN_PAPER — Geodesic Flow Networks: Documento Fundacional

### Estado de fase
$$S_t = (x_t, v_t) \in \mathcal{M} \times T\mathcal{M}$$

### Dinámica continua (ecuación geodésica forzada con disipación)
1. **Cinemática:** $\dot{x} = v$
2. **Dinámica:** $\dot{v} = F_{\text{ext}}(u) - \Gamma(x, v) - \mu(x, u) \odot v$

### Curvatura low-rank (Christoffel aprendidos)
$$\Gamma^k_{ij}(x) \approx \sum_{r=1}^R W_{kr}(U_{ir} \cdot U_{jr})$$

**Forma práctica:**
$$\Gamma(x, v) = W \cdot \left(\frac{(U^T v)^2}{1 + \|U^T v\|}\right), \quad U, W \in \mathbb{R}^{d \times R}$$

### Clutch (disipación termódinámica)
$$\mu(x, u) = \sigma(W_f \cdot \phi(x) + W_i \cdot \text{Embed}(u) + b_f)$$

### Integrador Leapfrog (Kick-Drift-Kick)
1. **Kick:** $v_{n+1/2} = \dfrac{v_n + \frac{\Delta t}{2}(F_{\text{ext}} - \Gamma)}{1 + \frac{\Delta t}{2}\mu}$
2. **Drift:** $x_{n+1} = x_n + \Delta t \cdot v_{n+1/2}$
3. **Kick:** simétrico en posición nueva

### Topología toroidal
- Coordenadas envueltas en $[0, 2\pi)$
- Features de Fourier $[\sin(x), \cos(x)]$ para continuidad métrica

---

<a id="paper-00"></a>
## Paper 00 — Foundational Hypothesis (Geodesic Flow Hypothesis)

### Dualismo Inercia-Acumulación
$$\frac{Dv^k}{dt} = \dot{v}^k + \Gamma^k_{ij}(x) v^i v^j = F^k_{\text{ext}}(u_t) - \mu^k(x, u_t) v^k$$

### Readout (modo no-holográfico)
$$\hat{y} = W_{\text{out}} \cdot x + b_{\text{out}}$$

### Clutch termodinámico
- $\mu \to 0$: fase conservativa (memoria Hamiltoniana)
- $\mu \to \infty$: fase disipativa (reseteo de estado)

---

<a id="paper-01"></a>
## Paper 01 — Hyper-Torus $\mathbb{T}^n$

### Geometría del toro
$$\mathcal{M} = \mathbb{T}^n = \underbrace{S^1 \times S^1 \times \cdots \times S^1}_{n \text{ veces}}$$

### Métrica Riemanniana del toro
$$ds^2 = (R + r\cos\theta)^2 d\phi^2 + r^2 d\theta^2$$

donde $R$ = radio mayor, $r$ = radio menor.

### Símbolos de Christoffel del toro (no nulos)
$$\Gamma^\phi_{\phi\theta} = \Gamma^\phi_{\theta\phi} = \frac{-r\sin\theta}{R + r\cos\theta}$$
$$\Gamma^\theta_{\phi\phi} = \frac{(R + r\cos\theta)\sin\theta}{r}$$

### Ecuación geodésica con forcing y disipación
$$\ddot{\phi} + 2\Gamma^\phi_{\phi\theta}\dot{\phi}\dot{\theta} = F^\phi - \mu^\phi \dot{\phi}$$
$$\ddot{\theta} + \Gamma^\theta_{\phi\phi}\dot{\phi}^2 = F^\theta - \mu^\theta \dot{\theta}$$

### Condiciones de contorno periódicas
$$x^i \leftarrow x^i \mod 2\pi$$

### Distancia toroidal
$$d_{\text{torus}}(\theta_1, \theta_2) = \min(|\theta_1 - \theta_2|, 2\pi - |\theta_1 - \theta_2|)$$

### Integrador Leapfrog con fricción implícita
$$v_{n+1/2} = \frac{v_n + \frac{\Delta t}{2}(F - \Gamma(x_n, v_n))}{1 + \frac{\Delta t}{2}\mu}$$

---

<a id="paper-02"></a>
## Paper 02 — Reactive Geometry

### Operador de curvatura low-rank
$$\Gamma^k_{ij}(x) v^i v^j \approx U^k_a(x)\, W^a_{ij}(x)\, v^i v^j$$

### Modulación reactiva de la curvatura (Plasticidad)
$$g_{\text{eff},ij}(x, v) = g_{\text{static},ij}(x) \cdot (1 + \alpha \cdot \text{KE}(v))$$
$$\text{KE}(v) = \frac{1}{2} g_{ij}(x) v^i v^j$$

### Gate de disipación
$$\mu(x, u) = \sigma(W_\mu \cdot \phi(x) + b_\mu)$$

### Potencial de singularidad
$$V(x) = A \cdot \exp\left(-\frac{\|x - x_{\text{center}}\|^2}{2\sigma^2}\right)$$

### Aceleración efectiva
$$a^k = F^k_{\text{ext}} - \Gamma^k_{\text{eff},ij}(x, v) v^i v^j - \mu^k v^k$$

---

<a id="paper-03"></a>
## Paper 03 — Thermodynamic Gating (Clutch)

### Coeficiente de fricción dependiente del estado e input
$$\mu(x, u) = \sigma(W_f \cdot \phi(x) + W_i \cdot \text{Embed}(u) + b_f)$$

### Gate temporal dinámico
$$g(x) = \sigma(W_g \cdot \phi(x) + b_g)$$
$$\Delta t_{\text{eff}} = g(x) \cdot \Delta t_{\text{base}}$$

### Leapfrog conformal simpléctica con fricción implícita
$$v_{n+1/2} = \frac{v_n + \frac{\Delta t}{2}(F_{\text{ext}} - \Gamma^k_{ij}(x_n) v_n^i v_n^j)}{1 + \frac{\Delta t}{2}\mu(x_n, u)}$$
$$x_{n+1} = x_n + \Delta t \cdot v_{n+1/2}$$
$$v_{n+1} = \frac{v_{n+1/2} + \frac{\Delta t}{2}(F_{\text{ext}} - \Gamma^k_{ij}(x_{n+1}) v_{n+1/2}^i v_{n+1/2}^j)}{1 + \frac{\Delta t}{2}\mu(x_{n+1}, u)}$$

### Features periódicos (toro)
$$\phi(x) = [\sin(x^1), \cos(x^1), \sin(x^2), \cos(x^2), \ldots]$$

---

<a id="paper-04"></a>
## Paper 04 — Symplectic Attention

### Ecuación geodésica covariante con fuerza aprendida
$$\frac{Dv^k}{dt} = \dot{v}^k + \Gamma^k_{ij}(x) v^i v^j = F^k_\theta(u_t) - (\mu_\theta(x, u_t) v)^k$$

### Christoffel symbols (definición estándar)
$$\Gamma^k_{ij} = \frac{1}{2} g^{kl}\left(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij}\right)$$

### Multi-head: manifolds independientes
Cada cabeza $h$ tiene su propia $(x_h, v_h)$ y evoluciona:
$$\dot{v}_h = F_h(u) - \Gamma_h(x_h, v_h) - \mu_h v_h$$

### Inferencia con memoria constante: $O(1)$ en longitud de secuencia

---

<a id="paper-05"></a>
## Paper 05 — Holographic Latent Space

### Readout holográfico
$$\hat{y} = x \quad \text{(el estado latente ES la predicción)}$$

### Loss holográfico
$$\mathcal{L} = d_{\mathcal{M}}(x_T, y_{\text{target}})$$

### Paso simpléctico con fricción implícita
$$v_{n+1/2} = \frac{v_n + \frac{\Delta t}{2}(F - \Gamma^k_{ij}(x_n) v^i_n v^j_n)}{1 + \frac{\Delta t}{2}\mu}$$

### Topología periódica para tareas cíclicas
$$x \mapsto x \mod 2\pi$$

---

<a id="paper-06a"></a>
## Paper 06a — Gauge Theory Semantic Consistency

### Fibrado principal
$$P(\mathcal{M}, G) \to \mathcal{M}$$

### Conexión de gauge
$$A_\mu: T_x\mathcal{M} \to \mathfrak{g}$$

### Derivada covariante gauge
$$D_\mu \psi = \partial_\mu \psi + A_\mu \psi$$

### Tensor de fuerza de campo (curvatura del gauge)
$$F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu + [A_\mu, A_\nu]$$

### Christoffel corregidos por gauge
$$\tilde{\Gamma}^k_{ij}(x) = \Gamma^k_{ij}(x) + \delta\Gamma^k_{ij}(A)$$

### Loss gauge-invariante
$$\mathcal{L}_{\text{gauge}} = \|F_{\mu\nu}\|^2 + \lambda \|D_\mu \psi - \partial_\mu \psi\|^2$$

---

<a id="paper-06b"></a>
## Paper 06b — Semantic Event Horizons

### Conexión efectiva (singularidad semántica)
$$\Gamma^k_{\text{eff},ij}(x) = \Gamma^k_{\text{LC},ij}(x) \cdot (1 + \Psi(x))$$

### Potencial de singularidad
$$\Psi(x) = A \cdot \exp\left(-\frac{\|x - x_c\|^2}{2\sigma^2}\right)$$

### Aceleración con singularidad
$$a^k(x, v) = -\Gamma^k_{\text{eff},ij}(x) v^i v^j$$

### Energía cinética (tensor métrico)
$$E = \frac{1}{2} g_{ij}(x) v^i v^j$$

---

<a id="paper-07"></a>
## Paper 07 — Recursive Manifold Resolvers

### Densidad del manifold (proxy de curvatura)
$$D(x) = \mathbb{E}\left[\|\Gamma^k_{ij}(x) v^i v^j\|^2\right]$$

### Controlador neural de paso adaptativo
$$\Delta t_{\text{adapt}} = \Delta t_{\text{base}} \cdot \sigma(W_{\Delta t} \cdot [x, v, D(x)] + b)$$

### Integración recursiva (Adaptive Mesh Refinement analogy)
Si $D(x) > \text{threshold}$:
$$\Delta t_{\text{micro}} = \frac{\Delta t_{\text{macro}}}{N_{\text{sub}}}, \quad N_{\text{sub}} \text{ micro-pasos}$$

---

<a id="paper-08"></a>
## Paper 08 — Physicality of Confusion (Reactive Plasticity + Finsler)

### Escalar de plasticidad
$$\Phi(x, v) = \alpha \cdot \frac{1}{d}\sum_{i=1}^d (v^i)^2$$

### Métrica efectiva (dependiente de velocidad → Finsler)
$$g_{\text{eff},ij}(x, v) = g_{\text{static},ij}(x) \cdot (1 + \Phi(x, v))$$

### Conexión reactiva (derivada de la métrica efectiva)
$$\Gamma^k_{\text{eff},ij}(x, v) = \frac{1}{2} g^{kl}_{\text{eff}} \left(\partial_i g_{\text{eff},jl} + \partial_j g_{\text{eff},il} - \partial_l g_{\text{eff},ij}\right)$$

### Potencial de singularidad (cuantificación de incertidumbre)
$$V(x) = A \cdot \exp\left(-\frac{\|x - x_c\|^2}{2\sigma^2}\right)$$

---

<a id="paper-10"></a>
## Paper 10 — Implicit Neural Fields (INF)

### Campo neural implícito con activaciones periódicas (SIREN)
$$\Psi(c) = W_n(\sin(\omega_{n-1} W_{n-1}(\cdots \sin(\omega_0 W_0 c + b_0)\cdots) + b_{n-1})) + b_n$$

### Métrica inducida por el campo (pullback)
$$g_{kl}(c) = \sum_{\alpha=1}^D \frac{\partial \Psi^\alpha}{\partial c^k} \frac{\partial \Psi^\alpha}{\partial c^l} = J^T J$$

### Christoffel derivados de la métrica inducida
$$\Gamma^k_{ij}(c) = \frac{1}{2} g^{kl}(c)\left(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij}\right)$$

### Readout implícito (distancia metric-aware)
$$p(s_k | c) = \text{softmax}\left(-\frac{\|c - c_k\|^2_g}{T}\right)$$

---

<a id="paper-11"></a>
## Paper 11 — Exotic Symplectic Integrators

### Aceleración geodésica (base de todos los integradores)
$$a^k(x) = -\Gamma^k_{ij}(x) v^i v^j$$

### Composición de Yoshida (orden 4 a partir de orden 2)
$$S_4(\Delta t) = S_2(w_1 \Delta t) \circ S_2(w_0 \Delta t) \circ S_2(w_1 \Delta t)$$
$$w_1 = \frac{1}{2 - 2^{1/3}}, \quad w_0 = 1 - 2w_1$$

### Forest-Ruth (4to orden)
$$c_1 = c_4 = \frac{1}{2(2 - 2^{1/3})}, \quad c_2 = c_3 = \frac{1 - 2^{1/3}}{2(2 - 2^{1/3})}$$
$$d_1 = d_3 = \frac{1}{2 - 2^{1/3}}, \quad d_2 = \frac{-2^{1/3}}{2 - 2^{1/3}}$$

### Omelyan PEFRL
$$\xi = 0.1786178958448091, \quad \lambda = -0.2123418310626054$$
$$\chi = -0.6626458266981849 \times 10^{-1}$$

### Forma simpléctica conservada
$$\omega = dp_i \wedge dq^i, \quad \frac{d\omega}{dt} = 0$$

---

<a id="paper-12"></a>
## Paper 12 — Parallel Manifold Scans

### Reformulación como sistema LTV afín
$$v^k_t = A^{kj}_t v^j_{t-1} + B^k_t$$
$$x^i_t = x^i_{t-1} + v^i_t \Delta t$$

### Operador de retención (derivado de Christoffel)
$$A^{kj}_t \approx \delta^{kj} - \Delta t \cdot \Gamma^k_{ij}(x_t) v^i_t$$

### Scan paralelo asociativo ($O(\log N)$)
$$v_t = \left(\prod_{i=t}^{1} A_i\right) v_0 + \sum_{j=1}^t \left(\prod_{i=t}^{j+1} A_i\right) B_j$$

### Composición de propagadores
$$(A_2, B_2) \otimes (A_1, B_1) = (A_2 A_1,\; A_2 B_1 + B_2)$$

---

<a id="paper-13"></a>
## Paper 13 — Geometric Coupling Flows

### Flujo de acoplamiento simpléctico (tipo Normalizing Flow)
$$\hat{x}^i = x^i + G^i_\theta(v), \quad \hat{v}^k = v^k + H^k_\theta(\hat{x})$$

### Aceleración con Christoffel explícitos
$$a^k(x, v) = F^k - \Gamma^k_{ij}(x) v^i v^j$$

### Jacobiano unitario (preservación de volumen)
$$\det\frac{\partial(\hat{x}, \hat{v})}{\partial(x, v)} = 1$$

por la estructura triangular del coupling.

### Drift neural
$$\dot{x}^i = v^i + G^i_\theta(v)$$

---

<a id="paper-14"></a>
## Paper 14 — Piecewise Riemannian Varieties

### La Paradoja Runge-Kutta
Integradores de alto orden (RK4) fallan en variedades con discontinuidades métricas porque requieren evaluaciones en puntos intermedios donde la métrica cambia abruptamente.

### Ecuación geodésica covariante
$$\frac{Dv^k}{dt} = \dot{v}^k + \Gamma^k_{ij}(x) v^i v^j = 0$$

### Métrica efectiva con plasticidad
$$g_{\text{eff},ij}(x, v) = g_{\text{base},ij}(x) \cdot \Phi(v, x)$$

### Principio de Realismo Local
Leapfrog (orden 2) da mejor estabilidad que RK4 en singularidades por adherir a la "realidad local" — solo usa información del punto actual.

---

<a id="paper-15"></a>
## Paper 15 — Thermodynamic Geometry

### Métrica dependiente de temperatura
$$g_{ij}(x, T) = g_{\text{base},ij}(x) \cdot f(T, x)$$

### Christoffel termodinámicos
$$\Gamma^k_{ij}(x, T) = \frac{1}{2} g^{kl}(x, T)\left(\partial_i g_{jl}(x, T) + \partial_j g_{il}(x, T) - \partial_l g_{ij}(x, T)\right)$$

### Energía libre
$$F = E - TS$$
$$E = \frac{1}{2} g_{ij}(x, T) v^i v^j, \quad S \approx \frac{1}{2}\log\det\Sigma(x)$$

### Gradiente de energía libre
$$\nabla_k F = \nabla_k E - T \nabla_k S$$

### Ecuación geodésica termodinámica
$$\frac{Dv^k}{dt} = \dot{v}^k + \Gamma^k_{ij}(x, T) v^i v^j = F^k - \mu^k v^k$$

---

<a id="paper-16"></a>
## Paper 16 — Stochastic Differential Geometry

### Langevin en variedades (Itô)
$$dv^i = \left(-\Gamma^i_{jk}(x) v^j v^k - \mu v^i + F^i + \sigma^2 \Gamma^i_{jk}(x) g^{jk}(x)\right) dt + \sigma\, dW^i$$

El término $\sigma^2 \Gamma^i_{jk} g^{jk}$ es la **corrección de Itô** necesaria en variedades curvadas.

### Ecuación de Fokker-Planck
$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial v^i}\left[a^i(x, v)\, p\right] + \frac{\sigma^2}{2}\frac{\partial^2 p}{\partial v^i \partial v^i}$$

### Christoffel estocásticos (métrica modulada por difusión)
$$g_{ij}(x, \sigma) = g_{\text{base},ij}(x) \cdot (1 + \beta\sigma^2)$$

### Relación fluctuación-disipación
$$\sigma^2 = 2\mu k_B T$$

### Hamiltoniano
$$H = \frac{1}{2} g_{ij}(x) v^i v^j + V(x)$$

---

<a id="paper-17"></a>
## Paper 17 — Ricci Flow Adaptive Geometry

### Flujo de Ricci (evolución de la métrica)
$$\frac{\partial g_{ij}}{\partial t} = -2R_{ij}$$

### Flujo de Ricci normalizado (preserva volumen)
$$\frac{\partial g_{ij}}{\partial t} = -2R_{ij} + \frac{2}{n} r\, g_{ij}$$

donde $r = \frac{1}{\text{Vol}} \int R\, dV$ es la curvatura escalar promedio.

### Tensor de Riemann
$$R^i_{jkl} = \partial_k \Gamma^i_{jl} - \partial_l \Gamma^i_{jk} + \Gamma^i_{mk}\Gamma^m_{jl} - \Gamma^i_{ml}\Gamma^m_{jk}$$

### Tensor de Ricci (contracción)
$$R_{ij} = R^k_{ikj} = g^{kl} R_{kilj}$$

### Christoffel de la métrica evolutiva
$$\Gamma^i_{jk}(x, t) = \frac{1}{2} g^{il}(x, t)\left(\partial_j g_{kl} + \partial_k g_{jl} - \partial_l g_{jk}\right)$$

---

<a id="paper-18"></a>
## Paper 18 — AdS/CFT Holographic Extensions

### Métrica AdS (Anti-de Sitter)
$$ds^2 = \frac{L^2}{z^2}\left(dz^2 + \eta_{\mu\nu} dx^\mu dx^\nu\right)$$

### Entropía de entrelazamiento (Ryu-Takayanagi)
$$S_A = \frac{\text{Area}(\gamma_A)}{4G_N}$$

### Flujo RG como profundidad
$$z \sim \frac{1}{\Lambda} \quad \text{(escala de energía inversa)}$$

### Dualidad bulk/boundary
- **Bulk:** manifold de alta dimensión (latente profundo)
- **Boundary:** manifold de baja dimensión (representación final)

### Christoffel en el bulk
$$\Gamma^k_{ij}(x, z) = \text{derivados de la métrica AdS}$$

---

<a id="paper-19"></a>
## Paper 19 — Hysteresis Semantic Memory (Ghost Force)

### Estado de histéresis (acumulación de trayectoria)
$$h_{t+1} = \gamma \cdot h_t + (1 - \gamma) \cdot \Phi(x_t, v_t)$$

donde $\gamma \in [0, 1]$ es el coeficiente de decaimiento y $\Phi$ es el mapeo de plasticidad.

### Fuerza Fantasma (Ghost Force)
$$F^k_{\text{ghost}}(h) = W_{\text{ghost}} \cdot h$$

### Christoffel dependientes de trayectoria
$$\Gamma^k_{\text{hyst},ij}(x, h) = \Gamma^k_{\text{base},ij}(x) + \delta\Gamma^k_{ij}(h)$$

### Ecuación geodésica con histéresis
$$\frac{Dv^k}{dt} = F^k_{\text{ext}} + F^k_{\text{ghost}}(h) - \Gamma^k_{\text{hyst},ij}(x, h) v^i v^j - \mu^k v^k$$

---

<a id="paper-21"></a>
## Paper 21 — Adaptive Rank Christoffel Symbol Decomposition (AR-CSD)

### Factorización low-rank adaptativa
$$\Gamma_\theta(v)^k = U(v)^k_a \tilde{z}^a, \quad z^a = W(v)^a_{ij} v^i v^j$$

### Slicing de factores por rango efectivo
$$U(v) = U_{\text{full}}[:, :r(v)], \quad W(v) = W_{\text{full}}[:, :r(v)]$$

### Red predictora de complejidad
$$\mathcal{C}_\phi(v) = \sigma(W_2 \cdot \text{ReLU}(W_1 v + b_1) + b_2)$$

### Ratio de rango
$$\rho(v) = 0.1 + 0.9 \cdot \mathcal{C}_\phi(v)$$

### Rango efectivo
$$r(v) = \text{clamp}\left(\lfloor \rho(v) \cdot R_{\max} \rfloor, r_{\min}, R_{\max}\right)$$

### Normalización de escala
$$\tilde{z}_a = z_a \cdot \frac{1}{1 + \|z\|_2 + \epsilon}$$

---

<a id="paper-22"></a>
## Paper 22 — Hierarchical Multi-Scale Christoffel Symbolic Mixture (HM-CSM)

### Mezcla multi-escala
$$\Gamma_{\text{HM}}(v) = \sum_{i=1}^k w_i \cdot \Gamma_i(v)$$

### Pesos de mezcla (softmax)
$$w_i = \frac{\exp(\alpha_i)}{\sum_{j=1}^k \exp(\alpha_j)}$$

### Módulos por escala (rango diferente cada uno)
$$\Gamma_i(v) = U_i W_i^T v, \quad U_i, W_i \in \mathbb{R}^{d \times r_i}$$

con $r_1 < r_2 < \cdots < r_k$ (escala gruesa a fina).

### Escalas recomendadas
$$\{r_i\} = \{8, 16, 32\} \quad \text{(progresión geométrica)}$$

---

<a id="paper-23"></a>
## Paper 23 — Riemannian Adam (R-Adam)

### Regla de actualización Riemanniana
$$p_{t+1} = R_{p_t}\left(-\eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t + \epsilon}}\right)$$

### Momentos con transporte vectorial
$$m_t \leftarrow \mathcal{T}_{p_t \to p_{t+1}}(m_t)$$
$$v_t \leftarrow \mathcal{T}_{p_t \to p_{t+1}}(v_t)$$

### Retracción normalizada (esfera)
$$R_p(v) = \frac{p + v}{\|p + v\|}$$

### Retracción toroidal (periódica)
$$R_p(v) = \text{atan2}(\sin(p + v), \cos(p + v))$$

### Retracción de Cayley (ortogonal)
$$R_p(v) = \left(I - \tfrac{1}{2}v\right)\left(I + \tfrac{1}{2}v\right)^{-1} p$$

### Adam estándar (Euclídeo, para referencia)
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$

---

<a id="paper-24"></a>
## Paper 24 — Fractal Manifold Tunneling (FMT)

### Blending macro-micro
$$x_{\text{final}} = x_{\text{macro}} + \alpha(x_{\text{micro}} - x_{\text{macro}})$$

equivalente a: $x_{\text{final}} = (1 - \alpha) x_{\text{macro}} + \alpha\, x_{\text{micro}}$

### Norma de Christoffel (proxy de curvatura)
$$R = \|\Gamma\| = \sqrt{\sum_i \|\Gamma_i\|_F^2}$$

### Tunnel gate (sigmoid)
$$\alpha = \sigma\left((R - \theta) \cdot \kappa\right)$$

donde $\theta$ = umbral de curvatura, $\kappa$ = sensibilidad.

### Dimensión fractal efectiva
$$D_f = D_{\text{macro}} + (D_{\text{micro}} - D_{\text{macro}}) \cdot p$$

### Costo computacional
$$\text{Cost} = \text{Cost}_{\text{macro}} + p \cdot \text{Cost}_{\text{micro}}$$

---

<a id="paper-25"></a>
## Paper 25 — Hamiltonian Pooling (H-Pool)

### Hamiltoniano total por token
$$H_i = K_i + U_i$$

### Energía cinética (métrica Riemanniana)
$$K_i = \frac{1}{2} v_i^T g\, v_i = \frac{1}{2} \sum_j g_{jj} v_{ij}^2 \quad \text{(diagonal)}$$

### Energía potencial (oscilador armónico)
$$U_i = \frac{1}{2} \|x_i\|^2$$

### Pesos de atención Boltzmann
$$\alpha_i = \frac{\exp(-H_i / T)}{\sum_j \exp(-H_j / T)}$$

### Agregación ponderada
$$x_{\text{agg}} = \sum_i \alpha_i x_i, \quad v_{\text{agg}} = \sum_i \alpha_i v_i$$

### Ecuaciones de Hamilton (referencia)
$$\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}$$

### Potencial generalizado (Mahalanobis)
$$U(x) = \frac{1}{2}(x - \mu)^T \Sigma^{-1}(x - \mu)$$

### Temperature annealing (coseno)
$$T_t = T_{\min} + \frac{1}{2}(T_{\max} - T_{\min})\left(1 + \cos\left(\frac{\pi t}{T_{\text{total}}}\right)\right)$$

---

<a id="paper-26"></a>
## Paper 26 — Curiosity-Driven Entropy Exploration (CDEE)

### Entropía diferencial (Gaussiana $d$-dimensional)
$$h = \frac{1}{2}\log\left((2\pi e)^d \det\Sigma\right)$$

### Proxy de entropía (diagonal)
$$S(V) = \frac{1}{2}\sum_{j=1}^d \log(\sigma_j(V)^2 + \epsilon)$$

### Log-det general
$$S(V) = \frac{1}{2}\log\det(\Sigma(V) + \epsilon I)$$

### Loss de curiosidad
$$\mathcal{L}_{\text{curiosity}} = -\lambda_c \cdot S(V)$$

### Objetivo combinado
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \mathcal{L}_{\text{curiosity}}$$

### Ecuación geodésica (referencia contextual)
$$\frac{Dv^k}{dt} = -\Gamma^k_{ij}(x) v^i v^j$$

---

<a id="paper-27"></a>
## Paper 27 — Parallel Manifold Scan Layer (P-MLayer)

### LTV linearizada de la dinámica geodésica
$$v_t = A_t v_{t-1} + B_t$$

### Predicción del factor de decaimiento
$$A_t = \sigma(W_A F_t + b_A)$$

### Modulación de input
$$B_t = W_B F_t + b_B$$

### Solución por scan paralelo
$$v_t = \left(\prod_{i=t}^{1} A_i\right) v_0 + \sum_{j=1}^t \left(\prod_{i=t}^{j+1} A_i\right) B_j$$

### Integración de posición
$$x_t = x_{t-1} + v_t \cdot \Delta t$$

### Escalas de tiempo multi-escala (wormholes)
$$\Delta t_t = \Delta t_{\text{learned}} \cdot \Delta t_{\text{base}}$$
$$s_i = 1.5^i \quad \text{(progresión exponencial por cabeza)}$$

### Linealización desde Christoffel (Apéndice B)
$$U W^T(v \odot v) \approx U W^T(v_0 \odot v_0) + J(v_0)(v - v_0)$$
$$\Rightarrow \frac{dv}{dt} = -D \cdot v + F \quad \text{(forma LTV)}$$

---

<a id="paper-28"></a>
## Paper 28 — Noether Symmetry Regularization (NSR)

### Grupo de simetría (cabezas isoméricas)
$$\mathcal{S}_k = \{\sigma: \{1, \ldots, k\} \to \{1, \ldots, k\} \mid \sigma \text{ es permutación}\}$$

### Condición de simetría
$$\Gamma_{h_i}(v) = \Gamma_{h_{\sigma(i)}}(v), \quad \forall \sigma \in \mathcal{S}_k$$

### Carga de Noether
$$Q_i(v) = \Gamma_{h_i}(v) - \Gamma_{\text{ref}}(v)$$

### Loss de simetría NSR
$$\mathcal{L}_{\text{NSR}} = \frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}} \frac{1}{|g|}\sum_{i,j \in g} \mathbb{E}_v\left[\|\Gamma_i(v) - \Gamma_j(v)\|^2\right]$$

### Forma simplificada (pairwise)
$$\mathcal{L}_{\text{NSR}} = \lambda_n \cdot \frac{1}{N_{\text{pairs}}} \sum_{(i,j) \in \text{pairs}} \text{MSE}(\Gamma_i, \Gamma_j)$$

### Christoffel por cabeza
$$\Gamma_i(v) = U_i W_i^T v$$

### Simetrías alternativas
- **Escalado:** $\Gamma_i(v) = s_i \cdot \Gamma_{\text{ref}}(s_i^{-1} v)$
- **Rotación:** $\Gamma_i(Rv) = R\,\Gamma_{\text{ref}}(v)$

---

<a id="auxiliares"></a>
## Documentos Auxiliares

### `posibles_problemas.md` — Revisión de Errores Matemáticos

Puntos clave identificados:

1. **Notación tensorial inconsistente:** Muchos papers usan $\Gamma(v, v)$ sin especificar índices. La forma correcta es $\Gamma^k_{ij}(x) v^i v^j$.
2. **Compatibilidad métrica:** Multiplicar Christoffel por escalares $\Gamma_{\text{eff}} = \Gamma \cdot (1 + \Phi)$ viola compatibilidad con métrica. Se debe modular $g_{\text{eff}}$ y rederivir $\Gamma$.
3. **Definida positividad de $g$:** Parametrizar como $g = UU^T$ o $g = \exp(S)$ para garantizar.
4. **Simplecticidad ≠ preservación de volumen:** $\det J = 1$ es necesario pero NO suficiente. Se necesita $J^T \Omega J = \Omega$.
5. **Langevin en variedades:** Falta corrección de Itô ($\sigma^2 \Gamma^i_{jk} g^{jk}$).
6. **Entropía diagonal vs. full:** El proxy $\sum \log \sigma_j$ solo es correcto si $\Sigma$ es diagonal.

### `flujodeiaarreglando.md` — Log de Correcciones

Registro cronológico de todas las correcciones tensioriales aplicadas a papers 00–17, incluyendo:
- Covariant derivative completa $\frac{Dv^k}{dt} = \dot{v}^k + \Gamma^k_{ij} v^i v^j$
- Reformulación de conexiones reactivas vía métrica efectiva
- Adición de fórmulas explícitas de Christoffel en cada paper
- Corrección de índices en Fokker-Planck y Langevin

---

## Ecuaciones Universales del Framework GFN

### La Ecuación Maestra
$$\frac{Dv^k}{dt} = \dot{v}^k + \Gamma^k_{ij}(x) v^i v^j = F^k_{\text{ext}}(u) - \mu^k(x, u) v^k + F^k_{\text{extra}}$$

donde $F^k_{\text{extra}}$ puede incluir: ghost force, stochastic noise, gauge corrections, etc.

### El Integrador Universal (Leapfrog con fricción implícita)
$$v_{n+1/2} = \frac{v_n + \frac{\Delta t}{2}\left(F - \Gamma^k_{ij}(x_n) v_n^i v_n^j\right)}{1 + \frac{\Delta t}{2}\mu}$$
$$x_{n+1} = x_n + \Delta t \cdot v_{n+1/2} \quad (\text{mod } 2\pi \text{ si toro})$$

### Los Christoffel Low-Rank (forma práctica)
$$(\Gamma(v, v))^k = U^k_a(x)\, W^a_{ij}(x)\, v^i v^j \approx U^k_a \cdot (W^T v)_a^2 \cdot \frac{1}{1 + \|W^T v\|}$$
