#ifndef GFN_CUDA_CHRISTOFFEL_IMPL_CUH
#define GFN_CUDA_CHRISTOFFEL_IMPL_CUH

#include "../common/types.cuh"
#include "../common/device_utils.cuh"
#include "../common/math_utils.cuh"

namespace gfn {
namespace cuda {

// ============================================================================
// Core Christoffel Device Function
// ============================================================================

/**
 * Normalize Christoffel structure to enforce symmetry.
 * This is CRITICAL for Python-CUDA parity.
 *
 * Implements: gamma_sym[i,j] = 0.5 * (gamma[i,j] + gamma[j,i])
 * This ensures Gamma^k_ij approx Gamma^k_ji numerically, which is required
 * for torsion-free connections.
 *
 * @param gamma Input/Output Christoffel symbols [dim x dim]
 * @param dim Dimension of manifold
 */
template <typename T>
GFN_DEVICE void normalize_christoffel_structure(T* gamma, int dim) {
    // Average with transpose to enforce approximate symmetry
    for (int i = 0; i < dim; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            T avg = static_cast<T>(0.5) * (gamma[i * dim + j] + gamma[j * dim + i]);
            gamma[i * dim + j] = avg;
            gamma[j * dim + i] = avg;
        }
    }
}

/**
 * Core Christoffel symbol computation using low-rank decomposition.
 * 
 * Computes: Γ(v,v) = Σ_r (h_r^2 * W_r) * S * M
 * where:
 *   h = U^T * v (projection to rank-R space)
 *   S = 1 / (1 + ||h||)  (stabilization factor)
 *   M = modulation from plasticity and singularities
 * 
 * @param v Velocity vector [dim]
 * @param U Low-rank matrix U [dim x rank]
 * @param W Low-rank matrix W [dim x rank]
 * @param x Position vector [dim] (optional, for friction/singularities)
 * @param V_w Potential weights [dim] (optional, for singularities)
 * @param dim Dimension of manifold
 * @param rank Rank of decomposition
 * @param plasticity Plasticity coefficient (energy-dependent curvature)
 * @param sing_thresh Singularity threshold
 * @param sing_strength Singularity strength multiplier
 * @param topology Topology type (EUCLIDEAN or TORUS)
 * @param R Toroidal major radius
 * @param r Toroidal minor radius
 * @param gamma Output Christoffel force [dim]
 */
template <typename T>
GFN_DEVICE void christoffel_device(
    const T* v,
    const T* U,
    const T* W,
    const T* x,
    const T* V_w,
    int dim,
    int rank,
    T plasticity,
    T sing_thresh,
    T sing_strength,
    Topology topology,
    T R,
    T r,
    T* gamma
) {
    if (topology == Topology::TORUS && x != nullptr && V_w == nullptr) {
        for (int i = 0; i < dim; ++i) gamma[i] = static_cast<T>(0);
        for (int i = 0; i < dim - 1; i += 2) {
            T th = x[i];
            T v_th = v[i];
            T v_ph = v[i + 1];
            T denom = fmax(R + r * cos(th), static_cast<T>(CLAMP_MIN_STRONG<T>));
            T term_th = denom * sin(th) / (r + static_cast<T>(EPSILON_SMOOTH<T>));
            gamma[i] = term_th * (v_ph * v_ph);
            T term_ph = -(r * sin(th)) / (denom + static_cast<T>(EPSILON_SMOOTH<T>));
            gamma[i + 1] = static_cast<T>(2) * term_ph * v_ph * v_th;
        }
        for (int i = 0; i < dim; ++i) {
            gamma[i] = soft_clamp<T>(gamma[i] * static_cast<T>(TOROIDAL_CURVATURE_SCALE<T>), static_cast<T>(CURVATURE_CLAMP<T>));
        }
        return;
    }
    T h[64];
    T h_sq[64];
    for (int i = 0; i < rank; ++i) {
        T sum = static_cast<T>(0);
        for (int j = 0; j < dim; ++j) {
            sum += U[j * rank + i] * v[j];
        }
        h[i] = sum;
    }
    T energy = static_cast<T>(0);
    for (int i = 0; i < rank; ++i) {
        energy += h[i] * h[i];
    }
    // AUDIT FIX: Normalize energy by rank to match Python ops.py
    if (rank > 0) energy /= static_cast<T>(rank);
    
    T norm_val = sqrt(energy);
    // AUDIT FIX: Use EPSILON_STRONG to match Python lowrank.py line 139
    T S = static_cast<T>(1) / (static_cast<T>(1) + norm_val + static_cast<T>(EPSILON_STRONG<T>));
    T M = static_cast<T>(1);
    if (plasticity != static_cast<T>(0)) {
        T v_energy = static_cast<T>(0);
        for (int i = 0; i < dim; ++i) {
            v_energy += v[i] * v[i];
        }
        v_energy /= static_cast<T>(dim);
        // AUDIT FIX: Add 0.1 factor to match Python ops.py
        M *= (static_cast<T>(1) + plasticity * static_cast<T>(0.1) * tanh(v_energy));
    }
    if (x != nullptr && V_w != nullptr) {
        T pot = static_cast<T>(0);
        if (topology == Topology::TORUS) {
            for (int i = 0; i < dim; ++i) {
                pot += sin(x[i]) * V_w[i];
            }
        } else {
            for (int i = 0; i < dim; ++i) {
                pot += x[i] * V_w[i];
            }
        }
        T gate = sigmoid<T>(pot);
        T soft_m = sigmoid<T>(static_cast<T>(SINGULARITY_GATE_SLOPE<T>) * (gate - sing_thresh));
        M *= (static_cast<T>(1) + (sing_strength - static_cast<T>(1)) * soft_m);
    }
    for (int i = 0; i < rank; ++i) {
        h_sq[i] = h[i] * h[i] * S * M;
    }
    for (int i = 0; i < dim; ++i) {
        T sum = static_cast<T>(0);
        for (int j = 0; j < rank; ++j) {
            sum += W[i * rank + j] * h_sq[j];
        }
        gamma[i] = sum;
    }

    for (int i = 0; i < dim; ++i) {
        gamma[i] = soft_clamp<T>(gamma[i], static_cast<T>(CURVATURE_CLAMP<T>));
    }
}

// ============================================================================
// Friction Computation
// ============================================================================

/**
 * Compute friction coefficient from position and force.
 * 
 * μ = sigmoid(W_f * features + b_f + W_i * force) * FRICTION_SCALE
 * 
 * @param x Position vector [dim]
 * @param force External force [dim] (optional)
 * @param W_forget Forget gate weights [dim x feature_dim]
 * @param b_forget Forget gate bias [dim]
 * @param W_input Input gate weights [dim x dim] (optional)
 * @param dim Dimension
 * @param topology Topology type
 * @param velocity_friction_scale Velocity friction scaling factor
 * @param v_norm_val Pre-computed velocity norm (optional, if available)
 * @param friction Output friction coefficients [dim]
 */
template <typename T>
GFN_DEVICE void compute_friction(
    const T* x,
    const T* force,
    const T* W_forget,
    const T* b_forget,
    const T* W_input,
    int dim,
    Topology topology,
    T velocity_friction_scale,
    T v_norm_val,
    T* friction
) {
    T features[128]; // Max 2*dim for Fourier features
    int feature_dim = dim;
    
    // Compute features based on topology
    if (topology == Topology::TORUS) {
        // Fourier features: [sin(x), cos(x)]
        compute_fourier_features<T>(features, x, dim);
        feature_dim = 2 * dim;
    } else {
        // Direct features
        vector_copy<T>(features, x, dim);
    }
    
    // Compute gate activation: W_f * features + b_f
    for (int i = 0; i < dim; ++i) {
        T gate_val = b_forget[i];
        for (int j = 0; j < feature_dim; ++j) {
            gate_val += W_forget[i * feature_dim + j] * features[j];
        }
        
        // Add force component if available
        if (force != nullptr && W_input != nullptr) {
            for (int j = 0; j < dim; ++j) {
                gate_val += W_input[i * dim + j] * force[j];
            }
        }
        
        T base_friction = sigmoid<T>(gate_val) * static_cast<T>(FRICTION_SCALE<T>);
        
        // Apply velocity scaling: mu = mu_base * (1 + scale * |v|)
        if (velocity_friction_scale > static_cast<T>(0)) {
            // Normalized by sqrt(dim) to be dimension-agnostic
            T v_scale = v_norm_val / (sqrt(static_cast<T>(dim)) + static_cast<T>(EPSILON_SMOOTH<T>));
            friction[i] = base_friction * (static_cast<T>(1) + velocity_friction_scale * v_scale);
        } else {
            friction[i] = base_friction;
        }
    }
}

// ============================================================================
// Combined Christoffel + Friction
// ============================================================================

/**
 * Compute combined Christoffel force with friction.
 * Output: Γ(v,v) + μ(x,F) * v
 */
template <typename T>
GFN_DEVICE void christoffel_with_friction(
    const T* v,
    const T* U,
    const T* W,
    const T* x,
    const T* V_w,
    const T* force,
    const T* W_forget,
    const T* b_forget,
    const T* W_input,
    int dim,
    int rank,
    T plasticity,
    T sing_thresh,
    T sing_strength,
    Topology topology,
    T R,
    T r,
    T velocity_friction_scale,
    T* output
) {
    T gamma[64];
    T friction[64];
    
    // Compute Christoffel force
    christoffel_device<T>(
        v, U, W, x, V_w, dim, rank,
        plasticity, sing_thresh, sing_strength,
        topology, R, r, gamma
    );
    
    // Compute friction if position is available
    if (x != nullptr && W_forget != nullptr && b_forget != nullptr) {
        // Calculate v_norm for friction scaling
        T v_norm_val = static_cast<T>(0);
        if (velocity_friction_scale > static_cast<T>(0)) {
            for(int i=0; i<dim; ++i) v_norm_val += v[i]*v[i];
            v_norm_val = sqrt(v_norm_val);
        }

        compute_friction<T>(
            x, force, W_forget, b_forget, W_input,
            dim, topology, velocity_friction_scale, v_norm_val, friction
        );
        
        // Add friction term: gamma + μ * v
        for (int i = 0; i < dim; ++i) {
            output[i] = gamma[i] + friction[i] * v[i];
        }
    } else {
        // No friction, just copy gamma
        vector_copy<T>(output, gamma, dim);
    }
}

// ============================================================================
// Separate Christoffel and Friction (for implicit integration)
// ============================================================================

/**
 * Compute Christoffel and friction separately.
 * Used for implicit integration schemes (e.g., Leapfrog).
 */
template <typename T>
GFN_DEVICE void christoffel_friction_separate(
    const T* v,
    const T* U,
    const T* W,
    const T* x,
    const T* V_w,
    const T* force,
    const T* W_forget,
    const T* b_forget,
    const T* W_input,
    int dim,
    int rank,
    T plasticity,
    T sing_thresh,
    T sing_strength,
    Topology topology,
    T R,
    T r,
    T velocity_friction_scale,
    T* gamma,
    T* friction
) {
    // Compute Christoffel force
    christoffel_device<T>(
        v, U, W, x, V_w, dim, rank,
        plasticity, sing_thresh, sing_strength,
        topology, R, r, gamma
    );
    
    // Compute friction if position is available
    if (x != nullptr && W_forget != nullptr && b_forget != nullptr) {
        // Calculate v_norm for friction scaling
        T v_norm_val = static_cast<T>(0);
        if (velocity_friction_scale > static_cast<T>(0)) {
            for(int i=0; i<dim; ++i) v_norm_val += v[i]*v[i];
            v_norm_val = sqrt(v_norm_val);
        }

        compute_friction<T>(
            x, force, W_forget, b_forget, W_input,
            dim, topology, velocity_friction_scale, v_norm_val, friction
        );
    } else {
        // No friction
        vector_zero<T>(friction, dim);
    }
}

/**
 * Backward pass for the Christoffel symbol computation.
 * 
 * Computes gradients: dL/dv, dL/dU, dL/dW, dL/dx.
 * 
 * @param grad_out Grad output from Loss [dim]
 * @param gamma Forward output Γ(v,v) [dim] (needed for soft_clamp derivative)
 * @param v Forward velocity vector [dim]
 * @param U Forward matrix U [dim x rank]
 * @param W Forward matrix W [dim x rank]
 * @param x Forward position vector [dim]
 * @param V_w Potential weights [dim]
 * @param dim Dimension
 * @param rank Rank
 * @param plasticity Plasticity coefficient
 * @param topology Topology type
 * @param grad_v Output gradient w.r.t v [dim]
 * @param grad_U Output gradient w.r.t U [dim x rank]
 * @param grad_W Output gradient w.r.t W [dim x rank]
 * @param grad_x Output gradient w.r.t x [dim] (optional)
 */
template <typename T>
GFN_DEVICE void christoffel_backward_device(
    const T* grad_out,
    const T* gamma,
    const T* v,
    const T* U,
    const T* W,
    const T* x,
    const T* V_w,
    int dim,
    int rank,
    T plasticity,
    T sing_thresh,
    T sing_strength,
    Topology topology,
    T R,
    T r,
    T* grad_v,
    T* grad_U,
    T* grad_W,
    T* grad_x,
    T* grad_V_w = nullptr
) {
    // Thread-local storage (max rank 64, max dim 128 supported)
    T h[64];
    T grad_h[64];
    T grad_q[64];  // Gradient w.r.t h^2
    
    // 1. Re-compute forward state h = U^T * v and modulate factor M
    T h_energy = static_cast<T>(0);
    for (int i = 0; i < rank; ++i) {
        T sum = static_cast<T>(0);
        for (int j = 0; j < dim; ++j) {
            sum += U[j * rank + i] * v[j];
        }
        h[i] = sum;
        h_energy += sum * sum;
    }
    if (rank > 0) {
        h_energy /= static_cast<T>(rank);
    }
    
    T norm_val = sqrt(h_energy);
    // AUDIT FIX: Use EPSILON_STRONG to match forward pass (line 114)
    T S = static_cast<T>(1) / (static_cast<T>(1) + norm_val + static_cast<T>(EPSILON_STRONG<T>));
    
    // 1b. Re-compute Modulation M
    T M_plas = static_cast<T>(1);
    T v_energy = static_cast<T>(0);
    if (plasticity != static_cast<T>(0)) {
        for (int i = 0; i < dim; ++i) v_energy += v[i] * v[i];
        v_energy /= static_cast<T>(dim);
        M_plas = (static_cast<T>(1) + plasticity * static_cast<T>(0.1) * tanh(v_energy));
    }
    
    T M_sing = static_cast<T>(1);
    T gate = static_cast<T>(0), soft_m = static_cast<T>(0);
    if (x != nullptr && V_w != nullptr) {
        T pot = static_cast<T>(0);
        if (topology == Topology::TORUS) {
            for (int i = 0; i < dim; ++i) pot += sin(x[i]) * V_w[i];
        } else {
            for (int i = 0; i < dim; ++i) pot += x[i] * V_w[i];
        }
        gate = sigmoid<T>(pot);
        soft_m = sigmoid<T>(static_cast<T>(SINGULARITY_GATE_SLOPE<T>) * (gate - sing_thresh));
        M_sing = (static_cast<T>(1) + (sing_strength - static_cast<T>(1)) * soft_m);
    }
    T M = M_plas * M_sing;

    // 2. Correct grad_out for soft_clamp derivative
    // d_clamped / d_raw = 1 - (clamped / 20)^2
    T grad_raw[128];
    for (int i = 0; i < dim; ++i) {
        T t = gamma[i] / static_cast<T>(CURVATURE_CLAMP<T>);
        grad_raw[i] = grad_out[i] * (static_cast<T>(1) - t * t);
    }
    
    // 3. Gradient w.r.t W and intermediate grad_q
    for (int j = 0; j < rank; ++j) {
        grad_q[j] = static_cast<T>(0);
        T q_base = h[j] * h[j] * S * M;
        for (int i = 0; i < dim; ++i) {
            grad_W[i * rank + j] += grad_raw[i] * q_base;
            grad_q[j] += W[i * rank + j] * grad_raw[i];
        }
    }
    
    // 4. Gradient w.r.t h (accounting for stabilization S)
    T sum_grad_q_h_sq = static_cast<T>(0);
    for (int i = 0; i < rank; ++i) {
        sum_grad_q_h_sq += grad_q[i] * h[i] * h[i];
    }

    // AUDIT FIX: Divide by (norm_val * rank) to match Python autograd.py
    // Python: denom = norm * max(1, rank), scale = M * S * S / denom
    T S_sq_M_norm = (norm_val > EPSILON_STANDARD<T> && rank > 0) ?
        (M * S * S / (norm_val * static_cast<T>(rank))) : static_cast<T>(0);
    T two_S_M = static_cast<T>(2) * S * M;
    
    for (int i = 0; i < rank; ++i) {
        grad_h[i] = grad_q[i] * h[i] * two_S_M - sum_grad_q_h_sq * S_sq_M_norm * h[i];
    }
    
    // 5. Gradient w.r.t U and v (projection part)
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < rank; ++j) {
            grad_U[i * rank + j] += v[i] * grad_h[j];
            grad_v[i] += U[i * rank + j] * grad_h[j];
        }
    }
    
    // 6. Plasticity contribution to grad_v
    if (plasticity != static_cast<T>(0)) {
        // dL/dM_plas = sum_i grad_q[i] * h[i]^2 * S * M_sing
        T dL_dM_plas = sum_grad_q_h_sq * S * M_sing;
        T tanh_v = tanh(v_energy);
        T sech_sq = static_cast<T>(1) - tanh_v * tanh_v;
        T factor = dL_dM_plas * (plasticity * static_cast<T>(0.1)) * sech_sq * (static_cast<T>(2) / static_cast<T>(dim));
        for (int i = 0; i < dim; ++i) {
            grad_v[i] += factor * v[i];
        }
    }
    
    // 7. Singularity contribution to grad_x and grad_V_w
    if ((grad_x != nullptr || grad_V_w != nullptr) && x != nullptr && V_w != nullptr) {
        T dL_dM_sing = sum_grad_q_h_sq * S * M_plas;
        T dM_dsoft = (sing_strength - static_cast<T>(1));
        T dsoft_dgate = static_cast<T>(SINGULARITY_GATE_SLOPE<T>) * soft_m * (static_cast<T>(1) - soft_m);
        T dgate_dpot = gate * (static_cast<T>(1) - gate);
        T factor = dL_dM_sing * dM_dsoft * dsoft_dgate * dgate_dpot;
        
        for (int i = 0; i < dim; ++i) {
            if (grad_x != nullptr) {
                T dpot_dxi = (topology == Topology::TORUS) ? cos(x[i]) * V_w[i] : V_w[i];
                grad_x[i] += factor * dpot_dxi;
            }
            if (grad_V_w != nullptr) {
                T dpot_dVwi = (topology == Topology::TORUS) ? sin(x[i]) : x[i];
                grad_V_w[i] += factor * dpot_dVwi;
            }
        }
    }
}

/**
 * Backward pass for friction computation.
 * 
 * Computes gradients: dL/dW_forget, dL/db_forget, dL/dx.
 */
template <typename T>
GFN_DEVICE void friction_backward_device(
    const T* grad_out,
    const T* x,
    const T* force,
    const T* W_forget,
    const T* b_forget,
    const T* W_input,
    int dim,
    Topology topology,
    T* grad_W_forget,
    T* grad_b_forget,
    T* grad_W_input,
    T* grad_x,
    T* grad_force
) {
    T features[128];
    int feature_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
    
    // Re-compute features
    if (topology == Topology::TORUS) {
        compute_fourier_features<T>(features, x, dim);
    } else {
        vector_copy<T>(features, x, dim);
    }
    
    // Gradient w.r.t. gate pre-activation z: mu = sigmoid(z) * scale
    for (int i = 0; i < dim; ++i) {
        T z = b_forget[i];
        for (int j = 0; j < feature_dim; ++j) {
            z += W_forget[i * feature_dim + j] * features[j];
        }
        
        if (force != nullptr && W_input != nullptr) {
            for (int j = 0; j < dim; ++j) {
                z += W_input[i * dim + j] * force[j];
            }
        }
        
        T s = sigmoid<T>(z);
        T dz = grad_out[i] * static_cast<T>(FRICTION_SCALE<T>) * s * (static_cast<T>(1) - s);
        
        // Accumulate gradients for weights and bias
        grad_b_forget[i] += dz;
        for (int j = 0; j < feature_dim; ++j) {
            grad_W_forget[i * feature_dim + j] += dz * features[j];
        }
        
        // Force and W_input gradients
        if (force != nullptr && W_input != nullptr) {
            for (int j = 0; j < dim; ++j) {
                grad_W_input[i * dim + j] += dz * force[j];
                if (grad_force != nullptr) {
                    grad_force[j] += dz * W_input[i * dim + j];
                }
            }
        }
        
        // Propagate to features
        if (topology == Topology::TORUS) {
            for (int j = 0; j < dim; ++j) {
                T d_sin = W_forget[i * feature_dim + j] * dz;
                T d_cos = W_forget[i * feature_dim + (dim + j)] * dz;
                grad_x[j] += d_sin * cos(x[j]) - d_cos * sin(x[j]);
            }
        } else {
            for (int j = 0; j < dim; ++j) {
                grad_x[j] += W_forget[i * feature_dim + j] * dz;
            }
        }
    }
}

} // namespace cuda
} // namespace gfn

#endif // GFN_CUDA_CHRISTOFFEL_IMPL_CUH
