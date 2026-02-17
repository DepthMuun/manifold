#ifndef GFN_CUDA_INTEGRATOR_UTILS_CUH
#define GFN_CUDA_INTEGRATOR_UTILS_CUH

#include "../geometry/christoffel_impl.cuh"

namespace gfn {
namespace cuda {

// ============================================================================
// Block Reduction Helpers
// ============================================================================

template <typename scalar_t>
__device__ inline scalar_t warp_reduce_sum_shared(scalar_t val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename scalar_t>
__device__ inline scalar_t block_reduce_sum_shared(scalar_t val) {
    static __shared__ scalar_t shared_red[32]; // Renamed for clarity
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum_shared(val);

    if (lane == 0) shared_red[wid] = val;
    __syncthreads();

    // Sum warp-level results
    val = (threadIdx.x < blockDim.x / warpSize) ? shared_red[lane] : 0;
    if (wid == 0) val = warp_reduce_sum_shared(val);
    
    // Broadcast result from thread 0 to all threads
    if (threadIdx.x == 0) shared_red[0] = val;
    __syncthreads();
    scalar_t res = shared_red[0];
    __syncthreads(); // Ensure shared_red is safe for next call

    return res;
}


// ============================================================================
// Distributed Christoffel (block-parallel)
// ============================================================================

template <typename scalar_t>
__device__ void christoffel_distributed_shared(
    scalar_t v_val,
    const scalar_t* U,
    const scalar_t* W,
    const scalar_t* x_shared,
    scalar_t* v_shared,
    int dim,
    int rank,
    scalar_t plasticity,
    scalar_t sing_thresh,
    scalar_t sing_strength,
    Topology topology,
    scalar_t R,
    scalar_t r,
    scalar_t thermo_alpha,
    scalar_t thermo_temp,
    scalar_t holo_z,
    const scalar_t* holo_grad_z,
    scalar_t f_val, // Local force for energy modulation
    scalar_t* gamma_val,
    scalar_t* h_shared
) {
    int tid = threadIdx.x;
    if (tid >= dim) return;

    // --- 1. Toroidal Contribution (Special Case) ---
    if (topology == Topology::TORUS && x_shared != nullptr) {
        *gamma_val = 0.0f;
        v_shared[tid] = v_val;
        __syncthreads();
        
        if (tid % 2 == 0 && tid < dim - 1) {
            scalar_t th = x_shared[tid];
            scalar_t v_ph = v_shared[tid + 1];
            scalar_t s, c;
            sincos_scalar(th, &s, &c);
            scalar_t denom = R + r * c;
            denom = (denom < static_cast<scalar_t>(CLAMP_MIN_STRONG<scalar_t>)) ? static_cast<scalar_t>(CLAMP_MIN_STRONG<scalar_t>) : denom;
            scalar_t term_th = denom * s / (r + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
            *gamma_val = term_th * (v_ph * v_ph) * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>);
        } else if (tid % 2 != 0) {
            scalar_t th = x_shared[tid - 1];
            scalar_t v_ph = v_val;
            scalar_t v_th = v_shared[tid - 1];
            scalar_t s, c;
            sincos_scalar(th, &s, &c);
            scalar_t denom = R + r * c;
            denom = (denom < static_cast<scalar_t>(CLAMP_MIN_STRONG<scalar_t>)) ? static_cast<scalar_t>(CLAMP_MIN_STRONG<scalar_t>) : denom;
            scalar_t term_ph = -(r * s) / (denom + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
            *gamma_val = 2.0f * term_ph * v_ph * v_th * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>);
        }
        // Fall through to modulation
    } else {
        // --- 2. Low-Rank Euclidean Contribution ---
        for (int k = 0; k < rank; ++k) {
            scalar_t prod = U[tid * rank + k] * v_val;
            scalar_t sum = block_reduce_sum_shared(prod);
            if (tid == 0) h_shared[k] = sum;
            __syncthreads(); // Ensure all threads wait for h_shared[k] to be written
        }
        __syncthreads(); // Ensure all h_shared values are written before proceeding
        
        __shared__ scalar_t S_shared;
        __shared__ scalar_t M_shared;
        if (tid == 0) {
            scalar_t energy = 0.0f;
            for (int k = 0; k < rank; ++k) energy += h_shared[k] * h_shared[k];
            if (rank > 0) energy /= static_cast<scalar_t>(rank);
            scalar_t norm = sqrt(energy);
            S_shared = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
            
            // PARITY FIX: Plasticity modulation (was hardcoded M=1.0)
            M_shared = static_cast<scalar_t>(1);
            if (plasticity > static_cast<scalar_t>(1e-6) || plasticity < static_cast<scalar_t>(-1e-6)) {
                scalar_t v_energy = energy; // Reuse normalized energy
                M_shared += plasticity * static_cast<scalar_t>(0.1) * tanh(v_energy);
            }
        }
        __syncthreads();
        
        scalar_t sum_gamma = 0.0f;
        for (int k = 0; k < rank; ++k) {
            sum_gamma += W[tid * rank + k] * h_shared[k] * h_shared[k] * S_shared * M_shared;
        }
        *gamma_val = sum_gamma;
    }

    // --- 3. AdS/CFT Holographic Term (Extension) ---
    // Gamma^k_ads = -(1/z) * (2 * (grad_z . v) * v^k - (v . v) * grad_z_k)
    // Add this to the base geometry before scaling
    if (holo_z > 0.0f && holo_grad_z != nullptr) {
        scalar_t v_dot_gz = v_val * holo_grad_z[tid];
        scalar_t v_dot_gz_sum = block_reduce_sum_shared(v_dot_gz);
        
        scalar_t v_sq = v_val * v_val;
        scalar_t v_sq_sum = block_reduce_sum_shared(v_sq);
        
        __shared__ scalar_t common_v_dot_gz;
        __shared__ scalar_t common_v_sq;
        if (tid == 0) {
            common_v_dot_gz = v_dot_gz_sum;
            common_v_sq = v_sq_sum;
        }
        __syncthreads();
        
        scalar_t term_ads = -(1.0f / holo_z) * (2.0f * common_v_dot_gz * v_val - common_v_sq * holo_grad_z[tid]);
        *gamma_val += term_ads; // Combine curvatures
    }

    // --- 4. Thermodynamic Modulation (Scaling) ---
    // Scale the entire connection
    if (thermo_alpha > 0.0f) {
        // energy_f = (force^2).mean()
        scalar_t f_sq = f_val * f_val;
        scalar_t head_energy = block_reduce_sum_shared(f_sq) / static_cast<scalar_t>(dim);
        __syncthreads(); // Ensure energy is calculated
        
        // Modulation: exp(-alpha * E / T)
        scalar_t T = (thermo_temp < static_cast<scalar_t>(CLAMP_MIN_STRONG<scalar_t>)) ? static_cast<scalar_t>(CLAMP_MIN_STRONG<scalar_t>) : thermo_temp;
        scalar_t modulator = expf(-thermo_alpha * head_energy / T);
        *gamma_val *= modulator;
    }

    // Final Clamping for stability (Paper 27)
    *gamma_val = soft_clamp<scalar_t>(*gamma_val, static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>));
}



// ============================================================================
// Distributed Friction (block-parallel)
// ============================================================================

template <typename scalar_t>
__device__ void friction_distributed_shared(
    scalar_t x_val,
    scalar_t f_val,
    const scalar_t* W_forget,
    const scalar_t* b_forget,
    const scalar_t* W_input,
    scalar_t* friction_val,
    int dim,
    Topology topology,
    scalar_t velocity_friction_scale,
    scalar_t v_norm,
    scalar_t* features_shared
) {
    int tid = threadIdx.x;
    if (tid >= dim) return;

    if (topology == Topology::TORUS) {
        scalar_t s, c;
        sincos_scalar(x_val, &s, &c);
        features_shared[tid] = s;
        features_shared[dim + tid] = c;
    } else {
        features_shared[tid] = x_val;
    }
    __syncthreads();
    
    int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
    scalar_t gate_sum = b_forget[tid];
    for (int j = 0; j < feat_dim; ++j) {
        gate_sum += W_forget[tid * feat_dim + j] * features_shared[j];
    }
    
    if (W_input != nullptr) {
        __syncthreads();
        features_shared[tid] = f_val;
        __syncthreads();
        for (int j = 0; j < dim; ++j) {
            gate_sum += W_input[tid * dim + j] * features_shared[j];
        }
    }
    
    scalar_t base_friction = sigmoid(gate_sum) * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);
    if (velocity_friction_scale > 0.0f) {
        scalar_t v_scale = v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
        *friction_val = base_friction * (1.0f + velocity_friction_scale * v_scale);
    } else {
        *friction_val = base_friction;
    }
}

} // namespace cuda
} // namespace gfn

#endif
