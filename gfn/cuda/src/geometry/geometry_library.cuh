#ifndef GFN_CUDA_GEOMETRY_LIBRARY_CUH
#define GFN_CUDA_GEOMETRY_LIBRARY_CUH

#include "../common/params.cuh"
#include "../common/integrator_utils.cuh"

namespace gfn {
namespace cuda {

/**
 * @brief Universal dispatcher for block-parallel Christoffel symbol computation.
 * 
 * This module handles:
 * 1. Base Geometry (Euclidean Low-Rank or Toroidal)
 * 2. Holographic Extension (AdS/CFT)
 * 3. Thermodynamic Modulation (Energy-dependent scaling)
 */
template <typename scalar_t>
GFN_DEVICE void compute_christoffel_distributed(
    const GeometryParams<scalar_t>& p,
    const MLayerState<scalar_t>& s,
    int dim,
    scalar_t local_holo_z, // Passed as pre-loaded scalar
    scalar_t* gamma_val, 
    scalar_t* h_shared,  
    scalar_t* v_shared   
) {
    int tid = threadIdx.x;
    if (tid >= dim) return;

    // --- Phase 1: Base Geometry (Curvature of the background) ---
    if (p.topology == Topology::TORUS && s.x != nullptr) {
        // Toroidal Geometry: Metric derived from donut-like fold
        *gamma_val = static_cast<scalar_t>(0);
        v_shared[tid] = *s.v;
        __syncthreads();
        
        if (tid % 2 == 0 && tid < dim - 1) {
            scalar_t th = s.x[tid];
            scalar_t v_ph = v_shared[tid + 1];
            scalar_t sin_th, cos_th;
            sincos_scalar(th, &sin_th, &cos_th);
            
            scalar_t denom = p.torus_R + p.torus_r * cos_th;
            denom = (denom < static_cast<scalar_t>(CLAMP_MIN_STRONG<scalar_t>)) ? static_cast<scalar_t>(CLAMP_MIN_STRONG<scalar_t>) : denom;
            
            scalar_t term_th = denom * sin_th / (p.torus_r + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
            *gamma_val = term_th * (v_ph * v_ph) * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>);
        } else if (tid % 2 != 0) {
            scalar_t th = s.x[tid - 1];
            scalar_t v_ph = *s.v;
            scalar_t v_th = v_shared[tid - 1];
            scalar_t sin_th, cos_th;
            sincos_scalar(th, &sin_th, &cos_th);
            
            scalar_t denom = p.torus_R + p.torus_r * cos_th;
            denom = (denom < static_cast<scalar_t>(CLAMP_MIN_STRONG<scalar_t>)) ? static_cast<scalar_t>(CLAMP_MIN_STRONG<scalar_t>) : denom;
            
            scalar_t term_ph = -(p.torus_r * sin_th) / (denom + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
            *gamma_val = static_cast<scalar_t>(2) * term_ph * v_ph * v_th * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>);
        }
    } else {
        // Euclidean Low-Rank: Learned manifold approximation
        for (int k = 0; k < p.rank; ++k) {
            scalar_t prod = p.U[tid * p.rank + k] * (*s.v);
            scalar_t sum = block_reduce_sum_shared(prod);
            if (tid == 0) h_shared[k] = sum;
            __syncthreads(); 
        }
        
        __shared__ scalar_t S_shared;
        if (tid == 0) {
            scalar_t h_energy = static_cast<scalar_t>(0);
            for (int k = 0; k < p.rank; ++k) h_energy += h_shared[k] * h_shared[k];
            if (p.rank > 0) h_energy /= static_cast<scalar_t>(p.rank);
            S_shared = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + sqrt(h_energy) + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
        }
        __syncthreads();
        
        scalar_t sum_gamma = static_cast<scalar_t>(0);
        for (int k = 0; k < p.rank; ++k) {
            sum_gamma += p.W[tid * p.rank + k] * h_shared[k] * h_shared[k] * S_shared;
        }
        *gamma_val = sum_gamma;
    }

    // --- Phase 2: Active Inference (Plasticity & Singularities) ---
    // (To be modularized into active_inference.cuh)
    scalar_t M = static_cast<scalar_t>(1.0);

    // Plasticity
    if (abs(p.plasticity) > static_cast<scalar_t>(1e-6)) {
        scalar_t v_sq = (*s.v) * (*s.v);
        scalar_t total_energy = block_reduce_sum_shared(v_sq) / static_cast<scalar_t>(dim);
        M += p.plasticity * static_cast<scalar_t>(0.1) * tanh(total_energy);
    }
    
    // Singularities (Curvature Amplification)
    if (p.V_w != nullptr) {
        scalar_t pot_term;
        if (p.topology == Topology::TORUS) {
             scalar_t sin_th, cos_th;
             sincos_scalar(s.x[tid], &sin_th, &cos_th);
             // V_w layout: [sin_weights... cos_weights...]
             scalar_t w_sin = p.V_w[tid];
             scalar_t w_cos = p.V_w[dim + tid];
             pot_term = sin_th * w_sin + cos_th * w_cos;
        } else {
             pot_term = s.x[tid] * p.V_w[tid];
        }
        
        scalar_t pot_sum = block_reduce_sum_shared(pot_term);
        
        // Broadcast potential
        __shared__ scalar_t shared_pot;
        if (tid == 0) shared_pot = pot_sum;
        __syncthreads();
        
        scalar_t gate = sigmoid(shared_pot);
        scalar_t slope = static_cast<scalar_t>(SINGULARITY_GATE_SLOPE<scalar_t>);
        scalar_t soft_m = sigmoid(slope * (gate - p.sing_thresh));
        
        M *= (static_cast<scalar_t>(1.0) + (p.sing_strength - static_cast<scalar_t>(1.0)) * soft_m);
    }
    
    *gamma_val *= M;

    // --- Phase 3: Holographic Extension (AdS/CFT) ---
    if (local_holo_z > static_cast<scalar_t>(0) && p.holo_grad_z != nullptr) {
        scalar_t v_dot_gz = (*s.v) * p.holo_grad_z[tid];
        scalar_t v_dot_gz_sum = block_reduce_sum_shared(v_dot_gz);
        scalar_t v_sq_sum = block_reduce_sum_shared((*s.v) * (*s.v));
        
        // Use shared memory for broadcasting reduction results
        __shared__ scalar_t common_v_dot_gz, common_v_sq;
        if (tid == 0) {
            common_v_dot_gz = v_dot_gz_sum;
            common_v_sq = v_sq_sum;
        }
        __syncthreads();
        
        scalar_t ads = -(static_cast<scalar_t>(1) / local_holo_z) * (static_cast<scalar_t>(2) * common_v_dot_gz * (*s.v) - common_v_sq * p.holo_grad_z[tid]);
        *gamma_val += ads;
    }

    // --- Phase 4: Thermodynamic Modulation ---
    if (p.thermo_alpha > static_cast<scalar_t>(0)) {
        scalar_t head_energy = block_reduce_sum_shared(s.f_ext * s.f_ext) / static_cast<scalar_t>(dim);
        scalar_t T = (p.thermo_temp < static_cast<scalar_t>(CLAMP_MIN_STRONG<scalar_t>)) ? static_cast<scalar_t>(CLAMP_MIN_STRONG<scalar_t>) : p.thermo_temp;
        scalar_t modulator = exp(-p.thermo_alpha * head_energy / T);
        *gamma_val *= modulator;
    }

    // --- Phase 5: Final Stability (Hard Clamping) ---
    *gamma_val = soft_clamp<scalar_t>(*gamma_val, static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>));
}

} // namespace cuda
} // namespace gfn

#endif // GFN_CUDA_GEOMETRY_LIBRARY_CUH
