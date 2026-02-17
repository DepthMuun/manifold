#ifndef GFN_CUDA_PHYSICS_LIBRARY_CUH
#define GFN_CUDA_PHYSICS_LIBRARY_CUH

#include "../common/params.cuh"
#include "../common/integrator_utils.cuh"

namespace gfn {
namespace cuda {

/**
 * @brief Modular block-parallel friction computation.
 * 
 * μ = sigmoid(W_f * features + b_f + W_i * force) * FRICTION_SCALE
 */
template <typename scalar_t>
GFN_DEVICE void compute_friction_distributed(
    const PhysicsParams<scalar_t>& p,
    const MLayerState<scalar_t>& s,
    int dim,
    Topology topology,
    scalar_t v_norm,
    scalar_t* friction_val,
    scalar_t* features_shared // Buffer for [2*dim]
) {
    int tid = threadIdx.x;
    if (tid >= dim) return;

    // Feature Extract
    if (topology == Topology::TORUS) {
        scalar_t sin_th, cos_th;
        sincos_scalar(s.x[tid], &sin_th, &cos_th);
        features_shared[tid] = sin_th;
        features_shared[dim + tid] = cos_th;
    } else {
        features_shared[tid] = s.x[tid];
    }
    __syncthreads();
    
    int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
    scalar_t gate_sum = p.b_forget[tid];
    
    // Position-dependent forget gate
    for (int j = 0; j < feat_dim; ++j) {
        gate_sum += p.W_forget[tid * feat_dim + j] * features_shared[j];
    }
    
    // Force-dependent input gate
    if (p.W_input != nullptr) {
        __syncthreads();
        features_shared[tid] = s.f_ext;
        __syncthreads();
        for (int j = 0; j < dim; ++j) {
            gate_sum += p.W_input[tid * dim + j] * features_shared[j];
        }
    }
    
    scalar_t base_mu = sigmoid<scalar_t>(gate_sum) * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);
    
    // Velocity dependence
    if (p.v_fric_scale > static_cast<scalar_t>(0)) {
        scalar_t v_scale = v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
        *friction_val = base_mu * (static_cast<scalar_t>(1) + p.v_fric_scale * v_scale);
    } else {
        *friction_val = base_mu;
    }
}

/**
 * @brief Modular Hysteresis state update and Ghost Force readout.
 */
template <typename scalar_t>
GFN_DEVICE void apply_hysteresis_distributed(
    const PhysicsParams<scalar_t>& p,
    MLayerState<scalar_t>& s,
    int dim,
    Topology topology,
    scalar_t* ghost_force_val,
    scalar_t* features_shared // Buffer for [2*dim + head_dim]
) {
    if (!p.hyst_enabled) {
        *ghost_force_val = static_cast<scalar_t>(0);
        return;
    }

    int tid = threadIdx.x;
    
    // 1. Prepare Features: [x_features, v]
    if (topology == Topology::TORUS) {
        scalar_t sin_th, cos_th;
        sincos_scalar(s.x[tid], &sin_th, &cos_th);
        features_shared[tid] = sin_th;
        features_shared[dim + tid] = cos_th;
    } else {
        features_shared[tid] = s.x[tid];
    }
    __syncthreads();
    
    int x_feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
    features_shared[x_feat_dim + tid] = s.v[tid];
    __syncthreads();
    
    // 2. Update Hysteresis State (forget-input gate logic)
    scalar_t up_sum = (p.hyst_up_b) ? p.hyst_up_b[tid] : static_cast<scalar_t>(0);
    int total_in = x_feat_dim + dim;
    for (int j = 0; j < total_in; ++j) {
        up_sum += p.hyst_up_w[tid * total_in + j] * features_shared[j];
    }
    
    // Symplectic state update (Additive Decay parity with manifold.py)
    *s.h = (*s.h) * p.hyst_decay + tanh(up_sum);
    __syncthreads();
    
    // 3. Readout Ghost Force
    features_shared[tid] = *s.h;
    __syncthreads();
    
    scalar_t rd_sum = (p.hyst_rd_b) ? p.hyst_rd_b[tid] : static_cast<scalar_t>(0);
    for (int j = 0; j < dim; ++j) {
        rd_sum += p.hyst_rd_w[tid * dim + j] * features_shared[j];
    }
    *ghost_force_val = rd_sum;
}

} // namespace cuda
} // namespace gfn

#endif // GFN_CUDA_PHYSICS_LIBRARY_CUH
