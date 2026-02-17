#ifndef GFN_CUDA_PARAMS_CUH
#define GFN_CUDA_PARAMS_CUH

#include "types.cuh"

namespace gfn {
namespace cuda {

/**
 * @brief Parameters defining the Manifold Geometry (Christoffel symbols context)
 */
template <typename scalar_t>
struct GeometryParams {
    const scalar_t* U;         // Low-rank projection matrix [dim, rank]
    const scalar_t* W;         // Low-rank reconstruction matrix [dim, rank]
    int rank;                 // Decomposition rank
    
    // Topology settings
    Topology topology;         // EUCLIDEAN or TORUS
    scalar_t torus_R;          // Major radius
    scalar_t torus_r;          // Minor radius
    
    // Stability & Regularization
    scalar_t plasticity;       // Active inference coefficient
    scalar_t sing_thresh;      // Singularity avoidance threshold
    scalar_t sing_strength;    // Singularity repulsion strength
    
    // Geometry Fusion (Phase 2 additions)
    scalar_t thermo_alpha;     // Thermodynamic scaling weight
    scalar_t thermo_temp;      // Learned temperature
    const scalar_t* holo_z_ptr;   // Holographic radial coordinate pointer [batch, num_heads]
    const scalar_t* holo_grad_z; // Holographic gradient (per head/batch)
    
    // Singularity Vector (Missing in Phase 1)
    const scalar_t* V_w;       // Singularity projection vector [dim] or [num_heads, head_dim]
};

/**
 * @brief Parameters defining the MLayer Physics (Friction, Hysteresis, Time)
 */
template <typename scalar_t>
struct PhysicsParams {
    scalar_t dt;               // Base time step
    const scalar_t* dt_scales; // Head-specific time scales pointer [num_heads]
    
    // Friction Gates
    const scalar_t* W_forget;  // Learned friction gate weights
    const scalar_t* b_forget;  // Learned friction gate biases
    const scalar_t* W_input;   // Force-dependent friction weights
    scalar_t v_fric_scale;     // Velocity-dependent friction scaling
    
    // Hysteresis
    bool hyst_enabled;         // Flag to enable hysteresis/ghost forces
    scalar_t hyst_decay;       // Hysteresis memory decay rate
    scalar_t* hysteresis_settings; // Hysteresis state pointer (mutable) [batch, total_dim]
    const scalar_t* hyst_up_w; // Hysteresis update weights
    const scalar_t* hyst_up_b; // Hysteresis update biases
    const scalar_t* hyst_rd_w; // Hysteresis readout weights
    const scalar_t* hyst_rd_b; // Hysteresis readout biases
};

/**
 * @brief Dynamic state for a single integration step
 */
template <typename scalar_t>
struct MLayerState {
    scalar_t* x;               // Shared pointer to position head
    scalar_t* v;               // Shared pointer to velocity head
    scalar_t* h;               // Shared pointer to hysteresis state
    scalar_t f_ext;            // Local external force value
};

} // namespace cuda
} // namespace gfn

#endif // GFN_CUDA_PARAMS_CUH
