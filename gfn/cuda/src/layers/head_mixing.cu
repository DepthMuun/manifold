#include <cuda.h>
#include <cuda_runtime.h>
#include "../../include/gradients.cuh"

#define BLOCK_SIZE 256

/**
 * HEAD MIXING KERNEL
 * ------------------
 * Mixes information between attention heads via linear projections.
 * Input: [H, B, D/H] per head
 * Output: [B, D] mixed state
 */

__global__ void head_mixing_kernel(
    const float* __restrict__ x_heads,  // [H, B, D/H]
    const float* __restrict__ v_heads,  // [H, B, D/H]
    const float* __restrict__ W_x,       // [D, D] or [D, 3*D]
    const float* __restrict__ W_v,       // [D, D]
    float* __restrict__ x_out,           // [B, D]
    float* __restrict__ v_out,           // [B, D]
    const int heads,
    const int batch,
    const int dim,
    const int head_dim,
    const int topology
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (b >= batch) return;
    
    extern __shared__ float s_mem[];
    float* s_concat_x = s_mem;                    // [D]
    float* s_concat_v = s_concat_x + dim;         // [D]
    float* s_out_x = s_concat_v + dim;            // [D]
    float* s_out_v = s_out_x + dim;               // [D]
    
    // Step 1: Concatenate heads [H, B, D/H] -> [B, D]
    for (int i = tid; i < dim; i += blockDim.x) {
        int h = i / head_dim;
        int d = i % head_dim;
        s_concat_x[i] = x_heads[(h * batch + b) * head_dim + d];
        s_concat_v[i] = v_heads[(h * batch + b) * head_dim + d];
    }
    __syncthreads();
    
    // Step 2: Projection
    for (int i = tid; i < dim; i += blockDim.x) {
        float sum_x = 0.0f;
        float sum_v = 0.0f;
        
        if (topology == 1) { // TORUS
            // W_x is [D, 3*D]
            // Input is [sin(x), cos(x), tanh(v/100)]
            int stride = 3 * dim;
            for (int j = 0; j < dim; j++) {
                float val_x = s_concat_x[j];
                float val_v = s_concat_v[j];
                float v_mix = tanhf(val_v / 100.0f);
                
                sum_x += sinf(val_x) * W_x[i * stride + j];
                sum_x += cosf(val_x) * W_x[i * stride + j + dim];
                sum_x += v_mix       * W_x[i * stride + j + 2 * dim];
            }
        } else { // EUCLIDEAN
            for (int j = 0; j < dim; j++) {
                sum_x += s_concat_x[j] * W_x[i * dim + j];  // W_x[i, j]
            }
        }

        // V projection is always linear [D, D]
        for (int j = 0; j < dim; j++) {
            sum_v += s_concat_v[j] * W_v[i * dim + j];  // W_v[i, j]
        }
        
        s_out_x[i] = sum_x;
        s_out_v[i] = sum_v;
    }
    __syncthreads();
    
    // Step 3: Write to global memory
    for (int i = tid; i < dim; i += blockDim.x) {
        x_out[b * dim + i] = s_out_x[i];
        v_out[b * dim + i] = s_out_v[i];
    }
}

__global__ void head_mixing_backward_kernel(
    float* __restrict__ g_x_heads,      // [H, B, D/H] (Out)
    float* __restrict__ g_v_heads,      // [H, B, D/H] (Out)
    const float* __restrict__ x_heads,  // [H, B, D/H] (In)
    const float* __restrict__ v_heads,  // [H, B, D/H] (In)
    const float* __restrict__ g_x_out,  // [B, D] (In: Gradient w.r.t Output of mixing)
    const float* __restrict__ g_v_out,  // [B, D] (In: Gradient w.r.t Output of mixing)
    const float* __restrict__ W_x,      // [D, D] or [D, 3D]
    const float* __restrict__ W_v,      // [D, D]
    float* __restrict__ g_W_x,          // [D, D] or [D, 3D] (Out: Accumulate)
    float* __restrict__ g_W_v,          // [D, D] (Out: Accumulate)
    const int heads,
    const int batch,
    const int dim,
    const int head_dim,
    const int topology
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (b >= batch) return;
    
    extern __shared__ float s_mem[];
    float* s_x = s_mem;                 // [D]
    float* s_v = s_x + dim;             // [D]
    float* s_gx = s_v + dim;            // [D]
    float* s_gv = s_gx + dim;           // [D]
    float* s_temp_x = s_gv + dim;       // [D]
    float* s_temp_v = s_temp_x + dim;   // [D]

    // 1. Load Inputs (x_heads, v_heads) -> s_x, s_v
    for (int i = tid; i < dim; i += blockDim.x) {
        int h = i / head_dim;
        int d = i % head_dim;
        s_x[i] = x_heads[(h * batch + b) * head_dim + d];
        s_v[i] = v_heads[(h * batch + b) * head_dim + d];
    }

    // 2. Load Gradients (g_x_out, g_v_out) -> s_gx, s_gv
    for (int i = tid; i < dim; i += blockDim.x) {
        s_gx[i] = g_x_out[b * dim + i];
        s_gv[i] = g_v_out[b * dim + i];
    }
    __syncthreads();

    // 3. Call Device Backward
    head_mixing_backward_device(
        s_gx, s_gv, // Gradients w.r.t output (will be modified to dL/dInput)
        s_x, s_v,   // Inputs to the layer
        W_x, W_v,
        g_W_x, g_W_v,
        s_temp_x, s_temp_v, // Scratch
        dim, tid, topology
    );

    // 4. Write Gradients (s_gx, s_gv) -> g_x_heads, g_v_heads
    // Note: head_mixing_backward_device modifies s_gx/s_gv in place to be dL/dInput
    for (int i = tid; i < dim; i += blockDim.x) {
        int h = i / head_dim;
        int d = i % head_dim;
        g_x_heads[(h * batch + b) * head_dim + d] = s_gx[i];
        g_v_heads[(h * batch + b) * head_dim + d] = s_gv[i];
    }
}

extern "C" void launch_head_mixing_fused(
    const float* x_heads, const float* v_heads,
    const float* W_x, const float* W_v,
    float* x_out, float* v_out,
    int heads, int batch, int dim,
    int topology,
    cudaStream_t stream
) {
    const int head_dim = dim / heads;
    const int shared_bytes = 4 * dim * sizeof(float);
    
    head_mixing_kernel<<<batch, BLOCK_SIZE, shared_bytes, stream>>>(
        x_heads, v_heads, W_x, W_v, x_out, v_out,
        heads, batch, dim, head_dim, topology
    );
}

extern "C" void launch_head_mixing_backward(
    float* g_x_heads, float* g_v_heads,
    const float* x_heads, const float* v_heads,
    const float* g_x_out, const float* g_v_out,
    const float* W_x, const float* W_v,
    float* g_W_x, float* g_W_v,
    int heads, int batch, int dim,
    int topology,
    cudaStream_t stream
) {
    const int head_dim = dim / heads;
    const int shared_bytes = 6 * dim * sizeof(float);
    
    head_mixing_backward_kernel<<<batch, BLOCK_SIZE, shared_bytes, stream>>>(
        g_x_heads, g_v_heads, x_heads, v_heads,
        g_x_out, g_v_out, W_x, W_v,
        g_W_x, g_W_v,
        heads, batch, dim, head_dim, topology
    );
}
