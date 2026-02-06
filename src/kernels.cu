#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

#include "../tester/utils.h"



// ---------------- warp reduce ----------------
template <typename T>
__inline__ __device__
T warp_reduce_sum(T val) {
    unsigned mask = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}


// ---------------- block reduce ----------------
template <typename T, int BLOCK_SIZE>
__inline__ __device__
T block_reduce_sum(T val) {
    static __shared__ T shared[BLOCK_SIZE / 32];
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0)
        shared[warp_id] = val;

    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / warpSize) ? shared[lane] : T(0);
    if (warp_id == 0)
        val = warp_reduce_sum(val);

    return val; // lane0 of warp0 has block sum
}

// ---------------- reduce kernel ----------------
template <typename T, int BLOCK_SIZE>
__global__
void reduce_kernel(const T* input, size_t stride_access, size_t offset, T* out) {
    T local = 0;

    size_t tid = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    size_t grid_stride = BLOCK_SIZE * gridDim.x;

    // grid-stride loop: 支持任意间距访问
    for (size_t i = tid; i < stride_access; i += grid_stride) {
        local += input[i * offset]; // offset = cols + 1 for trace
    }

    // block reduce
    local = block_reduce_sum<T, BLOCK_SIZE>(local);

    if (threadIdx.x == 0)
        atomicAdd(out, local);
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function
  size_t N = std::min(rows, cols);      // 对角线长度
    constexpr int BLOCK_SIZE = 256;

    // 申请 device memory
    T* d_input;
    T* d_out;
    cudaMalloc(&d_input, sizeof(T) * rows * cols);
    cudaMalloc(&d_out, sizeof(T));
    cudaMemcpy(d_input, h_input.data(), sizeof(T) * rows * cols, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(T));

    // grid size
    int grid = std::min((int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE), 1024); // 防止过大

    // 调用 kernel: offset = cols + 1 for diagonal
    reduce_kernel<T, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(d_input, N, cols + 1, d_out);

    // 拷贝回 host
    T h_out;
    cudaMemcpy(&h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_out);

    return h_out;
}











template <typename T>
__device__ inline float to_float(T x) { return static_cast<float>(x); }

template <>
__device__ inline float to_float<half>(half x) { return __half2float(x); }

template <typename T>
__device__ inline T from_float(float x) { return static_cast<T>(x); }

template <>
__device__ inline half from_float<half>(float x) { return __float2half(x); }

/**
 * @brief Warp-tiled FlashAttention kernel (supports GQA + causal)
 */
template <typename T, int TILE_Q = 16, int TILE_K = 16>
__global__ void flashAttentionKernel(
    const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V,
    T* __restrict__ O,
    int batch_size, int tgt_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    bool is_causal)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;

    int q_tile_start = warp_id * TILE_Q;
    if(q_tile_start >= batch_size * tgt_seq_len * query_heads) return;

    extern __shared__ float smem[];
    float* K_tile = smem;
    float* V_tile = smem + TILE_K * head_dim;

    // Compute mapping for GQA
    int head_ratio = query_heads / kv_heads;

    // Determine batch, tgt_seq, query_head for this warp tile
    int warp_idx = warp_id;
    int b   = warp_idx / (tgt_seq_len * query_heads);
    int rem = warp_idx % (tgt_seq_len * query_heads);
    int t   = rem / query_heads;
    int qh  = rem % query_heads;

    int kv_group = qh / head_ratio;

    // Loop over K tiles
    for(int k_tile_start = 0; k_tile_start < src_seq_len; k_tile_start += TILE_K) {
        int effective_tile = min(TILE_K, src_seq_len - k_tile_start);

        // Load K tile
        for(int i = lane; i < effective_tile * head_dim; i += 32) {
            int row = i / head_dim;
            int col = i % head_dim;
            int k_idx = b * src_seq_len * kv_heads * head_dim
                        + (k_tile_start + row) * kv_heads * head_dim
                        + kv_group * head_dim + col;
            K_tile[row * head_dim + col] = to_float(K[k_idx]);
        }

        // Load V tile
        for(int i = lane; i < effective_tile * head_dim; i += 32) {
            int row = i / head_dim;
            int col = i % head_dim;
            int v_idx = b * src_seq_len * kv_heads * head_dim
                        + (k_tile_start + row) * kv_heads * head_dim
                        + kv_group * head_dim + col;
            V_tile[row * head_dim + col] = to_float(V[v_idx]);
        }

        __syncwarp();

        // Loop over Q rows in tile
        for(int q_row = 0; q_row < TILE_Q; q_row++) {
            int q_idx_global = q_tile_start + q_row;
            if(q_idx_global >= tgt_seq_len) continue;

            // Compute dot products with K_tile
            float acc_out[32] = {0.0f}; // assume head_dim <= 32
            float max_score = -1e20f;
            float scores[TILE_K] = {0.0f};

            for(int k_inner = 0; k_inner < effective_tile; k_inner++) {
                int k_idx_global = k_tile_start + k_inner;
                if(is_causal && k_idx_global > q_idx_global) { scores[k_inner] = -1e20f; continue; }

                float dot = 0.0f;
                for(int d = lane; d < head_dim; d += 32) {
                    int q_idx = b * tgt_seq_len * query_heads * head_dim
                                + q_idx_global * query_heads * head_dim
                                + qh * head_dim + d;
                    dot += to_float(Q[q_idx]) * K_tile[k_inner * head_dim + d];
                }
                // warp reduction
                for(int offset = 16; offset > 0; offset /= 2)
                    dot += __shfl_down_sync(0xffffffff, dot, offset);

                if(lane % 32 == 0) {
                    scores[k_inner] = dot;
                    if(dot > max_score) max_score = dot;
                }
            }

            __syncwarp();

            // Compute softmax sum
            float sum_exp = 0.0f;
            for(int k_inner = 0; k_inner < effective_tile; k_inner++) {
                if(lane % 32 == 0)
                    sum_exp += expf(scores[k_inner] - max_score);
            }

            // Compute weighted sum over V
            for(int k_inner = 0; k_inner < effective_tile; k_inner++) {
                float weight = expf(scores[k_inner] - max_score) / sum_exp;
                for(int d = lane; d < head_dim; d += 32) {
                    acc_out[d] += weight * V_tile[k_inner * head_dim + d];
                }
            }

            // Write output
            for(int d = lane; d < head_dim; d += 32) {
                int o_idx = b * tgt_seq_len * query_heads * head_dim
                            + q_idx_global * query_heads * head_dim
                            + qh * head_dim + d;
                O[o_idx] = from_float<T>(acc_out[d]);
            }
        }

        __syncwarp();
    } // end K tile
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {    
  // TODO: Implement the flash attention function
// Allocate device memory
    T *d_q, *d_k, *d_v, *d_o;
    size_t q_size = h_q.size() * sizeof(T);
    size_t k_size = h_k.size() * sizeof(T);
    size_t v_size = h_v.size() * sizeof(T);
    size_t o_size = batch_size * target_seq_len * query_heads * head_dim * sizeof(T);

    cudaMalloc(&d_q, q_size);
    cudaMalloc(&d_k, k_size);
    cudaMalloc(&d_v, v_size);
    cudaMalloc(&d_o, o_size);

    cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice);

    // Launch kernel
    int total_q_rows = batch_size * target_seq_len * query_heads;
    int num_warps = (total_q_rows + 15) / 16; // TILE_Q = 16
    dim3 block(32 * 4); // 4 warps per block
    dim3 grid((num_warps + 3) / 4);

    size_t shared_bytes = 2 * 16 * head_dim * sizeof(float); // TILE_K * head_dim for K + V

    flashAttentionKernel<T,16,16><<<grid, block, shared_bytes>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal);

    // Copy back
    h_o.resize(batch_size * target_seq_len * query_heads * head_dim);
    cudaMemcpy(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);


}












// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
