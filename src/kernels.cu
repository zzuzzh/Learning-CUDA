#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cfloat>

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





























#define WARP_SIZE 32
#define BLOCK_SIZE 128 // 4 Warps per block

// -----------------------------------------------------------------------------
// Warp-level Reduction 辅助函数
// -----------------------------------------------------------------------------
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// -----------------------------------------------------------------------------
// CUDA Kernel
// -----------------------------------------------------------------------------
template <typename T>
__global__ void flash_attn_warp_tiled_kernel(
    const T* Q, const T* K, const T* V, T* O,
    int B, int N_target, int N_src, int H_q, int H_kv, int D,
    float scale, bool is_causal
) {
    // 每个 Warp 负责一行 (Query Row)
    int warp_id = (blockIdx.y * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    int batch_idx = blockIdx.x / H_q;
    int head_q_idx = blockIdx.x % H_q;
    int head_kv_idx = head_q_idx / (H_q / H_kv); 
    
    // 如果 warp_id 超出了目标序列长度，直接返回
    if (warp_id >= N_target) return;

    // 共享内存：用于存放当前 KV 块
    // 实际优化中 Q 也可放入 Shared，此处重点展示 KV Tiling
    extern __shared__ float s_mem[]; 
    float* s_K = s_mem;                    // Size: Bc * D
    float* s_V = s_mem + (WARP_SIZE * D);   // Size: Bc * D

    float row_m = -FLT_MAX;
    float row_l = 0.0f;
    
    // 每个线程负责输出 D 维向量中的一部分 (Warp-tiled GEMM 基础)
    float acc_o[128]; // 假设 D <= 128
    for (int d = 0; d < D; ++d) acc_o[d] = 0.0f;

    // 循环遍历 KV 的 Sequence dimension (Bc 为分块大小)
    int Bc = WARP_SIZE; 
    for (int j_start = 0; j_start < N_src; j_start += Bc) {
        
        // 1. 加载 K, V 到 Shared Memory (协作加载)
        if (j_start + lane_id < N_src) {
            for (int d = 0; d < D; ++d) {
                s_K[lane_id * D + d] = (float)K[((batch_idx * N_src + j_start + lane_id) * H_kv + head_kv_idx) * D + d];
                s_V[lane_id * D + d] = (float)V[((batch_idx * N_src + j_start + lane_id) * H_kv + head_kv_idx) * D + d];
            }
        }
        __syncthreads();

        // 2. 计算当前 Warp 负责的行与 Shared Memory 中 KV 块的点积
        // 这里每个线程计算该行与 Bc 个 Key 中的一个的点积
        float score = -FLT_MAX;
        int current_kv_idx = j_start + lane_id;
        
        if (current_kv_idx < N_src && (!is_causal || current_kv_idx <= warp_id)) {
            score = 0.0f;
            for (int d = 0; d < D; ++d) {
                float q_val = (float)Q[((batch_idx * N_target + warp_id) * H_q + head_q_idx) * D + d];
                score += q_val * s_K[lane_id * D + d];
            }
            score *= scale;
        }

        // 3. Online Softmax (Warp 内部归约)
        float max_in_block = warpReduceMax(score);
        float new_m = fmaxf(row_m, max_in_block);
        
        float exp_score = (score == -FLT_MAX) ? 0.0f : expf(score - new_m);
        float row_l_updated = row_l * expf(row_m - new_m) + warpReduceSum(exp_score);

        // 4. 更新输出累加器 (V 也是按 lane 加载的)
        float p_scale = expf(row_m - new_m);
        for (int d = 0; d < D; ++d) {
            // 先调整旧的累加值
            acc_o[d] *= p_scale;
            // 加上新的贡献：exp_score 对应的是当前 lane_id (即当前块内的某个 j)
            // 需要通过 Shuffle 让所有线程都能获取每个 exp_score 并累加对应的 V
            for (int k = 0; k < WARP_SIZE; ++k) {
                float e = __shfl_sync(0xffffffff, exp_score, k);
                float v_val = s_V[k * D + d];
                acc_o[d] += e * v_val;
            }
        }

        row_m = new_m;
        row_l = row_l_updated;
        __syncthreads();
    }

    // 5. 最终归一化并写回全局内存
    // 只有 lane_0 包含完整的 row_l，但此处所有线程都需要 row_l 归一化自己负责的 d
    float final_l = __shfl_sync(0xffffffff, row_l, 0); 
    for (int d = 0; d < D; ++d) {
        O[((batch_idx * N_target + warp_id) * H_q + head_q_idx) * D + d] = (T)(acc_o[d] / final_l);
    }
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
   T *d_q, *d_k, *d_v, *d_o;
    size_t q_size = h_q.size() * sizeof(T);
    size_t kv_size = h_k.size() * sizeof(T);
    
    cudaMalloc(&d_q, q_size);
    cudaMalloc(&d_k, kv_size);
    cudaMalloc(&d_v, kv_size);
    cudaMalloc(&d_o, q_size);

    cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), kv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), kv_size, cudaMemcpyHostToDevice);

    float scale = 1.0f / sqrtf((float)head_dim);

    // 计算共享内存大小: Bc * D * 2 (K and V) * 4 bytes
    int Bc = WARP_SIZE;
    size_t smem_size = Bc * head_dim * 2 * sizeof(float);

    // 一个 Block 4 个 Warps，Grid 根据目标序列长度分配
    dim3 block(BLOCK_SIZE); 
    dim3 grid(batch_size * query_heads, (target_seq_len + (BLOCK_SIZE/WARP_SIZE) - 1) / (BLOCK_SIZE/WARP_SIZE));

    flash_attn_warp_tiled_kernel<T><<<grid, block, smem_size>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim,
        scale, is_causal
    );

    cudaMemcpy(h_o.data(), d_o, q_size, cudaMemcpyDeviceToHost);

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
