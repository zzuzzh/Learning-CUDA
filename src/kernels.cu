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



// --- Warp 级归约辅助函数 ---
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
// Flash Attention 核心 Kernel
// -----------------------------------------------------------------------------
template <typename T>
__global__ void flash_attn_optimized_kernel(
    const T* Q, const T* K, const T* V, T* O,
    int B, int N_target, int N_src, int H_q, int H_kv, int D,
    float scale, bool is_causal
) {
    // 基础索引计算
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int num_warps_per_block = blockDim.x / WARP_SIZE;
    int warp_global_id = blockIdx.y * num_warps_per_block + warp_id_in_block;
    int lane_id = threadIdx.x % WARP_SIZE;

    int batch_idx = blockIdx.x / H_q;
    int head_q_idx = blockIdx.x % H_q;
    int head_kv_idx = head_q_idx / (H_q / H_kv); // GQA 支持

    if (warp_global_id >= N_target) return;

    // 共享内存布局: s_Q (当前 Block 的所有 Warp 分占) | s_K (一个 Warp 大小) | s_V (一个 Warp 大小)
    extern __shared__ float s_mem[];
    float* s_Q_base = s_mem; 
    float* s_K = s_mem + num_warps_per_block * D;
    float* s_V = s_K + WARP_SIZE * D;

    // 1. 加载当前 Warp 负责的一行 Q 到 Shared Memory
    // 每个线程协作加载 D 维中的部分
    for (int d = lane_id; d < D; d += WARP_SIZE) {
        s_Q_base[warp_id_in_block * D + d] = (float)Q[((batch_idx * N_target + warp_global_id) * H_q + head_q_idx) * D + d];
    }
    __syncthreads();

    // 寄存器统计量与累加器
    float row_m = -FLT_MAX;
    float row_l = 0.0f;
    
    // 通用索引：每个线程分担 D 维中的 MAX_D_PER_THREAD 个元素
    const int MAX_D_PER_THREAD = 8; 
    float acc_o[MAX_D_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < MAX_D_PER_THREAD; ++i) acc_o[i] = 0.0f;

    // 外层分块循环 (Tiling KV rows)
    int Bc = WARP_SIZE; 
    for (int j_start = 0; j_start < N_src; j_start += Bc) {
        
        // 2. 协作加载 K, V 块到 Shared Memory
        int j_src = j_start + lane_id;
        if (j_src < N_src) {
            for (int d = 0; d < D; ++d) {
                s_K[lane_id * D + d] = (float)K[((batch_idx * N_src + j_src) * H_kv + head_kv_idx) * D + d];
                s_V[lane_id * D + d] = (float)V[((batch_idx * N_src + j_src) * H_kv + head_kv_idx) * D + d];
            }
        }
        __syncthreads();

        // 3. 计算 S = QK^T 
        // 这里的 Q 已经在 Shared Memory 中
        float local_s = -FLT_MAX;
        if (j_src < N_src && (!is_causal || j_src <= warp_global_id)) {
            local_s = 0.0f;
            for (int d = 0; d < D; ++d) {
                local_s += s_Q_base[warp_id_in_block * D + d] * s_K[lane_id * D + d];
            }
            local_s *= scale;
        }

        // 4. Online Softmax 归约 (Warp 级)
        float max_in_block = warpReduceMax(local_s);
        float new_m = fmaxf(row_m, max_in_block);
        
        float p_scale = expf(row_m - new_m);
        float exp_score = (local_s == -FLT_MAX) ? 0.0f : expf(local_s - new_m);
        float sum_exp = warpReduceSum(exp_score);
        float new_l = row_l * p_scale + sum_exp;

        // 5. 更新输出累加器 O = O * p_scale + P * V
        // 每个线程更新自己负责的 d 维度
        #pragma unroll
        for (int i = 0; i < MAX_D_PER_THREAD; ++i) {
            int d_idx = lane_id + i * WARP_SIZE;
            if (d_idx < D) {
                acc_o[i] *= p_scale;
                // 利用 Shuffle 获取当前块内每个线程对应的 exp_score 并累加对应的 V
                for (int k = 0; k < WARP_SIZE; ++k) {
                    float e = __shfl_sync(0xffffffff, exp_score, k);
                    acc_o[i] += e * s_V[k * D + d_idx];
                }
            }
        }

        row_m = new_m;
        row_l = new_l;
        __syncthreads();
    }

    // 6. 最终归一化并写回
    float final_l = __shfl_sync(0xffffffff, row_l, 0);
    #pragma unroll
    for (int i = 0; i < MAX_D_PER_THREAD; ++i) {
        int d_idx = lane_id + i * WARP_SIZE;
        if (d_idx < D) {
            O[((batch_idx * N_target + warp_global_id) * H_q + head_q_idx) * D + d_idx] = (T)(acc_o[i] / final_l);
        }
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
    size_t q_sz = h_q.size() * sizeof(T);
    size_t kv_sz = h_k.size() * sizeof(T);

    cudaMalloc(&d_q, q_sz);
    cudaMalloc(&d_k, kv_sz);
    cudaMalloc(&d_v, kv_sz);
    cudaMalloc(&d_o, q_sz);

    cudaMemcpy(d_q, h_q.data(), q_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), kv_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), kv_sz, cudaMemcpyHostToDevice);

    float scale = 1.0f / sqrtf((float)head_dim);

    // 共享内存计算: s_Q (每 block 4行) + s_K (1行) + s_V (1行)
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    size_t smem_size = (num_warps * head_dim + WARP_SIZE * head_dim * 2) * sizeof(float);

    // 解锁 A100 共享内存限制,会否超参数
    //cudaFuncSetAttribute(flash_attn_optimized_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);

    
    // 在 Kernel 启动前打印
    printf("Launch Config: B=%d, N_T=%d, H_Q=%d, D=%d, smem=%zu\n", 
       batch_size, target_seq_len, query_heads, head_dim, smem_size);
    
    

    dim3 grid(batch_size * query_heads, (target_seq_len + num_warps - 1) / num_warps);
    dim3 block(BLOCK_SIZE);

    flash_attn_optimized_kernel<T><<<grid, block, smem_size>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim,
        scale, is_causal
    );

    cudaDeviceSynchronize();
    cudaMemcpy(h_o.data(), d_o, q_sz, cudaMemcpyDeviceToHost);

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
