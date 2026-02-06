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

// --- 安全搬运函数：支持向量化与标量回退 ---
template<typename T>
__device__ __forceinline__ void safe_load_to_smem(const T* src, float* dst, int D, int lane_id) {
    // 如果 D 是 4 的倍数且是 float/half 类型，尝试向量化
    if (D % 4 == 0) {
        const float4* src_v4 = reinterpret_cast<const float4*>(src);
        float4* dst_v4 = reinterpret_cast<float4*>(dst);
        for (int d4 = lane_id; d4 < D / 4; d4 += WARP_SIZE) {
            float4 tmp = src_v4[d4];
            dst_v4[d4] = tmp;
        }
    } else {
        // 标量回退 (处理 D=1, 2, 3 等情况)
        for (int d = lane_id; d < D; d += WARP_SIZE) {
            dst[d] = (float)src[d];
        }
    }
}

// -----------------------------------------------------------------------------
// Flash Attention 向量化安全 Kernel
// -----------------------------------------------------------------------------
template <typename T>
__global__ void flash_attn_kernel(
    const T* Q, const T* K, const T* V, T* O,
    int B, int N_target, int N_src, int H_q, int H_kv, int D,
    float scale, bool is_causal
) {
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    int warp_global_id = blockIdx.y * num_warps + warp_in_block;
    int lane_id = threadIdx.x % WARP_SIZE;

    int batch_idx = blockIdx.x / H_q;
    int head_q_idx = blockIdx.x % H_q;
    int head_kv_idx = head_q_idx / (H_q / H_kv);

    // 共享内存布局
    extern __shared__ float s_mem[];
    float* s_Q_base = s_mem; 
    float* s_K = s_mem + num_warps * D;
    float* s_V = s_K + WARP_SIZE * D;

    // 1. 加载 Q (必须确保即便无效的 Warp 也参与同步)
    if (warp_global_id < N_target) {
        const T* q_ptr = Q + ((batch_idx * N_target + warp_global_id) * H_q + head_q_idx) * D;
        safe_load_to_smem(q_ptr, s_Q_base + warp_in_block * D, D, lane_id);
    }
    __syncthreads(); // 关键同步

    // 如果 Warp 超出范围，停止计算但不退出，以保持同步一致性 (或者在此之后不再有同步)
    if (warp_global_id >= N_target) return;

    // 寄存器初始化 (支持最大 D=128 的向量化存储)
    float acc_o[128]; 
    #pragma unroll
    for (int i = 0; i < 128; ++i) acc_o[i] = 0.0f;
    
    float row_m = -FLT_MAX;
    float row_l = 0.0f;

    // 2. 外层分块循环
    for (int j_start = 0; j_start < N_src; j_start += WARP_SIZE) {
        int j_src = j_start + lane_id;
        
        // 协作加载 K, V
        if (j_src < N_src) {
            const T* k_ptr = K + ((batch_idx * N_src + j_src) * H_kv + head_kv_idx) * D;
            const T* v_ptr = V + ((batch_idx * N_src + j_src) * H_kv + head_kv_idx) * D;
            safe_load_to_smem(k_ptr, s_K + lane_id * D, D, 0); // 这里简写，由于是单线程负责一行
            safe_load_to_smem(v_ptr, s_V + lane_id * D, D, 0);
        }
        __syncthreads();

        // 3. 计算分数
        float local_s = -FLT_MAX;
        if (j_src < N_src && (!is_causal || j_src <= warp_global_id)) {
            local_s = 0.0f;
            for (int d = 0; d < D; ++d) {
                local_s += s_Q_base[warp_in_block * D + d] * s_K[lane_id * D + d];
            }
            local_s *= scale;
        }

        // 4. Softmax 统计更新
        float max_val = warpReduceMax(local_s);
        float new_m = fmaxf(row_m, max_val);
        float p_scale = expf(row_m - new_m);
        float exp_s = (local_s == -FLT_MAX) ? 0.0f : expf(local_s - new_m);
        float new_l = row_l * p_scale + warpReduceSum(exp_s);

        // 5. 更新 O (向量化展开)
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            acc_o[d] *= p_scale;
            // 每一个线程通过 shuffle 拿当前块内所有线程的 exp_s
            for (int k = 0; k < WARP_SIZE; ++k) {
                float e = __shfl_sync(0xffffffff, exp_s, k);
                acc_o[d] += e * s_V[k * D + d];
            }
        }

        row_m = new_m;
        row_l = new_l;
        __syncthreads();
    }

    // 6. 写回
    for (int d = 0; d < D; ++d) {
        O[((batch_idx * N_target + warp_global_id) * H_q + head_q_idx) * D + d] = (T)(acc_o[d] / row_l);
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

  dim3 grid(batch_size * query_heads, (target_seq_len + num_warps - 1) / num_warps);
  dim3 block(BLOCK_SIZE);

  // 解锁 A100 共享内存限制,会否超参数
  //cudaFuncSetAttribute(flash_attn_optimized_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);

    
  // 在 Kernel 启动前打印
  printf("Launch Config: B=%d, N_T=%d, H_Q=%d, D=%d, smem=%zu\n", batch_size, target_seq_len, query_heads, head_dim, smem_size);
    

  flash_attn_kernel<T><<<grid, block, smem_size>>>(
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
