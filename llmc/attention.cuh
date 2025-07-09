/*
Attention, as a fallback when we do not use the Flash Attention from cuDNN
*/
#include <assert.h>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "cublas_common.h"

// ----------------------------------------------------------------------------
// CUDA kernels

#define QK_NORM_EPS 1e-5f
#define QK_NORM_SCALE 1.0f

// inputs floatX, outputs FP32 (for current FP32-only activation path for this WIP)
__global__ void permute_kernel(floatX* q, floatX* k, floatX* v,
                               const floatX* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { return; }

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    q[idx] = __ldcs(&inp[inp_idx]);
    k[idx] = __ldcs(&inp[inp_idx + NH * d]);
    v[idx] = __ldcs(&inp[inp_idx + 2 * (NH * d)]);
}

__global__ void permute_kernel_backward(floatX* dinp,
                                        const floatX* dq, const floatX* dk, const floatX* dv,
                                        int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;

    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    dinp[inp_idx] = dq[idx];
    dinp[inp_idx + NH * d] = dk[idx];
    dinp[inp_idx + 2 * (NH * d)] = dv[idx];
}

__global__ void unpermute_kernel(floatX* inp, floatX *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)

    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    out[other_idx] = __ldcs(&inp[idx]);
}

__global__ void unpermute_kernel_backward(floatX* dinp, const floatX *dout, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    dinp[idx] = (floatX)dout[other_idx];
}

__global__ void qk_norm_forward_kernel(floatX* q, floatX* k, float* qrstd, float* krstd,
                                       float scale, int N, int HS) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    int idx = blockIdx.x * num_warps + warp_id;
    if (idx >= N) return;
    q += idx * HS;
    k += idx * HS;
    float sumq = 0.0f;
    float sumk = 0.0f;
    for(int i = lane_id; i < HS; i += WARP_SIZE) {
        float qv = (float)__ldcs(q + i);
        float kv = (float)__ldcs(k + i);
        sumq += qv * qv;
        sumk += kv * kv;
    }
    sumq = warpReduceSum(sumq);
    sumk = warpReduceSum(sumk);
    float invq = rsqrtf(sumq / HS + QK_NORM_EPS);
    float invk = rsqrtf(sumk / HS + QK_NORM_EPS);
    if (lane_id == 0) {
        qrstd[idx] = invq;
        krstd[idx] = invk;
    }
    float qscale = scale * invq;
    float kscale = scale * invk;
    for(int i = lane_id; i < HS; i += WARP_SIZE) {
        float qv = (float)__ldcs(q + i) * qscale;
        float kv = (float)__ldcs(k + i) * kscale;
        __stcs(q + i, (floatX)qv);
        __stcs(k + i, (floatX)kv);
    }
}

__global__ void qk_norm_backward_kernel(floatX* dq, floatX* dk,
                                         const floatX* dqn, const floatX* dkn,
                                         const floatX* q, const floatX* k,
                                         const float* qrstd, const float* krstd,
                                         float scale, int N, int HS) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    int idx = blockIdx.x * num_warps + warp_id;
    if (idx >= N) return;
    q += idx * HS;
    k += idx * HS;
    dqn += idx * HS;
    dkn += idx * HS;
    dq += idx * HS;
    dk += idx * HS;
    float invq = qrstd[idx];
    float invk = krstd[idx];
    float inv_factor_q = 1.0f / (scale * invq);
    float inv_factor_k = 1.0f / (scale * invk);
    float dotq = 0.0f;
    float dotk = 0.0f;
    for(int i = lane_id; i < HS; i += WARP_SIZE) {
        float xq = (float)__ldcs(q + i) * inv_factor_q;
        float gq = (float)__ldcs(dqn + i);
        dotq += xq * gq;
        float xk = (float)__ldcs(k + i) * inv_factor_k;
        float gk = (float)__ldcs(dkn + i);
        dotk += xk * gk;
    }
    dotq = warpReduceSum(dotq);
    dotk = warpReduceSum(dotk);
    float coeffq1 = scale * invq;
    float coeffq2 = scale * invq * invq * invq / HS;
    float coeffk1 = scale * invk;
    float coeffk2 = scale * invk * invk * invk / HS;
    for(int i = lane_id; i < HS; i += WARP_SIZE) {
        float xq = (float)__ldcs(q + i) * inv_factor_q;
        float gq = (float)__ldcs(dqn + i);
        float xk = (float)__ldcs(k + i) * inv_factor_k;
        float gk = (float)__ldcs(dkn + i);
        __stcs(dq + i, (floatX)(coeffq1 * gq - coeffq2 * xq * dotq));
        __stcs(dk + i, (floatX)(coeffk1 * gk - coeffk2 * xk * dotk));
    }
}

__global__ void relu2_forward_kernel(floatX* out, const floatX* inp, int N, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * T * T;
    if(idx >= total) return;
    int j = idx % T;
    int i = (idx / T) % T;
    if(j <= i) {
        float v = (float)__ldcs(inp + idx);
        float r = fmaxf(v, 0.0f);
        __stcs(out + idx, (floatX)((r * r) * 0.01f));
    } else {
        __stcs(out + idx, (floatX)0.0f);
    }
}

__global__ void relu2_backward_kernel(floatX* dpreatt, const floatX* datt, const floatX* att, int N, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * T * T;
    if(idx >= total) return;
    int j = idx % T;
    int i = (idx / T) % T;
    if(j <= i) {
        float a = (float)__ldcs(att + idx);
        if(a > 0.0f) {
            float grad = (float)__ldcs(datt + idx);
            __stcs(dpreatt + idx, (floatX)(grad * (0.2f * sqrtf(a))));
        } else {
            __stcs(dpreatt + idx, (floatX)0.0f);
        }
    } else {
        __stcs(dpreatt + idx, (floatX)0.0f);
    }
}

__global__ void softmax_forward_kernel5(floatX* out, float inv_temperature, const floatX* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4  == 0);
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // micro-optimization: we iterate backwards so that
    // after the softmax backward operation completes, the cache retains the
    // part of the matrix close to the upper left corner, which benefits the
    // matmul operation that immediately follows.
    // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // forward order
    int idx = (gridDim.x - blockIdx.x - 1) * num_warps + warp_id; // backward order
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const floatX* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    const float flt_max = 340282346638528859811704183484516925440.0f; // to avoid including float.h
    float maxval = -flt_max;
    float sumval = 0.0f;

    const floatX* x_aligned = reinterpret_cast<const floatX*>(__builtin_assume_aligned(x, 16));
    for (int i = lane_id; i < pos_by_4; i += WARP_SIZE) {
        float regarray[4];
        for (int k = 0; k < 4; ++k) {
            regarray[k] = (float)x_aligned[4*i + k];
        }
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, regarray[k]);
        }
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (regarray[k] - maxval));
        }
    }

    if(4*pos_by_4 + lane_id <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, (float)x[4*pos_by_4 + lane_id]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * ((float)x[4*pos_by_4 + lane_id] - maxval));
    }

    float global_maxval = warpReduceMax(maxval);
    sumval *= expf(inv_temperature * (maxval - global_maxval));

    float sum = warpReduceSum(sumval);
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = lane_id; i <= own_pos; i += WARP_SIZE) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = expf(inv_temperature * ((float)__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, (floatX)(ev * norm));
    }
}

__global__ void softmax_autoregressive_backward_inplace_kernel(floatX* datt, const floatX* att,
                                                               int B, int T, int C, float scale) {
    constexpr const int BlockSize = 256;
    constexpr int T_per_block = 4;

    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block*blockIdx.x;
    int idx = blockIdx.y;

    att += idx * T * T;
    datt += idx * T * T;

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const floatX* att_bth = att + t * T;
        const floatX* datt_bth = datt + t * T;
        floatX* dpreatt_bth = datt + t * T;

        float local_sum = 0;
        for (int t2 = threadIdx.x; t2 <= t; t2 += BlockSize) {
            local_sum += (float)att_bth[t2] * (float)datt_bth[t2];
        }

        local_sum = blockReduce<warpReduceSum>(local_sum);

        for (int t3 = threadIdx.x; t3 < T; t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            if(t3 <= t) {
                float acc = (float) __ldcs(att_bth + t3) * ((float) __ldcs(datt_bth + t3) - local_sum);
                __stcs(dpreatt_bth + t3, (floatX) (scale * acc));
            } else {
                // explicitly set non-causal elements to zero
                __stcs(dpreatt_bth + t3, (floatX)0.f);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void attention_forward(floatX* out, floatX* qkvr, float* qrstd, float* krstd,
                       floatX* att, floatX* inp,
                       int B, int T, int C, int NH, cudaStream_t stream) {
    NVTX_RANGE_FN();
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    const int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size, 0, stream>>>(q, k, v, inp, B, T, NH, HS);

    floatX* preatt = inp; // reuse inp as scratch buffer
    matmul_cublaslt(preatt, k, q, nullptr, T, T, HS, stream, true, false, B * NH, T * HS, T * HS, T * T);

    int grid_size = CEIL_DIV(B * NH * T * WARP_SIZE, block_size);
    qk_norm_forward_kernel<<<grid_size, block_size, 0, stream>>>(q, k, qrstd, krstd, QK_NORM_SCALE, B * NH * T, HS);
    relu2_forward_kernel<<<grid_size, block_size, 0, stream>>>(att, preatt, B * NH, T);

    // new approach: first cuBLAS another batched matmul
    floatX* vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    matmul_cublaslt(vaccum, v, att, nullptr, HS, T, T, stream, false, false, B * NH, T * HS, T * T, T * HS);

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size, 0, stream>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
void attention_backward(floatX* dinp, floatX* dqkvr, floatX* datt, floatX* scratch,
                        const floatX* dout,
                        const floatX* qkvr, const float* qrstd, const float* krstd,
                        const floatX* att,
                        int B, int T, int C, int NH, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int HS = C / NH; // head size

    // unpack convenience pointers into q, k, v
    const floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    floatX *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;

    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(scratch, dout, B, T, NH, HS);
    int grid_size;
    // backward into datt
    matmul_cublaslt(datt, v, scratch, nullptr, T, T, HS, stream, true, false, B * NH, T * HS, T * HS, T * T);
    // backward into dv
    matmul_cublaslt(dv, scratch, att, nullptr, HS, T, T, stream, false, true, B * NH, T * HS, T * T, T * HS);
    relu2_backward_kernel<<<CEIL_DIV(B * NH * T * T, 256), 256>>>(datt, datt, att, B * NH, T);
    const floatX* dpreatt = datt;
    // backward into q
    matmul_cublaslt(dq, k, dpreatt, nullptr, HS, T, T, stream, false, false, B * NH, T * HS, T * T, T * HS);
    // backward into k
    matmul_cublaslt(dk, q, dpreatt, nullptr, HS, T, T, stream, false, true, B * NH, T * HS, T * T, T * HS);
    grid_size = CEIL_DIV(B * NH * T * WARP_SIZE, block_size);
    qk_norm_backward_kernel<<<grid_size, block_size, 0, stream>>>(dq, dk, dq, dk, q, k, qrstd, krstd, QK_NORM_SCALE, B * NH * T, HS);
    // backward into inp
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(dinp, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}
