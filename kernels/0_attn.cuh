#pragma once

#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// In place row-wise softmax on N x N row-major matrix mat
template <const int BLOCK_ROWS>
__global__ void softmax(int N, float *mat){
    const int tx = threadIdx.x; // row in output
    const int bx = blockIdx.x;

    if(bx * BLOCK_ROWS + tx >= N) return;

    // Fwd by block
    mat += bx * BLOCK_ROWS * N;

    float running_max = -1e100;
    float running_sum = 0;

    for(int i = 0; i < N; ++i){
        float cur_val = mat[tx * N + i];
        float old_max = running_max;
        if(running_max < cur_val){
            running_max = cur_val;
        }
        running_sum = running_sum * expf(old_max - running_max) + expf(cur_val - running_max);
    }

    for(int i = 0; i < N; ++i){
        float cur_val = mat[tx * N + i];

        mat[tx * N + i] = expf(cur_val - running_max) / running_sum;
    }
}

// Num cols (K, V) and Num rows (Q) to store in shared
// BLOCK = (BR, BC)
// Naive Mat Mul, return out with shape=(m x n), stride=(n, 1)
template <const int BLOCK_ROWS, const int BLOCK_COLS>
__global__ void mat_mul(int m, int n, int k, float *A, int a_stride0, int a_stride1, float *B, int b_stride0, int b_stride1, float *out){
    const int tx = threadIdx.x; // row in output
    const int ty = threadIdx.y; // col in output
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    if(tx + bx * BLOCK_ROWS >= m || ty + by * BLOCK_COLS >= n) return;

    // Fwd A/B/out to block starting pos
    A += bx * BLOCK_ROWS * a_stride0;
    B += by * BLOCK_COLS * b_stride1;
    out += bx * BLOCK_ROWS * n + by * BLOCK_COLS * 1;

    float ret = 0.0;
    for(int i = 0; i < k; ++i){
        ret += A[tx * a_stride0 + i * a_stride1] * B[i * b_stride0 + ty * b_stride1];
    }

    out[tx * n + ty * 1] = ret;
}
