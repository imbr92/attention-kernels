#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define LOG2_E 1.4426950408889634f
#define INF 1e30f

__device__ inline float fast_exp(float x){
    return exp2f(LOG2_E * x);
}

// Switch to WMMA
template <int BR, int BC, int MAX_D_HEAD>
__device__ inline void compute_QKT(const float (&Q_tile)[BR][MAX_D_HEAD], const float (&KT_tile)[MAX_D_HEAD][BC], const int d_head, float (&S_tile)[BR][BC]){
    // Low thread util in warp unless BR * BC >= warpSize
#ifdef DEBUG
    assert(BR * BC >= warpSize);
#endif

    for(int offset = 0; offset < BR * BC; offset += warpSize){
        const int tx = (threadIdx.x + offset) % BC;
        const int ty = (threadIdx.x + offset) / BC;
        float reg = 0.0f;
        for(int i = 0; i < d_head; ++i){
            reg += Q_tile[ty][i] * KT_tile[i][tx];
        }

        S_tile[ty][tx] = reg;
    }

#ifdef DEBUG
    if(threadIdx.x == 0 && blockIdx.x == 0){
        printf("Q = [[");
        for(int i = 0; i < BR; ++i){
            for(int j = 0; j < d_head; ++j){
                printf("%f, ", Q_tile[i][j]);
            }
            printf("],\n[");
        }

        printf("\nKT = [[");
        for(int i = 0; i < d_head; ++i){
            for(int j = 0; j < BC; ++j){
                printf("%f, ", KT_tile[i][j]);
            }
            printf("],\n[");
        }

        printf("\nS = [[");
        for(int i = 0; i < BR; ++i){
            for(int j = 0; j < BC; ++j){
                printf("%f, ", S_tile[i][j]);
            }
            printf("],\n[");
        }
    }
#endif

}

// Switch to WMMA
template <int BR, int BC, int MAX_D_HEAD>
__device__ inline void compute_PV(const float (&P_tile)[BR][BC], const float (&V_tile)[BC][MAX_D_HEAD], const int d_head, float (&temp)[BR][MAX_D_HEAD]){
    for(int offset = 0; offset < BR * d_head; offset += warpSize){
        const int tx = (threadIdx.x + offset) % d_head;
        const int ty = (threadIdx.x + offset) / d_head;
        float reg = 0.0f;
        for(int i = 0; i < BC; ++i){
            reg += P_tile[ty][i] * V_tile[i][tx];
        }

        temp[ty][tx] = reg;
    }
}

template<int BR, int BC, int MAX_D_HEAD>
__device__ inline void compute_PlmO(float (&S_tile)[BR][BC], float (&l_vec)[BR], float (&m_vec)[BR], float (&O_tile)[BR][MAX_D_HEAD], int d_head, const float (&V_tile)[BC][MAX_D_HEAD]){
    const int tx = threadIdx.x;

    __shared__ float cur_max[BR];
    // Low thread util in warp
    if(tx < BR){
        cur_max[tx] = -INF;

        for(int i = 0; i < BC; ++i){
            cur_max[tx] = max(cur_max[tx], S_tile[tx][i]);
        }

        cur_max[tx] = max(cur_max[tx], m_vec[tx]);
        l_vec[tx] =  fast_exp(m_vec[tx] - cur_max[tx]) * l_vec[tx];
    }

#ifdef DEBUG
    assert(blockDim.x == warpSize);
#endif
    __syncwarp();

    // Low thread util in warp unless BR * BC >= warpSize
    assert(BR * BC >= warpSize);
    for(int offset = 0; offset < BR * BC; offset += warpSize){
        const int tx = (threadIdx.x + offset) % BC;
        const int ty = (threadIdx.x + offset) / BC;
        S_tile[ty][tx] = fast_exp(S_tile[ty][tx] - cur_max[ty]);
    }

    for(int offset = 0; offset < BR * d_head; offset += warpSize){
        const int tx = (threadIdx.x + offset) % d_head;
        const int ty = (threadIdx.x + offset) / d_head;
        O_tile[ty][tx] = 1/fast_exp(cur_max[ty] - m_vec[ty]) * O_tile[ty][tx];
    }

#ifdef DEBUG
    assert(blockDim.x == warpSize);
#endif

    __shared__ float temp[BR][MAX_D_HEAD];

    __syncwarp();


    // Compute P @ V --> (BR, d_head)
    compute_PV(S_tile, V_tile, d_head, temp);

    // Only need to do before O_tile update
    __syncwarp();

    // Low thread util in warp
    if(tx < BR){

        m_vec[tx] = cur_max[tx];

        float cur_sum = 0.0f;

        for(int i = 0; i < BC; ++i){
            cur_sum = cur_sum + S_tile[tx][i];
        }

        l_vec[tx] = l_vec[tx] + cur_sum;
    }

    for(int offset = 0; offset < BR * d_head; offset += warpSize){
        const int tx = (threadIdx.x + offset) % d_head;
        const int ty = (threadIdx.x + offset) / d_head;
        O_tile[ty][tx] = O_tile[ty][tx] + temp[ty][tx];
    }
}

template<int BR, int MAX_D_HEAD>
__device__ inline void normalize_O(float (&O_tile)[BR][MAX_D_HEAD], float (&l_vec)[BR], int d_head){
    for(int offset = 0; offset < BR * d_head; offset += warpSize){
        const int row = (threadIdx.x + offset) % d_head;
        const int col = (threadIdx.x + offset) / d_head;
        O_tile[row][col] /= l_vec[row];
    }
}


// Num cols (K, V) and Num rows (Q) to store in shared
// Q, K, V and out are seq_len x d_head row-major
// In out, (BR x BC) blocks are processed at a time
// Each block solves a BR x d region of out
// For now, no TMA, no tensor cores, no swizzling, no vectorized types
// For now, each block is 1 warp, BR | 32
template <int BR, int BC, int MAX_D_HEAD>
__global__ void flash_attn(const int seq_len, const int d_head, const float *Q, const float *K, const float *V, float *out){

    // TODO: Move to asserts to runner
    // Remove once adding dynamic smem?

#ifdef DEBUG
    assert(MAX_D_HEAD >= d_head);
    assert(blockDim.x == warpSize);
#endif

    // Load BR x d region of Q
    // Loop through d x BC regions of K^T (and BC x d regions of V)

    // Load BR x d region of Q
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    __shared__ float Q_tile[BR][MAX_D_HEAD];
    __shared__ float KT_tile[MAX_D_HEAD][BC];
    __shared__ float V_tile[BC][MAX_D_HEAD];
    __shared__ float S_tile[BR][BC];
    __shared__ float l_vec[BR];
    __shared__ float m_vec[BR];
    __shared__ float O_tile[BR][MAX_D_HEAD];

    if(threadIdx.x < BR){
        l_vec[threadIdx.x] = 0.0f;
        m_vec[threadIdx.x] = -INF;
    }

#ifdef DEBUG
    assert(BR * d_head % warpSize == 0);
#endif

    for(int i = 0; i < BR * d_head; i += warpSize){
        const int idx = i + tid;
        const int Q_row = (idx / d_head) + bid * BR;
        const int Q_col = (idx % d_head);
        Q_tile[idx / d_head][idx % d_head] = Q[Q_row * d_head + Q_col];
    }

#ifdef DEBUG
    assert(seq_len % BC == 0);
    if(threadIdx.x == 0){
        for(int i = 0; i < BR; ++i){
            for(int j = 0; j < d_head; ++j){
                assert(Q[(bid * BR * d_head + i * d_head + j)] == Q_tile[i][j]);
            }
        }
    }
#endif


    // Main Loop, col offset in K^T
    for(int col_offset = 0; col_offset < seq_len; col_offset += BC){

        // Load BC x d region of K (d x BC region of K^T)
        for(int i = 0; i < BC * d_head; i += warpSize){
            const int idx = i + tid;
            const int K_row = (idx / d_head) + col_offset;
            const int K_col = (idx % d_head);
            KT_tile[idx % d_head][idx / d_head] = K[K_row * d_head + K_col];
        }

#ifdef DEBUG
    if(threadIdx.x == 0){
        for(int i = 0; i < BC; ++i){
            for(int j = 0; j < d_head; ++j){
                assert(K[(col_offset * d_head + i * d_head + j)] == KT_tile[j][i]);
            }
        }
    }
#endif

        // Load BC x d region of V
        for(int i = 0; i < BC * d_head; i += warpSize){
            const int idx = i + tid;
            // col offset in KT is same as row offset in V
            const int V_row = (idx / d_head) + col_offset;
            const int V_col = (idx % d_head);
            V_tile[idx / d_head][idx % d_head] = V[V_row * d_head + V_col];
        }

#ifdef DEBUG
    if(threadIdx.x == 0){
        for(int i = 0; i < BC; ++i){
            for(int j = 0; j < d_head; ++j){
                assert(V[(col_offset * d_head + i * d_head + j)] == V_tile[i][j]);
            }
        }
    }
#endif

        // Compute Q @ KT --> shared memory tile of size (BR, BC)
        // Can do concurrently with loading of V tile to SMEM
        compute_QKT(Q_tile, KT_tile, d_head, S_tile);

        // In place in S_tile for P, l and m in smem, also update O_tile
        compute_PlmO(S_tile, l_vec, m_vec, O_tile, d_head, V_tile);

        // Now S_tile = P_tile + l, m and O are up to date
    }

    normalize_O(O_tile, l_vec, d_head);

    // Write O_tile to GMEM
    out += blockIdx.x * BR * d_head;

    for(int i = 0; i < BR * d_head; i += warpSize){
        const int idx = i + tid;
        const int O_row = (idx / d_head);
        const int O_col = (idx % d_head);
        out[O_row * d_head + O_col] = O_tile[idx / d_head][idx % d_head];
    }

}
