#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <cuda_runtime.h>
#include <ThunderKittens/kittens.cuh>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define LOG2_E 1.4426950408889634f
#define INF 1e30f

#define NUM_THREADS (kittens::WARP_THREADS)

template <int BR, int BC>
struct globals{
    // non-1 dimensions are {seq_len, d_head}
    using _Qgl  = kittens::gl2<float, -1, -1, kittens::st_fl<BR, 16>>;
    using _KVgl  = kittens::gl2<float, -1, -1, kittens::st_fl<BC, 16>>;
    using _Ogl  = kittens::gl2<float, -1, -1, kittens::st_fl<BR, BC>>;
    _Qgl Q;
    _KVgl K, V;
    _Ogl O;
};

// Num cols (K, V) and Num rows (Q) to store in shared
// Q, K, V and out are seq_len x d_head row-major
// In out, (BR x BC) blocks are processed at a time
// Each block solves a BR x d region of out
// For now, no TMA, no tensor cores, no swizzling, no vectorized types
// For now, each block is 1 warp, BR | 32
template<int BR, int BC>
__global__ __launch_bounds__(NUM_THREADS, 1)
void tk_flash_attn(const __grid_constant__ globals<BR, BC> g) {
    // 16-byte aligned dynamic shm
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);
    kittens::st_fl<BR, 16> (&q_s) = al.allocate<kittens::st_fl<BR, 16>>();
    kittens::st_fl<BC, 16> (&k_s) = al.allocate<kittens::st_fl<BC, 16>>();
    kittens::st_fl<BC, 16> (&v_s) = al.allocate<kittens::st_fl<BC, 16>>();
    kittens::st_fl<BR, BC> (&out_s) = al.allocate<kittens::st_fl<BR, BC>>();
}
