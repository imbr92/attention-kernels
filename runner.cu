#include "kernels.cuh"
#include "runner.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>

float get_sec(){
    struct timeval time;
    gettimeofday(&time, NULL);
    return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end){ return 1.0e-6 * (end - beg); }

void cudaCheck(cudaError_t error, const char *file, int line){
    if(error != cudaSuccess){
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

void CudaDeviceInfo(){
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, deviceId);

    printf("Device ID: %d\n\
            Name: %s\n\
            Compute Capability: %d.%d\n\
            memoryBusWidth: %d\n\
            maxThreadsPerBlock: %d\n\
            maxThreadsPerMultiProcessor: %d\n\
            maxRegsPerBlock: %d\n\
            maxRegsPerMultiProcessor: %d\n\
            totalGlobalMem: %zuMB\n\
            sharedMemPerBlock: %zuKB\n\
            sharedMemPerMultiprocessor: %zuKB\n\
            totalConstMem: %zuKB\n\
            multiProcessorCount: %d\n\
            Warp Size: %d\n",
           deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
           props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
           props.regsPerBlock, props.regsPerMultiprocessor,
           props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
           props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
           props.multiProcessorCount, props.warpSize
    );
};

void randomize_matrix(float *mat, int N){
    // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
    // precision is too low and the same random number is generated.
    struct timeval time{};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for(int i = 0; i < N; i++){
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

void range_init_matrix(float *mat, int N){
    for(int i = 0; i < N; i++){
        mat[i] = i;
    }
}

void zero_init_matrix(float *mat, int N){
    for(int i = 0; i < N; i++){
        mat[i] = 0.0;
    }
}

void copy_matrix(const float *src, float *dest, int N){
    int i;
    for(i = 0; src + i && dest + i && i < N; i++)
        *(dest + i) = *(src + i);
    if(i != N)
        printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs){
    int i;
    fs << std::setprecision(2) << std::fixed; // Set floating-point precision and fixed notation
    fs << "[";
    for(i = 0; i < M * N; i++){
        if((i + 1) % N == 0)
            fs << std::setw(5) << A[i]; // Set field width and write the value
        else
            fs << std::setw(5) << A[i] << ", ";
        if((i + 1) % N == 0){
            if(i + 1 < M * N)
                fs << ";\n";
        }
    }
    fs << "]\n";
}

bool verify_matrix(float *matRef, float *matOut, int N){
    double diff = 0.0;
    int i;
    for(i = 0; i < N; i++){
        diff = std::fabs(matRef[i] - matOut[i]);
        if(diff > 0.01){
            printf("Divergence! Should %5.2f, Is %5.2f(Diff %5.2f) at %d\n",
                   matRef[i], matOut[i], diff, i);
            return false;
        }
    }
    return true;
}

void run_attn_naive(int d_head, int seq_len, float *Q, float *K,
                    float *V, float *out, float *mat){

    // First Q @ K^T
    dim3 mm1_gridDim(CEIL_DIV(seq_len, 32), CEIL_DIV(seq_len, 32));
    constexpr dim3 mm1_blockDim(32, 32);
    // First matrix is M x K and has stride (K, 1)
    // Second matrix is K x N and has stride (1, K)
    int m = seq_len, n = seq_len;
    int k = d_head;
    mat_mul<mm1_blockDim.x, mm1_blockDim.y><<<mm1_gridDim, mm1_blockDim>>>(m, n, k, Q, k, 1, K, 1, k, mat);
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());

    // Now mat is (QK^T), row-major, softmax by row
    // No benefit in block being > warp size since we don't use shared mem?
    dim3 s_gridDim(CEIL_DIV(seq_len, 32));
    constexpr dim3 s_blockDim(32);
    softmax<s_blockDim.x><<<s_gridDim, s_blockDim>>>(n, mat);
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
    // Finally another mat mul between mat and V
    // TODO: Switch up
    dim3 mm2_gridDim(CEIL_DIV(seq_len, 32), CEIL_DIV(d_head, 32));
    constexpr dim3 mm2_blockDim(32, 32);
    mat_mul<mm2_blockDim.x, mm2_blockDim.y><<<mm2_gridDim, mm2_blockDim>>>(m, k, n, mat, n, 1, V, k, 1, out);
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
}

// template <int BR, int BC, int MAX_D_HEAD>
// __global__ void flash_attn(const int seq_len, const int d_head, const float *Q, const float *K, const float *V, float *out){

void run_attn_1(int d_head, int seq_len, float *Q, float *K,
                    float *V, float *out){

    const int BR = 4;
    const int BC = 8;
    const int MAX_D_HEAD = 64;
    const int WARP_SIZE = 32;
    assert(MAX_D_HEAD >= d_head);
    dim3 fa_gridDim(CEIL_DIV(seq_len, BR));
    constexpr dim3 fa_blockDim(WARP_SIZE);

    // cudaFuncAttributes attr;
    // cudaFuncGetAttributes(&attr, flash_attn<BR, BC, MAX_D_HEAD>);
    // printf("Static shared memory usage: %zu bytes\n", attr.sharedSizeBytes);
    flash_attn<BR, BC, MAX_D_HEAD><<<fa_gridDim, fa_blockDim>>>(seq_len, d_head, Q, K, V, out);
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
}


void run_kernel(int kernel_num, int d_head, int seq_len, float *Q,
                float *K, float *V, float *out, float *mat){
    switch(kernel_num){
        case 0:
            run_attn_naive(d_head, seq_len, Q, K, V, out, mat);
            break;
        case 1:
            run_attn_1(d_head, seq_len, Q, K, V, out);
            break;
        default:
            throw std::invalid_argument("Unknown kernel number");
    }
}
