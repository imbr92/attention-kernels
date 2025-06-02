#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <array>
#include <assert.h>
#include <algorithm>

#include "runner.cuh"

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "flashAttnValidationFailure.txt";

const int MIN_KERNEL_NO = 0;
const int MAX_KERNEL_NO = 1;


// Head vector dim (already multiplied W_Q @ X, etc. to get Q, K, V)
const std::array<int, 1> d_head = {8};
const std::array<int, 1> seq_len = {16};
// const std::array<int, 1> d_head = {64};
// const std::array<int, 1> seq_len = {128};

const int MAX_D_HEAD = *std::max_element(d_head.begin(), d_head.end());
const int MAX_SEQ_LEN = *std::max_element(seq_len.begin(), seq_len.end());


int main(int argc, char **argv){
    if(argc != 2){
        std::cerr << "Please select a kernel (range 0 - 12, 0 for ref kernel)"
            << std::endl;
        exit(EXIT_FAILURE);
    }

    // get kernel number
    int kernel_num = std::stoi(argv[1]);
    if(kernel_num < MIN_KERNEL_NO || kernel_num > MAX_KERNEL_NO){
        // TODO: FIX
        std::cerr << "Please enter a valid kernel number(0-12)" << std::endl;
        exit(EXIT_FAILURE);
    }

    // get environment variable for device
    int deviceIdx = 0;
    if(getenv("DEVICE") != NULL){
        deviceIdx = atoi(getenv("DEVICE"));
    }
    cudaCheck(cudaSetDevice(deviceIdx));

    printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

    // print some device info
    CudaDeviceInfo();

    // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
    // publishing event tasks in the target stream
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    float *Q = nullptr, *K = nullptr, *V = nullptr, *out = nullptr, *ref = nullptr;
    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dout = nullptr, *dref = nullptr;
    // Temporaries to hold seq_len dim vector following softmax and seq_len^2 matrix following QKT for naive case
    float *dmat = nullptr;

    Q = (float *)malloc(sizeof(float) * MAX_SEQ_LEN * MAX_D_HEAD);
    K = (float *)malloc(sizeof(float) * MAX_SEQ_LEN * MAX_D_HEAD);
    V = (float *)malloc(sizeof(float) * MAX_SEQ_LEN * MAX_D_HEAD);
    out = (float *)malloc(sizeof(float) * MAX_SEQ_LEN * MAX_D_HEAD);
    ref = (float *)malloc(sizeof(float) * MAX_SEQ_LEN * MAX_D_HEAD);

    randomize_matrix(Q, MAX_SEQ_LEN * MAX_D_HEAD);
    randomize_matrix(K, MAX_SEQ_LEN * MAX_D_HEAD);
    randomize_matrix(V, MAX_SEQ_LEN * MAX_D_HEAD);

    cudaCheck(cudaMalloc((void **)&dQ, sizeof(float) * MAX_SEQ_LEN * MAX_D_HEAD));
    cudaCheck(cudaMalloc((void **)&dK, sizeof(float) * MAX_SEQ_LEN * MAX_D_HEAD));
    cudaCheck(cudaMalloc((void **)&dV, sizeof(float) * MAX_SEQ_LEN * MAX_D_HEAD));
    cudaCheck(cudaMalloc((void **)&dref, sizeof(float) * MAX_SEQ_LEN * MAX_D_HEAD));
    cudaCheck(cudaMalloc((void **)&dout, sizeof(float) * MAX_SEQ_LEN * MAX_D_HEAD));
    // Hopefully not too big to materialize for naive check
    cudaCheck(cudaMalloc((void **)&dmat, sizeof(float) * MAX_SEQ_LEN * MAX_SEQ_LEN));

    cudaCheck(cudaMemcpy(dQ, Q, sizeof(float) * MAX_SEQ_LEN * MAX_D_HEAD,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dK, K, sizeof(float) * MAX_SEQ_LEN * MAX_D_HEAD,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dV, V, sizeof(float) * MAX_SEQ_LEN * MAX_D_HEAD,
                         cudaMemcpyHostToDevice));

    int repeat_times = 50;
    for(int dhead : d_head){
        for(int seqlen : seq_len){
            std::cout << "Dimensions: d_head=" << dhead << ", seq_len= " << seqlen << std::endl;

            // Not tested without this
            assert(dhead == MAX_D_HEAD);
            assert(seqlen == MAX_SEQ_LEN);

        // Verify the correctness of the calculation, and execute it once before the
        // kernel function timing to avoid cold start errors
        if(kernel_num != 0){
            run_kernel(0, dhead, seqlen, dQ, dK, dV, dref, dmat);
            run_kernel(kernel_num, dhead, seqlen, dQ, dK, dV, dout, dmat);
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
            cudaMemcpy(ref, dref, sizeof(float) * dhead * seqlen, cudaMemcpyDeviceToHost);
            cudaMemcpy(out, dout, sizeof(float) * dhead * seqlen, cudaMemcpyDeviceToHost);

            if(!verify_matrix(ref, out,  dhead)){
            // if(true){
                std::cout
                    << "Failed to pass the correctness verification against Naive Attention Implementation "
                    << std::endl;
                if(seqlen * dhead < 256){
                    std::cout << " Logging faulty output into " << errLogFile << "\n";
                    std::ofstream fs;
                    fs.open(errLogFile);
                    fs << "Q:\n";
                    print_matrix(Q, seqlen, dhead, fs);
                    fs << "K:\n";
                    print_matrix(K, seqlen, dhead, fs);
                    fs << "V:\n";
                    print_matrix(V, seqlen, dhead, fs);
                    fs << "expected output:\n";
                    print_matrix(ref, seqlen, dhead, fs);
                    fs << "output:\n";
                    print_matrix(out, seqlen, dhead, fs);
                }
                exit(EXIT_FAILURE);
            }
        }

        cudaEventRecord(beg);
        for(int j = 0; j < repeat_times; j++){
            // We don't reset dC between runs to save time
            run_kernel(kernel_num, dhead, seqlen, dQ, dK, dV, dout, dmat);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.; // Convert to seconds

        // TODO: Double check calc
        // approximate: QK^T = 2 N^2 * d, softmax = N^2, x * V = 2Nd
        int64_t flops = 2LL * seqlen * (seqlen + 1) * dhead + seqlen * seqlen;
        printf(
            "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. seq_len: "
            "(%d), d_head: (%d).\n",
            elapsed_time / repeat_times,
            (repeat_times * flops * 1e-9) / elapsed_time, seqlen, dhead);
        fflush(stdout);
        // TODO: Do I need this?
        // make dC and dC_ref equal again (we modified dC while calling our kernel
        // for benchmarking)
        // cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
        //                      cudaMemcpyDeviceToDevice));
        }
    }

    // Free up CPU and GPU space
    free(Q);
    free(K);
    free(V);
    free(ref);
    free(out);
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dref);
    cudaFree(dout);
    return 0;
};
