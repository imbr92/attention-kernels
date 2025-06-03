### Progressive Implementations of Standard Dot-Product Attention

#### Included Kernels
- Naive Unfused Attention (Naive Mat Mul Kernel, Online Softmax Kernel): Done
- Flash Attention 2, no tensor cores: Done
- Flash Attention 2, with tensor cores: Not Started
- Flash Attention 3: Not Started

### Misc

To Run
- Standard: `nvcc -std=c++20 --gpu-architecture=sm_86 --expt-relaxed-constexpr main.cu runner.cu -o exec && ./exec <kernel_num>`
- Debug (Slow): `nvcc -std=c++20 -DDEBUG --gpu-architecture=sm_86 --expt-relaxed-constexpr main.cu runner.cu -o exec && ./exec <kernel_num>`
- Profiling: `nvcc -std=c++20 main.cu runner.cu -o exec -DPROFILE --gpu-architecture=sm_86 --expt-relaxed-constexpr && sudo ncu ./exec <kernel_num>`

### Dependencies
- This has only been tested on an RTX 3070 (compute capability 8.6) and CUDA 12.8
- Requires [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens)

#### Credits
- Benchmarking code from [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) (which was from [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE))

