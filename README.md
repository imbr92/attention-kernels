### My Implementations of standard dot-product attention in various formats/degrees of sophistication

#### Included Kernels
- Naive Unfused Attention (Naive Mat Mul Kernel, Online Softmax Kernel): Done
- Flash Attention 2, no tensor cores: In Progress
- Flash Attention 2, with tensor cores: Not Started
- Flash Attention 3: Not Started

### Misc

To Run
- With Debug Checks (Slow): `nvcc main.cu runner.cu -o exec -DDEBUG && ./exec <kernel_num>`
- Without Debug Checks: nvcc main.cu runner.cu -o exec && ./exec <kernel_num>

This has only been tested on an RTX 3070 (compute capability 8.6) and CUDA 12.8

#### Credits
- Benchmarking code from [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) (which was from [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE))

