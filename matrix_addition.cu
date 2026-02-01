#include <cuda_runtime.h>

__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    int i = threadIdx.x + (blockDim.x * blockIdx.x);
    int j = threadIdx.y + (blockDim.y * blockIdx.y);
    if (i >= N || j >= N) {
        return;
    }
    C[(i * N) + j] = A[(i * N) + j] + B[(i * N) + j];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                          (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
