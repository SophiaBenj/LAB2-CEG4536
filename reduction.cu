%%writefile reduction.cu
// Tâche 2

#include <cuda_runtime.h>
#include <iostream>

#define cudaCheckError() {                                          \
    cudaError_t e = cudaGetLastError();                             \
    if (e != cudaSuccess) {                                         \
        std::cout << "CUDA error: " << cudaGetErrorString(e) << std::endl; \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

__global__ void reduce(int *input, int *output, int size) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? input[i] : 0;

    __syncthreads();


    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];

}

 

int main() {
    const int size = 1024;
    int input[size], output[size / 256];
    int *d_input, *d_output;


    for (int i = 0; i < size; i++) {
        input[i] = 1;
    }

    int cpu_sum = 0;
    for (int i = 0; i < size; i++) {
        cpu_sum += input[i];
    }

    std::cout << "Somme sur le CPU avant copie : " << cpu_sum << std::endl;

    cudaMalloc(&d_input, size * sizeof(int));

    cudaMalloc(&d_output, (size / 256) * sizeof(int));

    cudaCheckError();

    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);

    cudaCheckError();

    reduce<<<size / 256, 256, 256 * sizeof(int)>>>(d_input, d_output, size);

    cudaCheckError();

    cudaDeviceSynchronize();

    cudaCheckError();

    cudaMemcpy(output, d_output, (size / 256) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaCheckError();

    int sum = 0;
    for (int i = 0; i < size / 256; i++) {
        sum += output[i];
    }

    std::cout << "La somme totale est : " << sum << std::endl;

    cudaFree(d_input);

    cudaFree(d_output);

    return 0;

}