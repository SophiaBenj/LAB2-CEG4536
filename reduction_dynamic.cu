%%writefile reduction_dynamic.cu
// Tâche 3

#include <stdio.h>
#include <cuda.h>

__global__ void reduction_kernel(int *input, int *output, int size) {
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


    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }

}

__global__ void reduction_with_dynamic_parallelism(int *input, int *output, int size) {

    int num_blocks = (size + 255) / 256;

 

    if (size <= 256) {

        reduction_kernel<<<1, size, size * sizeof(int)>>>(input, output, size);

    } else {

        reduction_kernel<<<num_blocks, 256, 256 * sizeof(int)>>>(input, output, size);

        if (blockIdx.x == 0 && threadIdx.x == 0) {

            reduction_with_dynamic_parallelism<<<1, num_blocks, num_blocks * sizeof(int)>>>(output, output, num_blocks);

        }

    }

}

 

int main() {

    const int size = 1024;

    int h_input[size], h_output[1];

    int *d_input, *d_output;

 

    for (int i = 0; i < size; i++) {

        h_input[i] = 1;

    }

 

    cudaMalloc((void**)&d_input, size * sizeof(int));

    cudaMalloc((void**)&d_output, sizeof(int));

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

 

    reduction_with_dynamic_parallelism<<<1, 256, 256 * sizeof(int)>>>(d_input, d_output, size);

 

    cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

 

    printf("Sum with dynamic parallelism: %d\n", h_output[0]);

 

    cudaFree(d_input);

    cudaFree(d_output);

    return 0;

}