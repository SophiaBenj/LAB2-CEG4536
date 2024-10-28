%%writefile parallel_multiplication.cu
// Tâche 1

#include <iostream>
#include <cuda.h>

__global__ void reduction(int *input, int *output, int n) {
   extern __shared__ int shared_data[];
   int tid = threadIdx.x;
   int index = blockIdx.x * blockDim.x + threadIdx.x;

   shared_data[tid] = (index < n) ? input[index] : 0;
   __syncthreads();

   for (int s = blockDim.x / 2; s > 0; s >>= 1) {
       if (tid < s) {
           shared_data[tid] += shared_data[tid + s];
       }
       __syncthreads();
   }

   if (tid == 0) output[blockIdx.x] = shared_data[0];
}

int main() {
   const int n = 1024;
   int size = n * sizeof(int);
   int *input, *output;
   int *d_input, *d_output;

   input = (int*)malloc(size);
   output = (int*)malloc(size);

   for (int i = 0; i < n; i++) input[i] = 1;

   cudaMalloc(&d_input, size);
   cudaMalloc(&d_output, size);
   cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

   int block_size = 256;
   int grid_size = (n + block_size - 1) / block_size;

   int num_elements = n;
   while (num_elements > 1) {
       reduction<<<grid_size, block_size, block_size * sizeof(int)>>>(d_input, d_output, num_elements);
       
       num_elements = grid_size;
       grid_size = (num_elements + block_size - 1) / block_size;

       int *temp = d_input;
       d_input = d_output;
       d_output = temp;
   }

   int result;
   cudaMemcpy(&result, d_input, sizeof(int), cudaMemcpyDeviceToHost);

   std::cout << "Résultat final (sur GPU) : " << result << std::endl;

   cudaFree(d_input);
   cudaFree(d_output);
   free(input);
   free(output);

   return 0;
}