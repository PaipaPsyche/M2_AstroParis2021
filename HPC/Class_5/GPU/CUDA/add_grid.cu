#include <iostream>
#include <math.h>

// function to add the elements of two arrays

__global__ void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
     y[i] = x[i] + y[i];
}

int main(void)
{
  int RUNS = 1000;
  int N = 1<<20; // 1M elements
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
     x[i] = 1.0f;
     y[i] = 2.0f;
  }
  
  // Prefetch the data to the GPU
  int device = -1;
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(x, N*sizeof(float), device, NULL);
  cudaMemPrefetchAsync(y, N*sizeof(float), device, NULL);

  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize -1) / blockSize;

  for (int i = 0; i < RUNS; i++)
      add<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be RUNS+2.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
      maxError = fmax(maxError, fabs(y[i]-2.0f-RUNS));
  std::cout << "Max error: " << maxError << std::endl;
     
  // Free memory
  cudaFree(x);
  cudaFree(y);
  return 0;
}  
