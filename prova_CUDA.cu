#include <stdio.h>
#include <cuda_runtime.h>

__global__ void loop(int N)
{
  int i = threadIdx.x;
  printf("This is iteration number %d\n", i);
}

int main()
{
  int N = 10;

  loop<<<1, N>>>(N);

  cudaDeviceSynchronize();
  return 0
}
