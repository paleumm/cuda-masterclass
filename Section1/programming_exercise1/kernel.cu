/*

Programming Exercise 1:
print threadIdx, blockIdx, gridDim in each dimension
grid with 4 threads in X, Y, Z
thread block size 2 in X, Y, Z

name: Permpoon
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_details() {
	printf("ThreadIdx.x : %d, ThreadIdx.y : %d,ThreadIdx.z : %d, BlockdIdx.x : %d, BlockdIdx.y : %d, BlockdIdx.z : %d, gridDim.x : %d, gridDim.y : %d, gridDim.z : %d\n",
		threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z);
}

int main() {
	int nx, ny, nz;
	nx = 4;
	ny = 4;
	nz = 4;

	dim3 block(2, 2, 2);
	dim3 grid(nx / block.x, ny / block.y, nz / block.z);

	print_details << <grid, block >> > ();
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}