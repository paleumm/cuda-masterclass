
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>



__global__ void print_warp_details() {
	int gid = blockIdx.y * gridDim.x * blockIdx.x + blockDim.x * blockIdx.x + threadIdx.x;
	
	int warp_id = threadIdx.x / 32;

	// block index
	int gbid = blockIdx.y * gridDim.x + blockIdx.x;

	printf("tid: %d, bid.x: %d, bid.y: %d, gid: %d, warp_id: %d, gbid: %d\n", threadIdx.x, blockIdx.x, blockIdx.y, gid, warp_id, gbid);
}

int main() {

	dim3 block(42);
	dim3 grid(2, 2);

	print_warp_details << <grid, block >> > ();
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return EXIT_SUCCESS;
}