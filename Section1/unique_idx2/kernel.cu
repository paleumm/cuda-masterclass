#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void unique_gid_calc_2d_2d(int* input) {
	int tid = blockDim.x * threadIdx.y + threadIdx.x;

	int num_threads_per_block = blockDim.x * blockDim.y;
	int block_offset = blockIdx.x * num_threads_per_block;

	int num_threads_per_row = num_threads_per_block * gridDim.x;
	int row_offset = num_threads_per_row * blockIdx.y;

	int gid = tid + block_offset + row_offset;
	printf("gridDim.x: %d, blockIdx.x: %d, blockIdx.y: %d,,threadIdx: %d, gid: %d, value: %d\n", gridDim.x, blockIdx.x, blockIdx.y, tid, gid, input[gid]);
}

int main() {
	int arr_sz = 16;
	int arr_byte_sz = sizeof(int) * arr_sz;
	int h_data[] = { 23, 54, 54, 3, 23, 65, 32, 9, 10, 2, 33, 342, 45, 654, 7, 234 };

	for (int i = 0; i < arr_sz; i++) {
		printf("%d ", h_data[i]);
	}
	printf("\n\n");

	int* d_data;
	cudaMalloc((void**)&d_data, arr_byte_sz);
	cudaMemcpy(d_data, h_data, arr_byte_sz, cudaMemcpyHostToDevice);

	dim3 block(2, 2);
	dim3 grid(2, 2);

	unique_gid_calc_2d_2d << <grid, block >> > (d_data);
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}
