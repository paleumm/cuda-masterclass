#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void unique_idx_calc_thrdIdx(int* input) {
	int tid = threadIdx.x;
	printf("threadIdx: %d, value: %d\n", tid, input[tid]);
}

__global__ void unique_gid_calc(int* input) {
	int tid = threadIdx.x;
	int offset = blockIdx.x * blockDim.x;
	int gid = tid + offset;
	printf("blockIdx.x: %d, threadIdx: %d, gid: %d, value: %d\n", blockIdx.x, tid, gid, input[gid]);
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

	dim3 block(4);
	dim3 grid(4);

	unique_idx_calc_thrdIdx << <grid, block >> > (d_data);
	cudaDeviceSynchronize();

	unique_gid_calc << <grid, block >> > (d_data);
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}
