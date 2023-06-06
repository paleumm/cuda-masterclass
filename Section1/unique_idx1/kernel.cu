#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void unique_idx_calc_thrdIdx(int* input) {
	int tid = threadIdx.x;
	printf("threadIdx: %d, value: %d\n", tid, input[tid]);
}

int main() {
	int arr_sz = 8;
	int arr_byte_sz = sizeof(int) * arr_sz;
	int h_data[] = { 23, 54, 54, 3, 23, 65, 32, 9 };

	for (int i = 0; i < arr_sz; i++) {
		printf("%d ", h_data[i]);
	}
	printf("\n\n");

	int* d_data;
	cudaMalloc((void**)&d_data, arr_byte_sz);
	cudaMemcpy(d_data, h_data, arr_byte_sz, cudaMemcpyHostToDevice);

	dim3 block(8);
	dim3 grid(1);

	unique_idx_calc_thrdIdx << <grid, block >> > (d_data);
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}
