
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <stdlib.h>
#include <time.h>

#include <cstring>

#include "cuda_common.cuh"

void compare_arrays(int* a, int* b, int size);

__global__ void sum_array(int* a, int* b, int* c, int size) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size) {
		c[gid] = a[gid] + b[gid];
	}
}

void sum_array_cpu(int* a, int* b, int* c, int size) {
	for (int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
	}
}

int main() {
	int size = 1 << 25;
	int byte_size = size * sizeof(int);

	int block_size = 1024;

	cudaError err;

	int* h_a, * h_b, * h_c, * results;

	h_a = (int*)malloc(byte_size);
	h_b = (int*)malloc(byte_size);
	h_c = (int*)malloc(byte_size);
	results = (int*)malloc(byte_size);

	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++) {
		h_a[i] = (int)(rand() & 0xff);
	}

	for (int i = 0; i < size; i++) {
		h_b[i] = (int)(rand() & 0xff);
	}

	sum_array_cpu(h_a, h_b, h_c, size);

	memset(results, 0, byte_size);

	int* d_a, * d_b, * d_c;

	gpuAssert(cudaMalloc((int**)&d_a, byte_size));
	gpuAssert(cudaMalloc((int**)&d_b, byte_size));
	gpuAssert(cudaMalloc((int**)&d_c, byte_size));
	
	cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice);

	dim3 block(block_size);
	dim3 grid((size / block.x) + 1);

	sum_array << <grid, block >> > (d_a, d_b, d_c, size);
	cudaDeviceSynchronize();

	cudaMemcpy(results, d_c, byte_size, cudaMemcpyDeviceToHost);

	// array comparison
	compare_arrays(h_c, results, size);

	/*for (int i = 0; i < size; i++) {
		if (i % 128 == 0) printf("\n\n");

		printf("%d ", results[i]);
	}*/

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(h_a);
	free(h_b);
	free(results);

	cudaDeviceReset();
	return 0;
}

void compare_arrays(int* a, int* b, int size) {
	for (int i = 0; i < size; i++) {
		if (a[i] != b[i]) {
			printf("Arrays are different\n");
			return;
		}
	}
	printf("Arrays are same\n");
}