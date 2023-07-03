#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_common.cuh"

__global__ void sum_arrays_1Dgrid_1Dblock(float* a, float* b, float* c, int nx) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	c[gid] = a[gid] + b[gid];
}

__global__ void sum_arrays_2Dgrid_2Dblock(float* a, float* b, float* c, int nx, int ny) {
	int gidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gidy = blockIdx.y * blockDim.y + threadIdx.y;

	int gid = gidy * nx + gidx;

	if(gidx < nx && gidy < ny) 
		c[gid] = a[gid] + b[gid];
}

void sum_array_cpu(float* a, float* b, float* c, int size) {
	for (int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
	}
}

void compare_arrays(float* a, float* b, float size) {
	for (int i = 0; i < size; i++) {
		if (a[i] != b[i]) {
			printf("Arrays are different \n");
			return;
		}
	}
	printf("Arrays are same \n");
}

void run_sum_array_1D(int argc, char** argv) {
	printf("Running 1D grid\n");
	int size = 1 << 22;
	int block_size = 128;

	int nx, ny = 0;

	if (argc > 2) 
		size = 1 << atoi(argv[2]);
	
	if (argc > 4)
		block_size = 1 << atoi(argv[4]);

	unsigned int byte_size = size * sizeof(float);

	printf("Input Size : %d \n", size);

	float* h_a, * h_b, * h_out, * h_ref;
	h_a = (float*)malloc(byte_size);
	h_b = (float*)malloc(byte_size);
	h_out = (float*)malloc(byte_size);
	h_ref = (float*)malloc(byte_size);

	if (!h_a)
		printf("Host memory allocation error\n");

	for (size_t i = 0; i < size; i++) {
		h_a[i] = i % 10;
		h_b[i] = i % 7;
	}

	sum_array_cpu(h_a, h_b, h_out, size);

	dim3 block(block_size);
	dim3 grid((size + block.x - 1) / block.x);

	printf("Kernel is lauch with grid(%d,%d,%d) and block(%d,%d,%d) \n",
		grid.x, grid.y, grid.z, block.x, block.y, block.z);

	float* d_a, * d_b, * d_c;

	gpuErrchk(cudaMalloc((void**)&d_a, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_b, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_c, byte_size));
	gpuErrchk(cudaMemset(d_c, 0, byte_size));

	gpuErrchk(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice));

	sum_arrays_1Dgrid_1Dblock << <grid, block >> > (d_a, d_b, d_c, size);

	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_c, byte_size, cudaMemcpyDeviceToHost));

	compare_arrays(h_out, h_ref);

	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);
	free(h_ref);
	free(h_out);
	free(h_b);
	free(h_a);
}

void run_sum_array_2D(int argc, char** argv) {
	printf("Running 2D grid\n");
	int size = 1 << 22;
	int block_x = 128;

	int nx = 1 << 14;
	int ny = size / nx;
	int block_y = 8;

	if (argc > 2)
		size = 1 << atoi(argv[2]);

	if (argc > 3)
		nx = 1 << atoi(argv[3]);

	ny = size / nx;

	if (argc > 4) {
		int pow = atoi(argv[4]);
		if (pow < 3 || pow > 10) {
			printf("Block size is invalid, default block size used (%d,%d)\n", block_x, block_y);
		}
		else {
			block_x = 1 << pow;
			block_y = 1024 / block_x;
		}
	}

	unsigned int byte_size = size * sizeof(float);

	printf("Input Size : %d \n", size);

	float* h_a, * h_b, * h_out, * h_ref;
	h_a = (float*)malloc(byte_size);
	h_b = (float*)malloc(byte_size);
	h_out = (float*)malloc(byte_size);
	h_ref = (float*)malloc(byte_size);

	if (!h_a)
		printf("Host memory allocation error\n");

	for (size_t i = 0; i < size; i++) {
		h_a[i] = i % 10;
		h_b[i] = i % 7;
	}

	sum_array_cpu(h_a, h_b, h_out, size);

	dim3 block(block_x, block_y);
	dim3 grid((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y);

	printf("Kernel is lauch with grid(%d,%d,%d) and block(%d,%d,%d) \n",
		grid.x, grid.y, grid.z, block.x, block.y, block.z);

	float* d_a, * d_b, * d_c;

	gpuErrchk(cudaMalloc((void**)&d_a, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_b, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_c, byte_size));
	gpuErrchk(cudaMemset(d_c, 0, byte_size));

	gpuErrchk(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice));

	sum_arrays_1Dgrid_1Dblock << <grid, block >> > (d_a, d_b, d_c, size);

	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_c, byte_size, cudaMemcpyDeviceToHost));

	compare_arrays(h_out, h_ref);

	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);
	free(h_ref);
	free(h_out);
	free(h_b);
	free(h_a);
}

int main(int argc, char** argv) {
	printf("\n----------------------- SUM ARRAY EXAMPLE FOR NVPROF ------------------------ \n\n");
	if (argc > 1) {
		if (atoi(argv[1]) > 0) {
			run_sum_array_2D(argc, argv);
		}
		else {
			run_sum_array_1D(argc, argv);
		}
	}
	else {
		run_sum_array_1D(argc, argv);
	}

	return 0;
}