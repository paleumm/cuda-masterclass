/*
Programming Exercise 2:
64 elements array, imdex the 3D grid
grid -> 4 x 4 x 4
block -> 2 x 2 x 2
name: Permpoon
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void print_3d_grid(int* data, int size) {
	int tid = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

	int num_threads_per_block = blockDim.x * blockDim.y * blockDim.z;
	int block_offset = num_threads_per_block * (gridDim.x * blockIdx.y + blockIdx.x);

	int num_threads_per_page = num_threads_per_block * gridDim.x * gridDim.y;
	int page_offset = num_threads_per_page * blockIdx.z;

	int gid = tid + block_offset + page_offset;
	
	if(gid < size)
		printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x, tid, gid, data[gid]);
}

int main() {

	// init
	int size = 64;
	int byte_size = size * sizeof(int);

	int* h_data;
	h_data = (int*)malloc(byte_size);

	// random
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++) {
		h_data[i] = (int)(rand() & 0xff);
	}

	//0 - 63 (no random)
	//for (int i = 0; i < size; i++) {
	//	h_data[i] = i;
	//}

	printf("Input Data\n");

	for (int i = 0; i < size; i++) {
		printf("%d ", h_data[i]);
	}

	printf("\n\n");

	// input with format
	//for (int i = 0; i < size; i++) {
	//	if (i % 2 == 0) {
	//		printf("\n");
	//		if (i % 4 == 0)
	//			printf("\n");
	//	}
	//	printf("%d ", h_data[i]);
	//}
	//printf("\n\n");

	int* d_data;
	cudaMalloc((void**)&d_data, byte_size);

	cudaMemcpy(d_data, h_data, byte_size, cudaMemcpyHostToDevice);

	dim3 block(2, 2, 2);
	dim3 grid(2, 2, 2);

	printf("blk.z\tblk.y\tblk.x\tthrd.z\tthrd.y\tthrd.x\ttid\tgid\tdata\n");
	print_3d_grid << <grid, block >> > (d_data, size);
	cudaDeviceSynchronize();

	cudaFree(d_data);
	free(h_data);


	cudaDeviceReset();
	return 0;
}