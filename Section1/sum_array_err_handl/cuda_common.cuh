#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans)); }

inline void gpuAssert(cudaError_t code, bool abort = true){
	if (code != cudaSuccess){
		fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(code));
		if (abort) exit(code);
	}
}
