#include <cuda_runtime.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include "simulation.cuh"
#include "common.cuh"


__global__ void kernel() {
    
}


int main() {
    checkCudaErrors(cudaSetDevice(0));

    kernel<<<1,1>>>();
    return 0;
}
