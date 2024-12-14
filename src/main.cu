#include <cuda_runtime.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include "simulation.cuh"
#include "common.cuh"
#include <stdio.h>
#include <iostream>
#include <string>

__global__ void kernel() {
    
}


int main(int argc, char** argv) {
    if(argc!=2){
        std::cout <<"Wrong number of arguments\n" <<std::endl;
        return 0;
    }
    std::string sim_output = "sim_output.png";
    std::string arg_str = argv[1];

    std::cout <<"Starting simulation\n" <<std::endl;

    checkCudaErrors(cudaSetDevice(0));

    Simulation sim(arg_str);
    

    kernel<<<1,1>>>();

    sim.save(sim_output);
    return 0;
}
