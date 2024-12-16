
#include "cuda_sim.cuh"
#include "gpu_sim.cuh"
#include "helper_cuda.h"

#include <cstdint>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>



GPUsim::GPUsim(float viscosity, float density, float max_time, uint32_t max_it, uint32_t particle_num):
        viscosity_(viscosity),
        density_(density),
        max_time_(max_time),
        max_it_(max_it),
        particle_num_(particle_num)
{

    device_ = 0;
    checkCudaErrors(cudaSetDevice(device_));

    particles_ = (Sand*)std::malloc(sizeof(Sand)*particle_num);

    for(int i=0; i< particle_num_;i++){
    //sand particle location should be changed
    //but for simple test it should not matter
        particles_[i] = Sand(0,0);
    }

    checkCudaErrors(cudaMalloc(&particles_p_, 2*particle_num_* sizeof( Sand)));

    checkCudaErrors(cudaMemcpy(particles_p_, particles_,
    2*particle_num_* sizeof( Sand), cudaMemcpyHostToDevice));

}

__global__ void sim_kernel(Sand* particles, uint32_t particle_size, float time_step, float density, float viscosity){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<particle_size) {
       particles[i].calculateNextLocation(time_step, density, viscosity);
    }
}

__host__ void GPUsim::simStep(int i){
    cudaEvent_t start, stop;

    //particles_p_[i].calculateNextLocation(time_step_,
     //density_, viscosity_);

    //checkCudaErrors(cudaEventCreate(&start));
	//checkCudaErrors(cudaEventCreate(&stop));
	//checkCudaErrors(cudaEventRecord(start, 0));
    std::cout<< "Joł start" << std::endl;
    sim_kernel<<<(particle_num_+K-1)/K, K>>>(
        particles_p_,
        particle_num_,
        time_step_,
        density_,
        viscosity_
    );
    checkCudaErrors(cudaGetLastError());
}

__host__ void GPUsim::collect() {
    checkCudaErrors(cudaMemcpy(particles_, particles_p_, 2*particle_num_* sizeof( Sand), cudaMemcpyDeviceToHost));
}

__host__ Sand& GPUsim::operator[](size_t idx) {
    return particles_[idx];
}

uint32_t GPUsim::getParticleNum() {
    return particle_num_;
}



