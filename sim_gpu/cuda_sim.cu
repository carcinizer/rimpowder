
#include "cuda_sim.hpp"
#include "gpu_sim.cuh"
#include "helper_cuda.h"

#include <cstdint>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <mutex>



GPUsim::GPUsim(float viscosity, float density, float max_time, uint32_t max_it, uint32_t particle_num,  const std::chrono::steady_clock::time_point& start):
        viscosity_(viscosity),
        density_(density),
        max_time_(max_time),
        max_it_(max_it),
        particle_num_(particle_num)
{

    device_ = 0;
    checkCudaErrors(cudaSetDevice(device_));

    particles_ = (Sand*)std::malloc(sizeof(Sand)*particle_num);
    std::cout << " setting up buffors " << std::endl;
    if(particles_ == nullptr) {
        std::cout <<" error malloc" << std::endl;
    }

    for(int i=0; i< particle_num_;i++){
    //sand particle location should be changed
    //but for simple test it should not matter
        particles_[i] = Sand(i,10);
    }
    std::cout <<" loop " << std::endl;

    checkCudaErrors(cudaMalloc(&particles_p_, particle_num_* sizeof( Sand)));
    std::cout <<" cuda malloc " << std::endl;

    checkCudaErrors(cudaMemcpy(particles_p_, particles_,
    particle_num_* sizeof( Sand), cudaMemcpyHostToDevice));

    std::cout << " end of setting up buffors " << std::endl;
}

__global__ void sim_kernel(Sand* particles, uint32_t particle_size, double time_step, float density, float viscosity){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<particle_size) {
       particles[i].calculateNextLocation(time_step, density, viscosity);
    }
}

__host__ void GPUsim::simStep(int i, double delta_time_s = 0.f){
    cudaEvent_t start, stop;

    //particles_p_[i].calculateNextLocation(time_step_,
     //density_, viscosity_);

    //checkCudaErrors(cudaEventCreate(&start));
	//checkCudaErrors(cudaEventCreate(&stop));
	//checkCudaErrors(cudaEventRecord(start, 0));
    sim_kernel<<<(particle_num_+K-1)/K, K>>>(
        particles_p_,
        particle_num_,
        delta_time_s,
        density_,
        viscosity_
    );
    checkCudaErrors(cudaGetLastError());
}

__host__ void GPUsim::collect() {
    checkCudaErrors(cudaMemcpy(particles_, particles_p_, particle_num_* sizeof( Sand), cudaMemcpyDeviceToHost));
}

__host__ Sand& GPUsim::operator[](size_t idx) {
    return particles_[idx];
}


simulation_pixel_adapter GPUsim::get_display_adapter(buffor_drawable_ptr<uint32_t>& pix_buff) {
    return std::move(simulation_pixel_adapter(particles_p_, pix_buff, particle_num_));
}

uint32_t GPUsim::getParticleNum() {
    return particle_num_;
}

simulation_pixel_adapter::simulation_pixel_adapter(Sand*& hardware_particles, buffor_drawable_ptr<uint32_t>& buffor, uint64_t hardware_particles_len):
    hardware_ptr(hardware_particles),
    hardware_particles_len(hardware_particles_len),
    pix_buff(buffor)
{
    auto res = pix_buff->getAABB();
    std::cout << res.x << " by " << res.y << std::endl;
    checkCudaErrors(cudaMalloc(&gpu_pix_buff, sizeof(uint32_t)* res.x * res.y ));
}

__global__ void convert_to_pixels_kernel(Sand* sim_ptr, uint64_t sim_len, uint32_t* pix_buff, uint64_t x_res, uint64_t y_res) {
    size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx > sim_len) return;
    //printf("(idx: %lu, tx: %d, blk: %d)\n", idx, threadIdx.x, blockIdx.x);
    uint64_t x = static_cast<uint64_t>(sim_ptr[idx].getX());
    uint64_t y = static_cast<uint64_t>(sim_ptr[idx].getY());

    if(x >= x_res) return;
    if(y >= y_res) return;

    pix_buff[x + y*x_res] = 0xFF0000FF;
}

void simulation_pixel_adapter::actuate() {
    auto res = pix_buff->getAABB();
    size_t mem_size_bytes = sizeof(uint32_t)* res.x * res.y;
    checkCudaErrors(cudaMemset(gpu_pix_buff, 0, mem_size_bytes));
    std::cout<< hardware_particles_len << std::endl;
    convert_to_pixels_kernel<<<(hardware_particles_len+K-1)/K, K>>>(
        hardware_ptr,
        hardware_particles_len,
        gpu_pix_buff,
        res.x,
        res.y
    );
    std::cout<< std::endl;
    cudaDeviceSynchronize();
    std::lock_guard<std::mutex> lg(pix_buff->get_mtx());
    cudaMemcpy(*(pix_buff->get()), gpu_pix_buff, mem_size_bytes, cudaMemcpyDeviceToHost);
}
