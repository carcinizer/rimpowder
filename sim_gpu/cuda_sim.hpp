#pragma once
#include "gpu_sim.cuh"
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iterator>
#include <memory.h>
#include <memory>
#include <vector>
#include <time.h>
#include <chrono>


#include "drawables/buffer_drawable.hpp"

class simulation_pixel_adapter;

class GPUsim{
    private:
        Sand* particles_;
        Sand* particles_p_;
        //values for reistance medium
        float viscosity_, density_;
        //values for simulation time duration
        float max_time_, time_step_;
        uint32_t max_it_, particle_num_;
        // CUDA vals
        int device_;
    public:
        GPUsim(float viscosity, float density, float max_time, uint32_t max_it, uint32_t particle_num_, const std::chrono::steady_clock::time_point& start_point);
        __host__ void simStep(int i, double delta_time_s);
        friend __global__ void sim_kernel(GPUsim&);
        uint32_t getParticleNum();
        simulation_pixel_adapter get_display_adapter(buffor_drawable_ptr<uint32_t>&);

        ~GPUsim(){};
        __host__ void collect();
        __host__ Sand& operator[](size_t idx);
};

class simulation_pixel_adapter {
    private:
        Sand*& hardware_ptr;
        uint64_t hardware_particles_len;
        buffor_drawable_ptr<uint32_t> pix_buff;
        uint32_t* gpu_pix_buff;
    public:
        simulation_pixel_adapter(Sand*& hardware_particles, buffor_drawable_ptr<uint32_t>& buffor, uint64_t hardware_particles_len);
        void set_hwd_ptr(Sand*& ptr);
        void actuate();
};