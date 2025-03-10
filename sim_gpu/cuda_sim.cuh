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
        __host__ void simStep(int i, float delta_time_s);
        friend __global__ void sim_kernel(GPUsim&);
        uint32_t getParticleNum();

        ~GPUsim(){};
        __host__ void collect();
        __host__ Sand& operator[](size_t idx);
};

class simulation_pixel_adapter {
    simulation_pixel_adapter(Sand* hardware_particles, buffor_drawable<uint32_t> buffor);
};