#include "gpu_sim.cuh"
#include "cuda_sim.cuh"
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <iterator>
#include <memory.h>
#include <memory>
#include <vector>
#include <time.h>


typedef struct timespec app_timer_t;
#define timer(t_ptr) clock_gettime(CLOCK_MONOTONIC, t_ptr)
void elapsed_time(app_timer_t start, app_timer_t stop)
{
	double etime;
	etime = 1e+3 * (stop.tv_sec - start.tv_sec) +
		1e-6 * (stop.tv_nsec - start.tv_nsec);
	printf("GPU (total!) time = %.3f ms\n",
		etime);
}

__host__
static void sim_sand_kernel(GPUsim &sim){
    sim.simStep(1);
    //if(i<sim.getParticleNum()){
      //  sim.simStep(i);
     //   particles_p_[i].calculateNextLocation(time_step_, density_, viscosity_);
    //}
}

Sand *particle_vec_p;

int main(){
   // Sand particle_vec_p[sim_size];
   //     cudaEvent_t start, stop;


    //float time_step = (float)MAX_TIME/SIM_STEPS;
    float n = 0.0000107; //dynamic viscosity of liquid  in Pa*s
    float p = 1;// Density of the liquid in kg/m3
    GPUsim testSim(n,p,
    MAX_TIME,SIM_STEPS,SIM_SIZE);
    //checkCudaErrors(cudaSetDevice(0));
    std::cout<<"before:" << std::endl;
    testSim[1].printPosition();
    sim_sand_kernel(testSim);
    testSim.collect();
    std::cout<<"after:" << std::endl;
    testSim[1].printPosition();

    //checkCudaErrors(cudaMalloc((void**)&particle_vec, (particle_vec.size()-1)*sizeof(Sand)));
    //checkCudaErrors(cudaMemcpy(particle_vec_p, particle_vec, (particle_vec.size()-1)*sizeof(Sand), cudaMemcpyHostToDevice));


   // elapsed_time(start,stop);
    return 0;
};

