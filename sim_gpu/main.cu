#include "disp/window.hpp"
#include "gpu_sim.cuh"
#include "cuda_sim.hpp"
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "primitives/vec2.hpp"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
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
static void sim_sand_kernel_iterative(GPUsim &sim, bool init_time){
    using std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::seconds;

    static steady_clock::time_point last_iter;
    if(init_time) {
        last_iter = steady_clock::now();
    }
    steady_clock::time_point now = steady_clock::now();
    double delta_time = duration_cast<std::chrono::microseconds>(now - last_iter).count();
    last_iter = now;
    //std::cout<< "dt..." << delta_time*1e6 << std::endl;
    sim.simStep(1, delta_time/1e6);
    //if(i<sim.getParticleNum()){
    //  sim.simStep(i);
    //   particles_p_[i].calculateNextLocation(time_step_, density_, viscosity_);
    //}
}

__host__
static void sim_sand_kernel(GPUsim &sim, float time_step){

    sim.simStep(1, time_step);

}

Sand *particle_vec_p;

int main(int argc, char *argv[]){
   // Sand particle_vec_p[sim_size];
    cudaEvent_t start, stop;
    float time;
    float time_step = (float)MAX_TIME/SIM_STEPS;
    float n = 0.00000107; //dynamic viscosity of liquid  in Pa*s
    float p = 1000;// Density of the liquid in kg/m3
    char gui_en[] = "-gui";
    GPUsim testSim(n,p,
    MAX_TIME,SIM_STEPS,SIM_SIZE, std::chrono::steady_clock::now());
    checkCudaErrors(cudaSetDevice(0));
    std::cout<<"Running simulation for:"<<std::endl;
    std::cout<<SIM_SIZE<<" Particles"<<std::endl;
    std::cout<<"Time step: "<< time_step<<std::endl;
    std::cout<<SIM_STEPS<<" Steps"<<std::endl;
    std::cout<<"before:" << std::endl;
    testSim[1].printPosition();


    sim_sand_kernel_iterative(testSim, true);
    if(argc == 1){
        std::cout << "Running simulation without GUI" << std::endl;

        checkCudaErrors( cudaEventCreate(&start) );
        checkCudaErrors( cudaEventCreate(&stop) );
        checkCudaErrors( cudaEventRecord(start, 0) );
        for(int k = 0; k<SIM_STEPS; k++){
            sim_sand_kernel(testSim, time_step);
        }
        checkCudaErrors( cudaEventRecord(stop, 0) );
        checkCudaErrors( cudaEventSynchronize(stop) );
        checkCudaErrors( cudaEventElapsedTime(&time, start, stop) );
        printf("GPU time  %3.1f ms \n", time);

    }else if(argc > 1 && strcmp(argv[1],gui_en)==0){

        vec2<int> resolution{1280, 720};
        disp::Window main_wnd("test fizyki", resolution);
        if(main_wnd.initialise()) {
            return -1;
        }

        auto pix_art = std::make_shared<buffor_drawable<uint32_t>>(resolution.x, resolution.y);
        auto adapter = testSim.get_display_adapter(pix_art);
        main_wnd.add_drawable(pix_art);

        for(int idx = 0; idx < 1000; idx ++) {
            main_wnd.update();
            if(main_wnd.should_close())
                break;
            main_wnd.clear(0x0);
           sim_sand_kernel_iterative(testSim, false);

            adapter.actuate();

           testSim.collect();
           testSim[1].printPosition();

           main_wnd.draw();
        }
    } else {
        std::cout << "could not deduce simulation parameters -gui?" << std::endl;
    }
    testSim.collect();
    std::cout<<"after:" << std::endl;
    testSim[1].printPosition();

    //checkCudaErrors(cudaMalloc((void**)&particle_vec, (particle_vec.size()-1)*sizeof(Sand)));
    //checkCudaErrors(cudaMemcpy(particle_vec_p, particle_vec, (particle_vec.size()-1)*sizeof(Sand), cudaMemcpyHostToDevice));


   // elapsed_time(start,stop);
    return 0;
};

