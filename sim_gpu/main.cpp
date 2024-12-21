#include "disp/window.hpp"
#include "gpu_sim.cuh"
#include "cuda_sim.hpp"
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "primitives/vec2.hpp"
#include <chrono>
#include <cstdint>
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
static void sim_sand_kernel(GPUsim &sim, bool init_time){
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

Sand *particle_vec_p;

int main(){
   // Sand particle_vec_p[sim_size];
   //     cudaEvent_t start, stop;


    //float time_step = (float)MAX_TIME/SIM_STEPS;
    float n = 0.0000107; //dynamic viscosity of liquid  in Pa*s
    float p = 1;// Density of the liquid in kg/m3
    GPUsim testSim(n,p,
    MAX_TIME,SIM_STEPS,SIM_SIZE, std::chrono::steady_clock::now());
    //checkCudaErrors(cudaSetDevice(0));
    std::cout<<"before:" << std::endl;
    testSim[1].printPosition();

    vec2<int> resolution{1280, 720};
    disp::Window main_wnd("test fizyki", resolution);
    if(main_wnd.initialise()) {
        return -1;
    }

    auto pix_art = std::make_shared<buffor_drawable<uint32_t>>(resolution.x, resolution.y);
    auto adapter = testSim.get_display_adapter(pix_art);
    main_wnd.add_drawable(pix_art);

    int dummy;
    std::cin >> dummy;

    sim_sand_kernel(testSim, true);
    for(int idx = 0; idx < 1000; idx ++) {
        main_wnd.update();
        if(main_wnd.should_close())
            break;
        main_wnd.clear(0x0);
        sim_sand_kernel(testSim, false);

        adapter.actuate();

        testSim.collect();
        testSim[1].printPosition();

        main_wnd.draw();
    }
    testSim.collect();
    std::cout<<"after:" << std::endl;
    testSim[1].printPosition();

    //checkCudaErrors(cudaMalloc((void**)&particle_vec, (particle_vec.size()-1)*sizeof(Sand)));
    //checkCudaErrors(cudaMemcpy(particle_vec_p, particle_vec, (particle_vec.size()-1)*sizeof(Sand), cudaMemcpyHostToDevice));


   // elapsed_time(start,stop);
    return 0;
};

