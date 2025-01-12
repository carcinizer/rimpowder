#include "txt_sim.h"
#include <memory.h>
#include <memory>
#include <vector>
#include <time.h> 

#define sim_size 100000
#define sim_steps 10000
#define max_time 10 

typedef struct timespec app_timer_t;
#define timer(t_ptr) clock_gettime(CLOCK_MONOTONIC, t_ptr)
void elapsed_time(app_timer_t start, app_timer_t stop)
{
	double etime;
	etime = 1e+3 * (stop.tv_sec - start.tv_sec) +
		1e-6 * (stop.tv_nsec - start.tv_nsec);
	printf("CPU (total!) time = %.3f ms\n",
		etime);
}


static void sim_sand_kernel(Sand &sand, float time_step, float p, float n){
    sand.calculateNextLocation(time_step,p,n);
}



int main(){
    std::vector<std::shared_ptr<Sand>> particle_vec;
    app_timer_t start, stop;
    float time_step = (float)max_time/sim_steps;
    float n = 0.00000107; //dynamic viscosity of liquid  in Pa*s
    float p = 1000;// Density of the liquid in kg/m3
    for(int i = 0; i< sim_size; i++){
        particle_vec.push_back(std::make_shared<Sand>(0,i));
    }
    particle_vec[0]->printPosition();

    timer(&start);
    for(int k = 0; k<sim_steps; k++){
        for(int i = 0; i< particle_vec.size()-1; i++){
            sim_sand_kernel(*particle_vec[i],time_step,p,n);
        }
    }
    
    timer(&stop);
    particle_vec[0]->printPosition();
    elapsed_time(start,stop);
    return 0;
};

