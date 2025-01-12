#pragma once
#include <cstdint>
#include <iostream>
#include <string>
#include <math.h>
#include <cmath>
#include <cuda_runtime.h>

#define K 512
#define SIM_SIZE 100000
#define SIM_STEPS 10000
#define MAX_TIME 10 //time over 3.5 causes some oscilations for air density and viscosity


typedef struct timespec app_timer_t;
#define timer(t_ptr) clock_gettime(CLOCK_MONOTONIC, t_ptr)
void elapsed_time(app_timer_t start, app_timer_t stop);

class sim_element{
    private:

    public:
    //virtual void calculateNextLocation(float time_step, float density, float viscosity) = 0;
    virtual void getPosition(float &x, float &y) = 0;
    virtual void setMovementParam(float velocity, float acceleration) = 0;
    virtual float getVelocity() = 0;
    virtual float getAcceleration() = 0;
};

// sand is a sphere
class Sand: public sim_element{
    private:

    //variables for zero values
    float  x0_, v0x_, ax0_; 
    float  y0_, v0y_, ay0_; 

    //variables for saving values from previous iteration
    float y_pre_, vy_pre_, ay_pre_;
    float x_pre_, vx_pre_, ax_pre_;

    //variables for saving values from current iteration
    float Fyw_, y_, vy_, ay_;
    float Fxw_, x_, vx_, ax_;

    // values for resistance calcuations
    float R_,C_D_,S_D_;
    float mass_;
    public:
    Sand(uint16_t x,uint16_t y);
    __device__ __host__ void calculateNextLocation(double time_step, float density, float viscosity);
    void getPosition(float &x, float &y);
    __host__ __device__ float getX();
    __host__ __device__ float getY();
    void setMovementParam(float velocity, float acceleration);
    float getVelocity();
    float getAcceleration();
    void printPosition();
    ~Sand(){};

};
