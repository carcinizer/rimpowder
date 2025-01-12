#include "gpu_sim.cuh"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>


Sand::Sand(uint16_t x,uint16_t y):x0_(x),y0_(y){

// zero values
v0x_ = 0.1;
v0y_ = 0.7;
ax0_ = 0;
ay0_ = 9.81;
//values for the first iteration they will be used
// to save previous iteration values
vx_pre_ = v0x_;
vy_pre_ = v0y_;
ax_pre_ = ax0_;
ay_pre_ = ay0_;
x_pre_  = x0_;
y_pre_  = y0_;

vx_ = 0;

ax_ = 0;
vy_ = 0;




R_= 0.1;
mass_ = 1600*(4*3.14*R_*R_*R_/3);//Density of sand is 1600 kg/m3
C_D_ = 0.45; //value for a sphere
S_D_ = 2*3.14*R_*R_; // half of the area of a sphere
Fxw_ = 0;
Fyw_ = 0;
}

//case appropriate for gravity
__device__ __host__ void Sand::calculateNextLocation(double time_step, float density, float viscosity){
    /* https://pl.wikipedia.org/wiki/Op%C3%B3r_aero(hydro)dynamiczny
    Re = (p*v*2r)/n
    if Re<1
    F=-6*pi*n*r*v;
    else
    F=C_D*(p*abs(V)^2)/2*S_D

    Fw = F- Fo = Fx - Fox = m*ax-Fox;
    */
    float v_w = 0;
    float Re = 0;
    float awx,awy;

    //make n iteration values n-1 iteration

    vx_pre_ = vx_;
    vy_pre_ = vy_;
    x_pre_  = x_;
    y_pre_  = y_;
    ax_pre_ = ax_;
    ay_pre_ = ay_;

    v_w = std::sqrt(vx_pre_*vx_pre_+vy_pre_*vy_pre_);

    //Hydro/Areodynamic resistance calcuations
    Re = density*v_w*2*R_/viscosity;
    // split resistance to x and y components and
    // calculate x and y opposing force vectors.
    if(Re<1){
        Fxw_ = mass_*ax0_  -6*3.14*viscosity*R_*vx_pre_;
        Fyw_ = mass_*ay0_  -6*3.14*viscosity*R_*vy_pre_;
    }
    else
    {
        Fxw_ = mass_*ax0_ - C_D_*S_D_*(density*abs(vx_pre_)*abs(vx_pre_))/2;
        Fyw_ = mass_*ay0_ - C_D_*S_D_*(density*abs(vy_pre_)*abs(vy_pre_))/2;
    }


    // calculate movement restricted by opposing force
    // aw = Fw/m
    ax_ = Fxw_/mass_;
    ay_ = Fyw_/mass_;

    vx_ = vx_pre_ + ax_*time_step;//Fxw_/mass_*time_step;
    vy_ = vy_pre_ + ay_*time_step; //Fyw_/mass_*time_step;

    x_ = x_pre_ + (vx_*time_step) + (ax_*time_step*time_step/2);
    y_ = y_pre_ + (vy_*time_step) + (ay_*time_step*time_step/2);

    //time_ += time_s;
}

void Sand::setMovementParam(float velocity, float acceleration){
    ax_ = acceleration;
    v0x_ = velocity;

}

void Sand::getPosition(float &x, float &y){
    x = x_;
    y = x_;
}

float Sand::getX() {
    return x_;
}

float Sand::getY() {
    return y_;
}

float Sand::getVelocity(){
 return vx_;
}

float Sand::getAcceleration(){
 return ax_;
}

void Sand::printPosition(){
    std::cout << " x = " << x_ << " y = " << y_ ;
     std::cout <<" vx = " << vx_ <<" vy = " << vy_ <<std::endl;
}
