#include "txt_sim.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>


Sand::Sand(uint16_t x,uint16_t y):x0_(x),y0_(y){
x_ = 0;
y_ = 0;
vx_ = 0;
v0x_ = 0.1;
ax_ = 9.81;
vy_ = 0;
v0y_ = 0.7;
ay_ = 0;
time_ = 0;
R_= 0.1;
mass_= 0;
C_D_ = 0.45; //value for a sphere
S_D_ = 2*3.14*R_*R_; // half of the area of a sphere
Fxw_ = 0;
Fyw_ = 0;
}
//case appropriate for gravity
void Sand::calculateNextLocation(float time_step, float p, float n){
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
    time_ += time_step;

    //vx_ += v0x_+ax_*time_;
    //vy_ += v0y_+ay_*time_;

    mass_ = 1600*(4*3.14*R_*R_*R_/3);//Density of sand is 1600 kg/m3

    v_w = std::sqrt(vx_*vx_+vy_*vy_);

    //x_ = x0_ + (uint16_t)(v0x_*time_step) + (uint16_t)(ax_*time_*time_/2);
    //y_ = y0_ + (uint16_t)(v0y_*time_step) + (uint16_t)(ay_*time_*time_/2);
    
    
    //Hydro/Areodynamic resistance calcuations
    Re = p*v_w*2*R_/n;
    // split resistance to x and y components and
    // calculate x and y opposing force vectors.
    if(Re<1){
        Fxw_ = mass_*ax_  -6*3.14*n*R_*vx_;
        Fyw_ = mass_*ay_  -6*3.14*n*R_*vy_;
    }else{
        //return;
        Fxw_ = mass_*ax_ - C_D_*S_D_*(p*abs(vx_)*abs(vx_))/2;
        Fyw_ = mass_*ay_ - C_D_*S_D_*(p*abs(vy_)*abs(vy_))/2;
    }

    // calculate movement restricted by opposing force
    // aw = Fw/m
    vx_ = v0x_ + Fxw_/mass_*time_;
    vy_ = v0y_ + Fyw_/mass_*time_;
    x_ = x0_ + (uint16_t)(v0x_*time_) + (uint16_t)(Fxw_/mass_*time_*time_/2);
    y_ = y0_ + (uint16_t)(v0y_*time_) + (uint16_t)(Fyw_/mass_*time_*time_/2);

    //time_ += time_s;
}

void Sand::setMovementParam(float velocity, float acceleration){
    ax_ = acceleration;
    v0x_ = velocity;

}

void Sand::getPosition(uint16_t &x, uint16_t &y){
    x = x_;
    y = x_;
}

float Sand::getVelocity(){
 return vx_;
}

float Sand::getAcceleration(){
 return ax_;
}

void Sand::printPosition(){
    std::cout <<"time stamp = "<< time_ << " x = " << x_ << " y = " << y_ ;
     std::cout <<" vx = " << vx_ <<" vy = " << vy_ <<std::endl;
}