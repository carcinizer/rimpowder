#include "txt_sim.h"
#include <cstdint>
#include <iostream>


Sand::Sand(uint16_t x,uint16_t y):x0_(x),y0_(y){
x_ = 0;
y_ = 0;
v_ = 0;
v0_ = 0;
a_ = 0;
time_ = 0;

}
//case appropriate for gravity
void Sand::calculateNextLocation(){
    float g = 9.81;
    float time_s = 1;
    float v = v0_+a_*time_s;
    time_ += time_s;
    x_ = x0_ + (uint16_t)(v0_*time_s) + (uint16_t)(a_*time_*time_/2);
    v_ += v0_+a_*time_s;
    //time_ += time_s;
}

void Sand::setMovementParam(float velocity, float acceleration){
    a_ = acceleration;
    v0_ = velocity;

}

void Sand::getPosition(uint16_t &x, uint16_t &y){
    x = x_;
    y = x_;
}

float Sand::getVelocity(){
 return v_;
}

float Sand::getAcceleration(){
 return a_;
}

void Sand::printPosition(){
    std::cout <<"time stamp = "<< time_ << " x = " << x_ << " y = " << y_ <<" v = " << v_ <<std::endl;
}