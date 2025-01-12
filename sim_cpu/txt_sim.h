#include <cstdint>
#include <iostream>
#include <string>
#include <math.h>
#include <cmath>

class sim_element{
    private:

    public:
    virtual void calculateNextLocation(float time_step, float p, float n) = 0;
    virtual void getPosition(uint16_t &x, uint16_t &y) = 0;
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
    void calculateNextLocation(float time_step, float p, float n);
    void getPosition(uint16_t &x, uint16_t &y);
    void setMovementParam(float velocity, float acceleration);
    float getVelocity();
    float getAcceleration();
    void printPosition();
    ~Sand(){};

};