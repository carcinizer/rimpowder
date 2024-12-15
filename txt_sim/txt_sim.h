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
    uint16_t x_,y_, x0_, y0_;
    float time_;
   
    float Fxw_, v0x_, vx_, ax_; // values for x axis movement
    float Fyw_, v0y_, vy_, ay_; // values for y axis movement
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