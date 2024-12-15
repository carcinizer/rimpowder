#include <cstdint>
#include <iostream>
#include <string>
#include <math.h>


class sim_element{
    private:

    public:
    virtual void calculateNextLocation() = 0;
    virtual void getPosition(uint16_t &x, uint16_t &y) = 0;
    virtual void setMovementParam(float velocity, float acceleration) = 0;
    virtual float getVelocity() = 0;
    virtual float getAcceleration() = 0;
};


class Sand: public sim_element{
    private:
    uint16_t x_,y_, x0_, y0_;
    float v0_, v_, a_, time_;
    std::string type_;
    public:
    Sand(uint16_t x,uint16_t y);
    void calculateNextLocation();
    void getPosition(uint16_t &x, uint16_t &y);
    void setMovementParam(float velocity, float acceleration);
    float getVelocity();
    float getAcceleration();
    void printPosition();
    ~Sand(){};

};