#include "txt_sim.h"


int main(){
    Sand sandlet(0,0);
    sandlet.setMovementParam(0,10);
    sandlet.printPosition();
    sandlet.calculateNextLocation();

    sandlet.printPosition();
    sandlet.calculateNextLocation();

    sandlet.printPosition();
    sandlet.calculateNextLocation();
    sandlet.printPosition();

    return 0;
};