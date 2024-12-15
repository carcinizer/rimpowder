#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include "simulation.cuh"
#include "common.cuh"

#include <disp/window.hpp>
#include <iostream>
#include <drawables/buffer_drawable.hpp>
#include <memory>
#include <thread>
#define M_PI 3.14159265358979323846

__global__ void kernel() {

}

void draw_shit(buffor_drawable_ptr<uint32_t>& buff, size_t xSize, disp::Window& wnd) {
    std::mutex& mtx = buff->get_mtx();
    uint32_t* raw_buff = *buff->get();
    double R = 200;
    #define FLAT2D_IDX(cx, cy) cx + cy*xSize
    #define MAX_ITER 360
    for(int i = 0; i < MAX_ITER; i++) {
        wnd.update();
        if(wnd.should_close()) return;
        wnd.clear(0xFFU);
        {
            // dummy edit the buffor with buffor lock!
            std::lock_guard<std::mutex> lock(mtx);
            size_t x = R*(std::sin((double)2.0*M_PI*i/MAX_ITER) + 1);
            size_t y = R*(std::cos((double)2.0*M_PI*i/MAX_ITER) + 1);
            std::cout << " x= " << x << std::endl;
            std::cout << " y= " << y << std::endl;
            raw_buff[FLAT2D_IDX(x, y)] = 0xFF0000FF;
        }
        // remember that buffer_drawable needs an access to its lock in window draw!
        wnd.draw();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

}


int main(int argc, char** argv) {
    checkCudaErrors(cudaSetDevice(0));

    Simulation sim(arg_str);


    {
        size_t testx = 500, testy = 500;

        // IMPORTANT MESSAGE "HOW TO USE"
        // CUZ I DONT HAVE JUICE LEFT TO DOCUMENT THE CODE
        //
        // all you need to know it this thing bellow
        auto pix_art = std::make_shared<buffor_drawable<uint32_t>>(testx, testy);
        // that is the buffor you will draw into.
        //
        // you can use memcpu to draw into buff->get().
        // ex: cudamemcpy(buff->get(), from, sizeof(uint32_t)*testx*testy, cudaMemcpyHostToDevice);
        ////
        // this will draw every frame if you use
        // main_window.clear(0xFFU) // <- draw everything black
        // ... copy pixels to buffer -
        // ---- REMEMBER TO LOCK THE BUFFOR MUTEX IF YOU ARE MULTITHREADING!!
        //
        // main_window.draw();
        // ...
        //
        // You can use main_window.update() to poll all events from the window
        // (currently keyboard_button, mouse_button and close_window events are implemented).
        // Rn there is no functionality to check those events (GLHF :) )
        //
        // use main_window.should_close() to check if render window is closed.

        // MAIN WINDOW MUST RUN ALL FUNCTIONS ON MAIN THREAD!!
        disp::Window main_window("First window tests" , vec2<int>{1280, 720});
        main_window.add_drawable(pix_art);
        if(main_window.initialise()) {
            return -1;
        }

        draw_shit(pix_art, testx, main_window);
        int dummy;
        std::cin >> dummy;
        // remember window is RAII-style. it will close on destruction.
    }
    //kernel<<<1,1>>>();
    return 0;
}
