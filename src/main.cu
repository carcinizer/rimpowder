#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <simple_clock.hpp>
#include "common.cuh"
#include "simulation.cuh"
#include "stb_image.h"
#include "stb_image_write.h"

#include <disp/window.hpp>
#include <drawables/buffer_drawable.hpp>
#include <iostream>
#include <memory>
#include <thread>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Wrong number of arguments\n" << std::endl;
    return 0;
  }
  std::string sim_output = "sim_output.png";
  std::string arg_str = argv[1];

  std::cout << "Starting simulation\n" << std::endl;

  checkCudaErrors(cudaSetDevice(0));

  Simulation sim(arg_str);

  sim.save(sim_output);
  auto sim_pixel_dims = sim.simulation_pixel_size();
  auto pix_art = std::make_shared<buffor_drawable<uint32_t>>(
      sim_pixel_dims.x, sim_pixel_dims.y, vec2<unsigned>{2u, 2u});
  disp::Window main_window("First window tests", vec2<int>{1280, 720});
  main_window.add_drawable(pix_art);

#ifdef DEBUG_DRAW_VISITED_PX
  auto debug_px = std::make_shared<buffor_drawable<uint32_t>>(sim_pixel_dims.x, sim_pixel_dims.y);
  main_window.add_drawable(debug_px);
#endif

  if (main_window.initialise()) {
    return -1;
  }

  simple_clock sim_clock;
  using std::chrono::milliseconds;

  for (int iter = 0; iter < 10000; iter++) {
    main_window.update();
    if (main_window.should_close()) {
      std::cout << "window should close" << std::endl;
      return -1;
    }
    main_window.clear(0xFFU);
    long long dt = sim_clock.restart<milliseconds>().count();
    // SIMULATION RELATED STUFF
    sim.step(dt);
    {
      std::lock_guard<std::mutex> lock(pix_art->get_mtx());
      sim.put_pixel_data(*pix_art->get());
    }
#ifdef DEBUG_DRAW_VISITED_PX
    {
      std::lock_guard<std::mutex> lock(debug_px->get_mtx());
      sim.put_visited_pixel_data(*debug_px->get());
    }
#endif
    // SIMULATION RELATED STUFF END
    main_window.draw();
    std::cout << "dt: " << dt << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }
  std::cout << "donna" << std::endl;

  while (!main_window.should_close()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    main_window.update();
    std::this_thread::yield();
  }

  // kernel<<<1,1>>>();
  return 0;
}
