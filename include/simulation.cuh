#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
//#define DEBUG_DRAW_VISITED_PX

constexpr size_t CHUNK_SIZE = 16;
constexpr float GRAVITY = 0.981;  // in

enum class ParticleType : uint8_t { VOID_ = 0, SAND = 1, WALL = 2 };

struct Particle {
  ParticleType type;
  float2 pos;
  float2 velocity;

  Particle();
  Particle(uint32_t rgba, int2 pos_);
  Particle(const Particle&) = default;
  uint32_t to_rgba() const;
};

struct Chunk {
  Particle contents[CHUNK_SIZE][CHUNK_SIZE];
};

struct Collision {
  int2 last_free;  /// Last free spot found by the line drawing algorithm
  int2 collider;  /// Same as last_free if no collision occured.

  __host__ __device__ Collision(int2 last_free_, int2 collider_)
      : last_free(last_free_), collider(collider_) {}
  __host__ __device__ bool collided() {
    return !(last_free.x == collider.x && last_free.y == collider.y);
  }
};

class Simulation {
 public:
  /**
   * \brief Simulation constructor. This method loads simultaion setup from file named in "
   * types_filename". It then takes all pixel values and convert them into simulation types. The
   * simulation state is saved into the x_pixels by y_pixels buffer on the device.
   *
   * \param types_filename simulation file path to load.
   */
  Simulation(std::string& types_filename);

  /**
   * \brief Function calculating simulation steps. This function calculates every step of
   * simulation.
   */
  void step(uint32_t time_ms);

  /**
   * \brief Function saving current state of the simulation. This function dumps the simulation
   * state into given file. it does convert chung data into pixel values defined in simulation.h
   * \param filename simulation file path to save simulation state.
   */
  void save(std::string& filename) const;

  /**
   * \brief Method collecting data from gpu to cpu buffor.
   */
  void collect_from_gpu() const;

  /**
   * \brief Method synchronising data from cpu to gpu buffor.
   * Remember to synchronise gpu threads from simulation, and collecting data before that operation.
   */
  void synchronise_to_gpu();

  /**
   * \brief Method putting pixel data into given uint32_t buffor. User have to be sure, that:
   *    - bufor fits the data withing the simulation,
   *    - cuda threads are synchronised before data collection.
   */
  void put_pixel_data(uint32_t* buff) const;

  ulong2 simulation_pixel_size() const;

  ~Simulation();

#ifdef DEBUG_DRAW_VISITED_PX
  void put_visited_pixel_data(uint32_t* buff) const;
#endif

 private:
  Chunk* dev_chunks;
  mutable Chunk* cpu_chunks;

#ifdef DEBUG_DRAW_VISITED_PX
  mutable uint32_t* dummy_buffor_visited_device;
#endif
  int2 dims;  /// Simulation area size, in chunks
};

__host__ __device__ Chunk& get_chunk(Chunk* chunks, int2 dims, int2 coord);
__host__ __device__ Particle& get_particle(Chunk* chunks, int2 dims, int2 coord);
__device__ __host__ Collision find_collision(Chunk* chunks, int2 dims, int2 from, int2 to);

__global__ static void simulation_kernel(Chunk* chunks, int2 dims, int time_ms);
__device__ __host__ Collision find_collision(Chunk* chunks, int2 dims, int2 from, int2 to);
