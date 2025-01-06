#pragma once

#include <string>
#include <cstdint>
#include <cuda_runtime.h>

constexpr size_t CHUNK_SIZE = 16;
constexpr float GRAVITY = 9.81; // in 


enum class ParticleType : uint8_t {
    VOID_ = 0,
    SAND = 1,
    WALL = 2
};


struct Particle {
    ParticleType type;
    float2 pos;
    float2 velocity;

    Particle() = default;
    Particle(uint32_t rgba, int2 pos_);
    uint32_t to_rgba() const;
};

struct Chunk {
    Particle contents[CHUNK_SIZE][CHUNK_SIZE];
};

struct Collision {
    int2 last_free; /// Last free spot found by the line drawing algorithm
    int2 collider; /// Same as last_free if no collision occured.

    __host__ __device__ Collision(int2 last_free_, int2 collider_) : last_free(last_free_), collider(collider_) {}
    __host__ __device__ bool collided() {
        return last_free.x == collider.x && last_free.y == collider.y;
    }
};


class Simulation {
private:
    Chunk* dev_chunks;
    int2 dims; /// Simulation area size, in chunks

public:
    Simulation(std::string& types_filename);
    
    void step(uint32_t time_ms); 
    void save(std::string& filename) const;

    ~Simulation();
};

Chunk& __host__ __device__ get_chunk(Chunk* chunks, int2 dims, int2 coord);
Particle& __host__ __device__ get_particle(Chunk* chunks, int2 dims, int2 coord);
Collision __device__ __host__ find_collision(Chunk* chunks, int2 dims, int2 from, int2 to);

static void __global__ simulation_kernel(Chunk* chunks, int2 dims, int time_ms);
Collision __device__ __host__ find_collision(Chunk* chunks, int2 dims, int2 from, int2 to);
