#pragma once

#include <string>
#include <cstdint>

constexpr size_t CHUNK_SIZE = 16;


enum class ParticleType : uint8_t {
    VOID = 0,
    SAND = 1,
    WALL = 2
};


struct Particle {
    ParticleType type;

    Particle() = default;
    Particle(uint32_t rgba);
    uint32_t to_rgba() const;
};

struct Chunk {
    Particle contents[CHUNK_SIZE][CHUNK_SIZE];
};


class Simulation {
private:
    Chunk* dev_chunks;
    int2 dims;

public:
    Simulation(std::string& types_filename);
    
    void step(uint32_t time_ms); 
    void save(std::string& filename) const;

    ~Simulation();
};

Chunk& __host__ __device__ get_chunk(Chunk* chunks, int2 dims, int2 coord);
Particle& __host__ __device__ get_particle(Chunk* chunks, int2 dims, int2 coord);

static void __global__ simulation_kernel(Chunk* chunks, unsigned width, unsigned height);
