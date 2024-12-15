#include "stb_image.h"
#include "stb_image_write.h"
#include "common.cuh"
#include "simulation.cuh"


Particle::Particle(uint32_t rgba) {
    switch(rgba) {
        case 0xFFFF00FF:
            type = ParticleType::SAND;
            break;
        case 0xFFFFFFFF:
            type = ParticleType::WALL;
            break;
        default:
            type = ParticleType::VOID_;
            break;
    }
}

uint32_t Particle::to_rgba() const {
    switch(type) {
        case ParticleType::SAND:
            return 0xFFFF00FF;
        case ParticleType::WALL:
            return 0xFFFFFFFF;
        default:
            return 0x00000000;
    }
}


/**
 * \brief Simulation constructor
 *
 *  This function loads simultaion setup from file named in " types_filename".
 *  it then takes all pixel values and convert them into simulation types.
 *  the simulation state is saved into the x_pixels by y_pixels buffer on
 *  the device.
 *
 * \param types_filename simulation file path to load.
 */

Simulation::Simulation(std::string& types_filename) {
    int xs, ys, channels;
    unsigned char* char_buffer = stbi_load(types_filename.c_str(), &xs, &ys, &channels, 4);
    uint32_t* color_buffer = reinterpret_cast<uint32_t*>(char_buffer);

    // "ceil"
    int chunks_x = ( (xs-1)/CHUNK_SIZE ) + 1;
    int chunks_y = ( (ys-1)/CHUNK_SIZE ) + 1;

    dims = int2{chunks_x, chunks_y};

    Chunk* host_chunks = new Chunk[chunks_x * chunks_y];

    for(int y = 0; y < ys; y++) {
        for(int x = 0; x < xs; x++) {
            uint32_t color = color_buffer[y*xs+x];
            int2 coord{x,y};

            get_particle(host_chunks, dims, coord) = Particle(color);
        }
    }

    checkCudaErrors(cudaMalloc(&dev_chunks, sizeof(Chunk) * dims.x * dims.y));
    checkCudaErrors(cudaMemcpy(dev_chunks, host_chunks, sizeof(Chunk) * dims.x * dims.y, cudaMemcpyHostToDevice));

    delete[] host_chunks;
    delete[] char_buffer;
}



/**
 * \brief Function calculating simulation steps
 *
 * This function calculates every step of simulation.
 * 
 * 
 *
 * 
 */

void Simulation::step(uint32_t time_ms){
    unsigned xs = dims.x * CHUNK_SIZE;
    unsigned ys = dims.y * CHUNK_SIZE;
    uint32_t* color_buffer = new uint32_t[xs*ys];

    Chunk* host_chunks = new Chunk[dims.x * dims.y];
    checkCudaErrors(cudaMemcpy(host_chunks, dev_chunks, sizeof(Chunk) * dims.x * dims.y, cudaMemcpyDeviceToHost));

    for(int y = 0; y < ys; y++) {
        for(int x = 0; x < xs; x++) {
            int2 coord{x,y};
            uint32_t color = get_particle(host_chunks, dims, coord).to_rgba();
            color_buffer[y*xs+x] = color;
        }
    }
}

/**
 * \brief Function saving current state of the simulation
 *
 * This function dumps the simulation state into given file.
 * it does convert chung data into pixel values defined in
 * simulation.h
 *
 * \param filename simulation file path to save simulation state.
 */
void Simulation::save(std::string& filename) const {
    unsigned xs = dims.x * CHUNK_SIZE;
    unsigned ys = dims.y * CHUNK_SIZE;
    uint32_t* color_buffer = new uint32_t[xs*ys];

    Chunk* host_chunks = new Chunk[dims.x * dims.y];
    checkCudaErrors(cudaMemcpy(host_chunks, dev_chunks, sizeof(Chunk) * dims.x * dims.y, cudaMemcpyDeviceToHost));

    for(int y = 0; y < ys; y++) {
        for(int x = 0; x < xs; x++) {
            int2 coord{x,y};
            uint32_t color = get_particle(host_chunks, dims, coord).to_rgba();
            color_buffer[y*xs+x] = color;
        }
    }

    stbi_write_png(filename.c_str(), xs, ys, 4, color_buffer, xs*sizeof(uint32_t));

    delete[] host_chunks;
    delete[] color_buffer;
}

Simulation::~Simulation() {
    checkCudaErrors(cudaFree(dev_chunks));
}

Chunk& __host__ __device__ get_chunk(Chunk* chunks, int2 dims, int2 coord) {
    int chunk_x = coord.x/CHUNK_SIZE;
    int chunk_y = coord.y/CHUNK_SIZE;

    return chunks[chunk_y*dims.x + chunk_x];
}

Particle& __host__ __device__ get_particle(Chunk* chunks, int2 dims, int2 coord) {
    int chunk_x = coord.x/CHUNK_SIZE;
    int chunk_y = coord.y/CHUNK_SIZE;
    int x_within_chunk = coord.x%CHUNK_SIZE;
    int y_within_chunk = coord.y%CHUNK_SIZE;

    return chunks[chunk_y*dims.x + chunk_x].contents[chunk_y][chunk_x];
}
