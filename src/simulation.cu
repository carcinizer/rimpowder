#include "stb_image.h"
#include "stb_image_write.h"
#include "common.cuh"
#include "simulation.cuh"


Particle::Particle(uint32_t rgba, int2 pos_) 
    : velocity{0.0f, 0.0f}, pos{pos_.x + 0.5f, pos_.y + 0.5f}
{
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

            get_particle(host_chunks, dims, coord) = Particle{color, coord};
        }
    }

    checkCudaErrors(cudaMalloc(&dev_chunks, sizeof(Chunk) * dims.x * dims.y));
    checkCudaErrors(cudaMemcpy(dev_chunks, host_chunks, sizeof(Chunk) * dims.x * dims.y, cudaMemcpyHostToDevice));

    delete[] host_chunks;
    delete[] char_buffer;
}


void __global__ simulation_kernel(Chunk* chunks, int2 dims, int2 offset, int2 stride, float time_ms) {
    uint2 chunk{threadIdx.x * stride.x + offset.x, threadIdx.y * stride.y + offset.y};
    float time = time_ms * 1000.0f;

    for(int y = 0; y < CHUNK_SIZE; y++) {
        for(int x = 0; x < CHUNK_SIZE; x++) {
            int2 pos = int2{int(chunk.x + x), int(chunk.y + y)};

            auto& particle = get_particle(chunks, dims, pos);

            if(particle.type != ParticleType::SAND) {
                break;
            }
            
            // Calculating next position
            float2 to_f{
                particle.pos.x + time * particle.velocity.x,
                particle.pos.y + time * particle.velocity.y
            };
            int2 to{int(to_f.x), int(to_f.y)};

            auto collision = find_collision(chunks, dims, pos, to);

            if(collision.collided()) {
                particle.pos = float2{pos.x + 0.5f, pos.y + 0.5f}; // TODO lepsza aproksymacja? albo i niekoniecznie
                particle.velocity = float2{0,0};// TODO lepsze zderzenia cząsteczek
            }
            else {
                particle.pos = to_f;
                auto vx = particle.velocity.x + time * GRAVITY;
                auto vy = particle.velocity.y + time * GRAVITY;
                // clamp so that no races occur between chunks
                // TODO także przekopiować opór jeśli trzeba
                particle.velocity.x = fminf(CHUNK_SIZE, fmaxf(float(-CHUNK_SIZE), vx));
                particle.velocity.y = fminf(CHUNK_SIZE, fmaxf(float(-CHUNK_SIZE), vy));
            }

            get_particle(chunks, dims, to) = particle; // copy particle to new position
            particle.type = ParticleType::VOID_; // old position is now empty
        }
    }
}

Collision __device__ __host__ find_collision(Chunk* chunks, int2 dims, int2 from, int2 to) {
    int2 last_free = from;
    int2 ptr = from;

    int dx = abs(from.x - to.x);
    int dy = abs(from.y - to.y);
    int sx = copysignf(1.0f, to.x - from.x);
    int sy = copysignf(1.0f, to.y - from.y);
    int error = dx + dy;

    while(true) {
        int e2 = 2*error;
        if(e2 >= dy) {
            error += dy;
            ptr.x += sx;
        }
        if(e2 <= dx) {
            error += dx;
            ptr.y += sy;
        }

        if((ptr.x == to.x && ptr.y == to.y) ) {
            break;
        }

        if(get_particle(chunks, dims, ptr).type != ParticleType::VOID_) {
            return Collision{last_free, ptr};
        }

        last_free = ptr;
    }

    return Collision{last_free, ptr};
}

/**
 * \brief Function calculating simulation steps
 *
 * This function calculates every step of simulation.
 */
void Simulation::step(uint32_t time_ms) {
    int2 stride{2,2};
    dim3 block_size{unsigned(dims.x/stride.x), unsigned(dims.y/stride.y), 1};

    for(const auto offset : { int2{0,0}, int2{0,1}, int2{1,0}, int2{1,1} }) {
        simulation_kernel<<<1, block_size>>>(dev_chunks, dims, offset, stride, time_ms);
        cudaDeviceSynchronize(); // TODO upewnić się, czy to jest najlepsza metoda synchronizacji kolejnych kroków
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

    return chunks[chunk_y*dims.x + chunk_x].contents[y_within_chunk][x_within_chunk];
}
