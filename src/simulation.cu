#include <cuda_runtime_api.h>
#include <cassert>
#include <iostream>
#include "common.cuh"
#include "simulation.cuh"
#include "stb_image.h"
#include "stb_image_write.h"

Particle::Particle() : type{ParticleType::VOID_}, velocity{-1.f, -1.f}, pos{-1.f, -1.f} {}

Particle::Particle(uint32_t rgba, int2 pos_)
    : type{ParticleType::VOID_}, velocity{0.0f, 0.0f}, pos{pos_.x + 0.5f, pos_.y + 0.5f} {
  switch (rgba) {
    case 0xFFFF00FF:
      type = ParticleType::SAND;
      break;
    case 0xFFFFFFFF:
      type = ParticleType::WALL;
      break;
    case 0x00000000:
      type = ParticleType::VOID_;
      break;
    default:
      assert(true);
      break;
  }
}

uint32_t Particle::to_rgba() const {
  switch (type) {
    case ParticleType::SAND:
      return 0xFFFF00FF;
    case ParticleType::WALL:
      return 0xFFFFFFFF;
    default:
      return 0x00000000;
  }
}

Simulation::Simulation(std::string& types_filename)
    : dev_chunks(nullptr), cpu_chunks(nullptr), dims{0} {
  int xs = 0, ys = 0, channels = 0;
  unsigned char* char_buffer = stbi_load(types_filename.c_str(), &xs, &ys, &channels, 4);
  uint32_t* color_buffer = reinterpret_cast<uint32_t*>(char_buffer);
  if (color_buffer == nullptr) {
    std::cout << " could not load file: " << types_filename << std::endl;
    return;
  }
  // "ceil"
  int chunks_x = ((xs - 1) / CHUNK_SIZE) + 1;
  int chunks_y = ((ys - 1) / CHUNK_SIZE) + 1;

  dims = int2{chunks_x, chunks_y};

  cpu_chunks = new Chunk[chunks_x * chunks_y];

  for (int y = 0; y < ys; y++) {
    for (int x = 0; x < xs; x++) {
      uint32_t color = color_buffer[y * xs + x];
      int2 coord{x, y};
      Particle& dummy = get_particle(cpu_chunks, dims, coord);
      dummy = Particle{color, coord};
      // printf(" %u ", dummy.type);
    }
  }
  printf("\n");
  checkCudaErrors(cudaMalloc(&dev_chunks, sizeof(Chunk) * dims.x * dims.y));
  checkCudaErrors(
      cudaMemcpy(dev_chunks, cpu_chunks, sizeof(Chunk) * dims.x * dims.y, cudaMemcpyHostToDevice));
#ifdef DEBUG_DRAW_VISITED_PX
  checkCudaErrors(cudaMalloc(&dummy_buffor_visited_device, sizeof(uint32_t) * xs * ys));
  checkCudaErrors(cudaMemset(dummy_buffor_visited_device, 0, sizeof(uint32_t) * xs * ys));
#endif
  // delete[] cpu_chunks;
  delete[] char_buffer;
}

__device__ inline float clamp(float in, float min, float max) {
  return fminf(max, fmaxf(min, in));
}

#ifdef DEBUG_DRAW_VISITED_PX
__global__ void simulation_kernel(
    Chunk* chunks,
    int2 dims,
    int2 offset,
    int2 stride,
    float time_ms,
    uint32_t* debug_buff)
#else
__global__ void simulation_kernel(Chunk* chunks, int2 dims, int2 offset, int2 stride, float time_ms)
#endif
{
  uint2 chunk{threadIdx.x * stride.x + offset.x, threadIdx.y * stride.y + offset.y};
  // printf("thread{%u, %u} chunk{%u, %u}\n", threadIdx.x, threadIdx.y, chunk.x, chunk.y);
  float time = time_ms / 1000.0f;
  int2 max_sim_pos = {dims.x * CHUNK_SIZE, dims.y * CHUNK_SIZE};

  for (int y = 0; y < CHUNK_SIZE; y++) {
    for (int x = 0; x < CHUNK_SIZE; x++) {
      int2 pos = int2{int(chunk.x * CHUNK_SIZE + x), int(chunk.y * CHUNK_SIZE + y)};

#ifdef DEBUG_DRAW_VISITED_PX
      int chunk_x = pos.x / CHUNK_SIZE;
      int chunk_y = pos.y / CHUNK_SIZE;
      int x_within_chunk = pos.x % CHUNK_SIZE;
      int y_within_chunk = pos.y % CHUNK_SIZE;
      debug_buff
          [(chunk_y * dims.x * CHUNK_SIZE + chunk_x + y_within_chunk) * CHUNK_SIZE +
           x_within_chunk] = 0xFF0000FF;
#endif

      auto& particle = get_particle(chunks, dims, pos);
      // printf(" %u ", static_cast<uint8_t>(particle.type));

      if (particle.type != ParticleType::SAND) {
        continue;
      }

      // Calculating next position
      // printf("succesfull move vx = %.4f vy = %.4f\n", particle.velocity.x, particle.velocity.y);

      float2 to_f{
          particle.pos.x + time * particle.velocity.x, particle.pos.y + time * particle.velocity.y};
      to_f.x = clamp(to_f.x, 0, (float)max_sim_pos.x);
      to_f.y = clamp(to_f.y, 0, (float)max_sim_pos.y);

      int2 to{int(to_f.x), int(to_f.y)};

      auto collision = find_collision(chunks, dims, pos, to, max_sim_pos);

      if (collision.collided()) {
        // TODO lepsza aproksymacja? albo i niekoniecznie
        // TODO lepsze zderzenia cząsteczek
        particle.pos = float2{0.5f + collision.last_free.x, 0.5f + collision.last_free.y};
        particle.velocity = float2{particle.velocity.x * 0.2f, particle.velocity.y * 0.2f};
        // particle.velocity = float2{0, 0};
      } else {
        particle.pos = to_f;
        auto vx = particle.velocity.x /*+ time * GRAVITY*/;
        auto vy = particle.velocity.y + time * GRAVITY;
        // clamp so that no races occur between chunks
        // TODO także przekopiować opór jeśli trzeba
        particle.velocity.x = clamp(vx, -((float)CHUNK_SIZE) / 2, ((float)CHUNK_SIZE) / 2);
        particle.velocity.y = clamp(vy, -((float)CHUNK_SIZE) / 2, ((float)CHUNK_SIZE) / 2);
      }
      auto dummy2 = particle;  // copy particle to new variable
      // dummy2 = Particle();
      particle.type = ParticleType::VOID_;  // old position is now empty
      get_particle(chunks, dims, collision.last_free) = dummy2;
    }
  }
}

__device__ __host__ Collision
find_collision(Chunk* chunks, int2 dims, int2 from, int2 to, int2 max_constrains) {
  int2 last_free = from;
  int2 ptr = from;

  int dx = abs(from.x - to.x);
  int dy = abs(from.y - to.y);
  int sx = copysignf(1.0f, to.x - from.x);
  int sy = copysignf(1.0f, to.y - from.y);
  int error = dx - dy;

  if ((ptr.x == to.x && ptr.y == to.y)) {
    return Collision{last_free, ptr};
  }

  while (true) {
    int e2 = 2 * error;
    if (e2 > -dy) {
      error -= dy;
      ptr.x += sx;
    }
    if (e2 < dx) {
      error += dx;
      ptr.y += sy;
    }

    if ((ptr.x < 0 || ptr.x >= max_constrains.x) || (ptr.y < 0 || ptr.y >= max_constrains.y))
      break;
    if ((ptr.x == to.x && ptr.y == to.y)) {
      last_free = ptr;
      break;
    }

    if (get_particle(chunks, dims, ptr).type != ParticleType::VOID_) {
      return Collision{last_free, ptr};
    }

    last_free = ptr;
  }

  return Collision{last_free, ptr};
}

void Simulation::step(uint32_t time_ms) {
  int2 stride{2, 2};
  dim3 block_size{unsigned(dims.x / stride.x), unsigned(dims.y / stride.y), 1};

  for (const auto offset : {int2{0, 0}, int2{0, 1}, int2{1, 0}, int2{1, 1}}) {
#ifdef DEBUG_DRAW_VISITED_PX
    simulation_kernel<<<1, block_size>>>(
        dev_chunks, dims, offset, stride, time_ms, dummy_buffor_visited_device);
#else
    simulation_kernel<<<1, block_size>>>(dev_chunks, dims, offset, stride, time_ms);
#endif
    checkCudaErrors(cudaGetLastError());
  }
  cudaDeviceSynchronize();  // TODO upewnić się, czy to jest najlepsza metoda synchronizacji
                            // kolejnych kroków
  std::cout << std::endl;
}

void Simulation::save(std::string& filename) const {
  unsigned xs = dims.x * CHUNK_SIZE;
  unsigned ys = dims.y * CHUNK_SIZE;
  uint32_t* color_buffer = new uint32_t[xs * ys];

  Chunk* host_chunks = new Chunk[dims.x * dims.y];
  checkCudaErrors(
      cudaMemcpy(host_chunks, dev_chunks, sizeof(Chunk) * dims.x * dims.y, cudaMemcpyDeviceToHost));

  // this discards the const qualifier
  // make cpu_chunks mutable?
  // collect_from_gpu();

  for (int y = 0; y < ys; y++) {
    for (int x = 0; x < xs; x++) {
      int2 coord{x, y};
      uint32_t color = get_particle(host_chunks, dims, coord).to_rgba();
      color_buffer[y * xs + x] = color;
    }
  }

  stbi_write_png(filename.c_str(), xs, ys, 4, color_buffer, xs * sizeof(uint32_t));

  delete[] host_chunks;
  delete[] color_buffer;
}

void Simulation::collect_from_gpu() const {
  checkCudaErrors(
      cudaMemcpy(cpu_chunks, dev_chunks, sizeof(Chunk) * dims.x * dims.y, cudaMemcpyDeviceToHost));
}

void Simulation::synchronise_to_gpu() {
  checkCudaErrors(
      cudaMemcpy(dev_chunks, cpu_chunks, sizeof(Chunk) * dims.x * dims.y, cudaMemcpyHostToDevice));
}

void Simulation::put_pixel_data(uint32_t* buff) const {
  collect_from_gpu();
  int xs = dims.x * CHUNK_SIZE;
  int ys = dims.y * CHUNK_SIZE;

  for (int y = 0; y < ys; y++) {
    for (int x = 0; x < xs; x++) {
      int2 coord{x, y};
      uint32_t color = get_particle(cpu_chunks, dims, coord).to_rgba();
      buff[y * xs + x] = color;
    }
  }
}

#ifdef DEBUG_DRAW_VISITED_PX
void Simulation::put_visited_pixel_data(uint32_t* buff) const {
  ulong2 sizes = simulation_pixel_size();
  cudaMemcpy(
      buff, dummy_buffor_visited_device, sizes.x * sizes.y * sizeof(uint32_t),
      cudaMemcpyDeviceToHost);
}
#endif

ulong2 Simulation::simulation_pixel_size() const {
  return {(unsigned long)(dims.x * CHUNK_SIZE), (unsigned long)(dims.y * CHUNK_SIZE)};
}

Simulation::~Simulation() {
  checkCudaErrors(cudaFree(dev_chunks));
  if (cpu_chunks)
    delete[] cpu_chunks;
}

__host__ __device__ Chunk& get_chunk(Chunk* chunks, int2 dims, int2 coord) {
  int chunk_x = coord.x / CHUNK_SIZE;
  int chunk_y = coord.y / CHUNK_SIZE;

  return chunks[chunk_y * dims.x + chunk_x];
}

__host__ __device__ Particle& get_particle(Chunk* chunks, int2 dims, int2 coord) {
  int chunk_x = coord.x / CHUNK_SIZE;
  int chunk_y = coord.y / CHUNK_SIZE;
  int x_within_chunk = coord.x % CHUNK_SIZE;
  int y_within_chunk = coord.y % CHUNK_SIZE;

  return chunks[chunk_y * dims.x + chunk_x].contents[y_within_chunk][x_within_chunk];
}
