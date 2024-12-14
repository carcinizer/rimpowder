#include <cstdint>
#include <png_creator.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

uint32_t default_pix_functor(size_t x, size_t y, uint32_t currPx) {
  return currPx;
}



png_creator::png_creator(uint32_t xSize, uint32_t ySize):
  xSize(xSize), ySize(ySize) {
    pix_buffor = new uint32_t[xSize*ySize];
}

png_creator_status png_creator::process(pix_functor fn) {
  //#define FLAT_IDX(cx, cy) cy + cx*ySize
  #define FLAT_IDX(cx, cy) cx + cy*xSize
  for (int cx = 0; cx < xSize; cx++) {
    for (int cy = 0; cy < ySize; cy++) {
      pix_buffor[FLAT_IDX(cx, cy)] = fn(cx, cy, pix_buffor[FLAT_IDX(cx, cy)]);
    }
  }
}

png_creator_status png_creator::save(std::string file_name) {
  stbi_write_png(file_name.c_str(), xSize, ySize, 4, pix_buffor, xSize*sizeof(uint32_t));
}

png_creator::~png_creator() {
  if (pix_buffor != nullptr) {
    delete [] pix_buffor;
    pix_buffor = nullptr;
  }
}