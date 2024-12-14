#pragma once
#include <cstddef>
#include <cstdint>
#include <string>


enum png_creator_status{
  CR_ERROR,
  CR_STATUS_OK
};



typedef uint32_t (*pix_functor)(size_t x, size_t y, uint32_t currPx);
uint32_t default_pix_functor(size_t x, size_t y, uint32_t currPx);

class png_creator {
  protected:
    uint32_t* pix_buffor;
    size_t xSize, ySize;
  public:
    png_creator(uint32_t xSize = 16, uint32_t ySize = 16);
    png_creator_status process(pix_functor = default_pix_functor);
    png_creator_status save(std::string file_name);
    virtual ~png_creator();
};