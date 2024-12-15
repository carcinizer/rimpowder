#pragma once
#include <cstdint>

struct RGBA {
    uint8_t r, g, b, a;
    RGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a):r(r), g(g), b(b), a(a) {};
    operator uint32_t () { return (uint32_t)r << 3*8 | g << 2*8 | b << 1*8 | a;}
  };