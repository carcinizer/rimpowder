#pragma once

#include <cstddef>
#include <disp/window.hpp>
#include <drawables/drawable.hpp>
#include <iostream>
#include <memory>
#include <mutex>
#include "primitives/vec2.hpp"

/**
 * \brief Lightweight buffor drawable
 * it is ment to only hold the buffor,
 * be modified by actuator, and consumed by
 * window context.
 */
template <typename buff_T>
class buffor_drawable : public disp::drawable {
 private:
  buff_T* buff;
  size_t x, y;
  mutable std::mutex mtx;

 public:
  buffor_drawable(size_t x, size_t y) : x(x), y(y) {
    buff = new buff_T[x * y];
    memset(buff, 0, x * y * sizeof(buff_T));
  }

  virtual ~buffor_drawable() { delete[] buff; }

  void draw(disp::internal::window_impl& wnd) const override {
    std::lock_guard<std::mutex> lock(mtx);
    // TODO: This should be kind of virtual idk?
    wnd.draw_buffer(0, 0, x, y, buff, sizeof(buff_T));
  }

  std::mutex& get_mtx() { return mtx; }

  vec2<uint64_t> getAABB() { return {x, y}; }

  buff_T** get() { return &buff; }
};

template <typename T>
using buffor_drawable_ptr = std::shared_ptr<buffor_drawable<T>>;