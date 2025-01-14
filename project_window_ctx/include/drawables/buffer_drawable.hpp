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
  bool free_buffer;
  vec2<unsigned> scale;

 public:
  /*!
   * \brief Constructor creating buffor for drawable context
   * of size x*y;
   */
  buffor_drawable(size_t x, size_t y, vec2<unsigned> scale = {1, 1})
      : x(x), y(y), free_buffer(true), scale(scale) {
    buff = new buff_T[x * y];
    memset(buff, 0, x * y * sizeof(buff_T));
  }

  /*!
   * \brief Constructor accepting existing buffor sized x*y
   * It will not destroy the buffor upon destruction.
   */
  buffor_drawable(
      size_t x,
      size_t y,
      buff_T* buff,
      bool free_on_destruction = false,
      vec2<unsigned> scale = {1, 1})
      : x(x), y(y), free_buffer(free_on_destruction), scale(scale) {
    buff = new buff_T[x * y];
    memset(buff, 0, x * y * sizeof(buff_T));
  }

  virtual ~buffor_drawable() {
    if (free_buffer) {
      delete[] buff;
    }
  }

  void draw(disp::internal::window_impl& wnd) const override {
    std::lock_guard<std::mutex> lock(mtx);
    // TODO: This should be kind of virtual idk?
    wnd.draw_buffer(0, 0, x, y, buff, sizeof(buff_T), scale);
  }

  std::mutex& get_mtx() { return mtx; }

  vec2<uint64_t> getAABB() { return {x, y}; }

  buff_T** get() { return &buff; }
};

template <typename T>
using buffor_drawable_ptr = std::shared_ptr<buffor_drawable<T>>;
