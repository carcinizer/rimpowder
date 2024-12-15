#pragma once
#include <mutex>
#include <memory>

namespace disp {
  namespace internal {
    class window_impl;
  };
  class drawable {
    public:
      virtual void draw(disp::internal::window_impl& ) const = 0;
  };
  using drawable_ptr = std::shared_ptr<disp::drawable>;
};