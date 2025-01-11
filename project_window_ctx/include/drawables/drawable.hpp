#pragma once
#include <memory>
#include <mutex>

namespace disp {
  namespace internal {
    class window_impl;
  };

  /*!
   * \brief Drawable interface. All that should draw onto the screen
   * should inherit and implement this interface.
   */
  class drawable {
   public:
    /*!
     * \brief Method drawing directly onto the window implementation (bad).
     *
     */
    virtual void draw(disp::internal::window_impl&) const = 0;
  };

  using drawable_ptr = std::shared_ptr<disp::drawable>;
};  // namespace disp
