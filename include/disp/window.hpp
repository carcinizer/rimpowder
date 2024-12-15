#pragma once

#include <disp/delegate.hpp>
#include <drawables/drawable.hpp>
#include <disp/event.hpp>
#include <cstdint>
//#include <functional>
#include <primitives/vec2.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <queue>
#include <memory>

#ifdef SDL_IMPL
  #include <disp/internal/SDL_window.hpp>
#endif

namespace disp {

  enum update_ret {
    HANDLED,
    PASS,
    ERRORED
  };

  //! update function handler
  //typedef  std::function<update_ret(disp::event::event_t&)> update_handler_t;

  class Window;
  void actuate(disp::Window& wnd, disp::event::event_ptr_t ev);
  /**
  * \brief Window class that should be used to display stuff.
  */
  class Window {
    private:
      static Window* current_instance ;

      disp::internal::window_impl underlying_window;

      void setup_callbacs();
      vec2<int> resolution;
      std::string window_title;
      std::vector<drawable_ptr> drawable_elements;

      std::mutex event_mutex;
      std::queue<disp::event::event_ptr_t> events;
      //update_handler_t update_handle;
      friend void actuate(disp::Window&, disp::event::event_ptr_t);
      bool window_closed;
    public:
      Window() = delete;
      ~Window();
      /**
      * \brief Constructore that will create the window, alocate
      * the sources and open up window instance.
      *
      * \param window_title Title of the window that will show on the taskbar.
      * \param resolution efective resolution of the window.
      */
      Window(std::string window_title, vec2<int> resolution= {1280, 720});
      int initialise();
      void add_drawable(const drawable_ptr&);
      void draw();
      void clear(uint32_t color);
      void update();
      bool should_close() const;
      void loop();
      void close_window();

  };
};
