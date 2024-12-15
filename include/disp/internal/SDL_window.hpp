#pragma once
#ifdef SDL_IMPL

#include <disp/event.hpp>
#include <cstddef>
#include <primitives/vec2.hpp>
#include <string>
#include <disp/color.hpp>
#include <queue>

#include <SDL3/SDL.h>

namespace disp{
  namespace internal {

    class window_impl {
        vec2<int> resolution;
        std::string window_title;
      // SDL related stuff
        SDL_Surface* window_surface;
        SDL_Window* window;
        SDL_Renderer* renderer;
      public:
        window_impl(vec2<int> resolution, const std::string& window_title);
        ~window_impl();
        int initialise();
        disp::event::event_t poll_event();

      // Specific draw functions
      public:
        void draw_buffer(int x_pos, int y_pos, int x_size, int y_size, void* buff, uint32_t stride);
        void draw_buffors();
        void set_color(const RGBA&);
        int poll_events(std::queue<disp::event::event_ptr_t>& ev_queue);
        void close_window();

    };
  };
};

#endif