#include "primitives/vec2.hpp"
#ifdef SDL_IMPL

#include <SDL3/SDL.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_init.h>
#include <cstddef>
#include <cstdint>
#include <disp/internal/SDL_window.hpp>
#include <iostream>
#include <memory>

namespace disp { namespace internal {

  window_impl::window_impl(vec2<int> resolution, const std::string& window_title)
      : resolution(resolution),
        window_title(window_title),
        window_surface(nullptr),
        window(nullptr) {}

  int window_impl::initialise() {
    // init everything ~0U hehe.
    if (!SDL_Init(SDL_INIT_VIDEO)) {
      std::cout << "SLD init error" << SDL_GetError() << std::endl;
      return 1;
    }

    if (!SDL_CreateWindowAndRenderer(
            window_title.c_str(), resolution.x, resolution.y, 0, &window, &renderer)) {
      std::cout << "SDL CreateWindowAndRenderer error" << SDL_GetError() << std::endl;
      return 2;
    }
    window_surface = SDL_GetWindowSurface(window);
    if (!window_surface) {
      std::cout << "SDL GetWindowSurface " << SDL_GetError() << std::endl;
      return 3;
    }
    return 0;
  }

  void window_impl::draw_buffer(
      int x_pos,
      int y_pos,
      int x_size,
      int y_size,
      void* buff,
      uint32_t stride,
      const vec2<unsigned> scale) {
    SDL_Texture* texture = SDL_CreateTexture(
        renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, x_size, y_size);
    if (!texture) {
      std::cout << "Failed to create texture: " << SDL_GetError() << std::endl;
      return;
    }

    void* pixels;
    int pitch;
    if (SDL_LockTexture(texture, NULL, &pixels, &pitch)) {
      if (sizeof(uint32_t) * x_size != pitch) {
        std::cout << "stride is not equal to pitch (byte size of pixel) " << std::endl;
        std::cout << "stride: " << stride << std::endl;
        std::cout << "pitch: " << pitch << std::endl;
        return;
      }
      memcpy(pixels, buff, x_size * y_size * sizeof(uint32_t));
      SDL_UnlockTexture(texture);
    } else {
      std::cout << "cannot lock texture" << SDL_GetError() << std::endl;
    }

    SDL_FRect tx_rect = {
        .x = 0, .y = 0, .w = (float)x_size * scale.x, .h = (float)y_size * scale.y};
    SDL_RenderTexture(renderer, texture, NULL, &tx_rect);
    SDL_DestroyTexture(texture);
  }

  void window_impl::draw_buffors() {
    SDL_RenderPresent(renderer);
  }

  void window_impl::set_color(const RGBA& col) {
    SDL_SetRenderDrawColor(renderer, col.r, col.g, col.b, col.a);
    SDL_RenderClear(renderer);
  }

  window_impl::~window_impl() {
    close_window();
    SDL_Quit();
  }
  void window_impl::close_window() {
    std::cout << "Closing window" << std::endl;
    if (window)
      SDL_DestroyWindow(window);
  }

  int window_impl::poll_events(std::queue<disp::event::event_ptr_t>& ev_queue) {
    using namespace disp::event;
    using event_inner_t = disp::event::event_t::event_inner_t;
#define make_shared_event std::make_shared<event_t>
    int ret = 0;

    SDL_Event event = {0};
    while (SDL_PollEvent(&event)) {
      switch (event.type) {
        case SDL_EVENT_KEY_DOWN: {
          keyboard_event ev;
          ev.key = event.key.key;
          ev.shift = (event.key.mod | SDL_KMOD_LSHIFT) || (event.key.mod | SDL_KMOD_RSHIFT);
          ev.alt = (event.key.mod | SDL_KMOD_ALT);
          ev.ctrl = (event.key.mod | SDL_KMOD_CTRL);
          ev_queue.push(make_shared_event(event_t{.type = KEYBOARD, .ev = {.keyboard_ev = ev}}));
        } break;
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED: {
          ret = 1;
          close_event ev;
          ev.should_close = true;
          ev_queue.push(make_shared_event(event_t{.type = CLOSE, .ev = {.close_ev = ev}}));
        } break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP: {
          mouse_event ev;
          ev.button = (mouse_event::mouse_button)event.button.button;
          ev.relative_position.x = event.button.x;
          ev.relative_position.y = event.button.y;
          ev.state = (mouse_event::mouse_state)event.button.down;
          ev_queue.push(make_shared_event(event_t{.type = MOUSE, .ev = {.mouse_ev = ev}}));
        } break;
      }
    }
    return ret;
  }

}; };  // namespace disp::internal

#endif
