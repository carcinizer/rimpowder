#include <disp/color.hpp>
#include <disp/delegate.hpp>
#include <disp/event.hpp>
#include <disp/window.hpp>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace disp {

  Window* Window::current_instance = nullptr;

  int Window::initialise() {
    // sanity checks
    if (current_instance != nullptr)
      throw std::runtime_error("Window current instance is already set!");

    current_instance = this;

    int ret = 0;
    if ((ret = underlying_window.initialise()) != 0) {
      std::cout << " Underlying errors detected" << std::endl;
      return ret;
    }
    setup_callbacs();

    return 0;
  }

  void actuate(disp::Window& wnd, disp::event::event_ptr_t ev) {
    // TODO: FIX/ not needed (glut remainings)?
    std::cout << "actually got event: " << std::endl;
    std::cout << "\tkey: " << ev->ev.keyboard_ev.key << std::endl;
    std::cout << "\tShift: " << ev->ev.keyboard_ev.shift << std::endl;
    std::cout << "\tAlt: " << ev->ev.keyboard_ev.alt << std::endl;
    std::cout << "\tctrl: " << ev->ev.keyboard_ev.ctrl << std::endl;
    std::lock_guard<std::mutex> lock(wnd.event_mutex);
    wnd.events.push(ev);
  }

  void Window::setup_callbacs() {}

  Window::Window(std::string window_title, vec2<int> resolution)
      : window_title(window_title),
        resolution(resolution),
        underlying_window(resolution, window_title),
        window_closed(false) {}

  void Window::add_drawable(const drawable_ptr& ptr) {
    drawable_elements.push_back(ptr);
  }

  void Window::draw() {
    if (Window::current_instance == nullptr) {
      return;
    }

    for (const auto& element : drawable_elements) {
      element->draw(underlying_window);
    }
    underlying_window.draw_buffors();
  }

  void Window::clear(uint32_t color) {
    underlying_window.set_color(RGBA(0, 0, 0, 255));
  }

  void Window::update() {
    std::lock_guard<std::mutex> lock(event_mutex);
    if (underlying_window.poll_events(events)) {
      close_window();
      window_closed = true;
    }
  }

  void Window::loop() {
    // TODO: hmmm?
  }

  bool Window::should_close() const {
    return window_closed;
  }

  void Window::close_window() {
    underlying_window.close_window();
  }

  Window::~Window() {
    Window::current_instance = nullptr;
    // TODO: deinit window
  }

};  // namespace disp