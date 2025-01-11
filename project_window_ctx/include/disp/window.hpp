#pragma once

#include <cstdint>
#include <disp/delegate.hpp>
#include <disp/event.hpp>
#include <drawables/drawable.hpp>

//#include <functional>
#include <memory>
#include <primitives/vec2.hpp>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SDL_IMPL
#include <disp/internal/SDL_window.hpp>
#endif

namespace disp {

  enum update_ret { HANDLED, PASS, ERRORED };

  //! update function handler
  // typedef  std::function<update_ret(disp::event::event_t&)> update_handler_t;

  class Window;
  void actuate(disp::Window& wnd, disp::event::event_ptr_t ev);
  /**
   * \brief Window class that should be used to display stuff.
   */
  class Window {
   private:
    //! actually obsolette from glut impl?
    static Window* current_instance;

    disp::internal::window_impl underlying_window;

    void setup_callbacs();
    vec2<int> resolution;
    std::string window_title;
    std::vector<drawable_ptr> drawable_elements;

    std::mutex event_mutex;
    std::queue<disp::event::event_ptr_t> events;
    // update_handler_t update_handle;
    friend void actuate(disp::Window&, disp::event::event_ptr_t);
    bool window_closed;

   public:
    Window() = delete;
    ~Window();

    /*!
     * \brief Constructor that will create the window, alocate
     * the sources and open up window instance.
     *
     * \param window_title Title of the window that will show on the taskbar.
     * \param resolution efective resolution of the window.
     */
    Window(std::string window_title, vec2<int> resolution = {1280, 720});

    /*!
     * \brief Function initialising underlying window implementation.
     *
     * \return 0 on success otherwise positive number (check selected impl code)
     */
    int initialise();

    /*!
     * \brief registering drawable context into current window drawables pool.
     * \param ptr pointer to drawable interface.
     */
    void add_drawable(const drawable_ptr& /*ptr*/);

    /*!
     * \brief Function drawing stuff onto the visual buffer.
     * All stuff of accesing and alternating the window context should probably
     * be done on the same thread. It mostly relys on underlying window
     * implementation, but nearly all of them run on single thread.
     */
    void draw();

    /*!
     * \brief Function drawing one color into the whole buffer.
     */
    void clear(uint32_t color);

    /*!
     * \brief Function pooling window events. Should be called every iteration
     * to poll events.
     * Calls close_window when user requests the window to close.
     *
     * Accesing the events is not currently accessable for user, but is a simple
     * add in to implement.
     */
    void update();

    /*!
     * \brief Function returning if user requested to close the window.
     */
    bool should_close() const;

    /*!
     * \brief Function currently doing nothing, but ideally writing
     * self orginising scene setup may leed to using it.
     */
    void loop();
    /*!
     * \brief Method closing window.
     */
    void close_window();
  };
};  // namespace disp
