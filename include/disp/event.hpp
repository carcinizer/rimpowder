#pragma once

#include "primitives/vec2.hpp"
#include <memory>
namespace disp {

namespace event {
  enum event_type {
    KEYBOARD,
    MOUSE,
    //RESIZE,
    CLOSE,
    NONE,
  };

  struct keyboard_event {
    unsigned char key;
    bool shift, alt, ctrl;
  };

  /**
  * \brief Mouse event structure.
  */
  struct mouse_event {
    enum mouse_state {
      down, up
    };
    //! What state is button, that generated the event
    mouse_state state;
    enum mouse_button {
      left, middle, right0
    };
    //! Determines what mouse button generated event.
    mouse_button button;
    //! Position relative to the window (upper-left corner?).
    vec2<int> relative_position;
  };

  struct close_event {
    bool should_close;
  };

  struct event_t {
    typedef union {
      keyboard_event keyboard_ev;
      mouse_event mouse_ev;
      close_event close_ev;
    } event_inner_t;

    //event_t(event_t&& ) = default;
    //event_t(const event_type& t, event_inner_t& ev): type(t), ev(ev) {};
    event_type type;
    event_inner_t ev;
  };
  using event_ptr_t = std::shared_ptr<event_t>;
}

};