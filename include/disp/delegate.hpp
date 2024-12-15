#pragma once

//TODO: Actually not needed... can be removed

#include <disp/event.hpp>
#include <initializer_list>
#include <memory>
#include <stdexcept>
// actually including this right here does throw
// an error related to g++11 issue
// https://github.com/stotko/stdgpu/issues/337
//#include <functional>


// All comments below are related to mentioned issue
namespace disp {
namespace delegates {
  //template<typename Target, void(*actuator)(Target&, disp::event::event_ptr_t ev)>
  //using namespace std::placeholders;
  template<typename Target, typename fnType, fnType actuator>
  class event_wrapper {
    private:
      static event_wrapper* current_instance;
      //typedef void(*keyboard_fun_ptr)(unsigned char, int, int);
      Target& tg;
      //std::function<void(unsigned char, int, int)> keyboard_fun;
    public:
      event_wrapper(Target& tg): tg(tg)//,
      {
        if(event_wrapper::current_instance != nullptr)
          throw std::runtime_error("Current instance of event_wrapper is already set!");
        event_wrapper::current_instance = this;
      }

      //keyboard_fun_ptr get_keyboard_fun() { return nullptr; } //keyboard_fun.target<void(unsigned char, int, int)>(); }

      static void keyboard_ev_wrap(unsigned char key, int x, int y) {
        // int mod = glutGetModifiers();
        // bool shift = (mod & GLUT_ACTIVE_SHIFT);
        // bool alt = (mod & GLUT_ACTIVE_ALT);
        // bool ctrl = (mod & GLUT_ACTIVE_CTRL);
        //auto ptr = std::make_shared<disp::event::event_t>(disp::event::event_t{disp::event::KEYBOARD, disp::event::keyboard_event{key, shift, alt, ctrl}});
        //actuator(event_wrapper::current_instance->tg, ptr);
      }
  };

  template<typename Target, typename fnType, fnType actuator>
  event_wrapper<Target, fnType, actuator>* event_wrapper<Target, fnType, actuator>::current_instance = nullptr;


};
};