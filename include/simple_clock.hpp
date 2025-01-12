#pragma once

#include <chrono>
#include <type_traits>

class simple_clock {
  template <typename _Tp>
  struct is_duration : std::false_type {};

  template <typename _Rep, typename _Period>
  struct is_duration<std::chrono::duration<_Rep, _Period>> : std::true_type {};

 public:
  simple_clock() : last(std::chrono::steady_clock::now()) {}

  template <typename time_quant>
  time_quant restart() {
    static_assert(is_duration<time_quant>::value, "template arg must be duration!");
    const auto before = last;
    last = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<time_quant>(last - before);
  }

  template <typename time_quant>
  time_quant get() const {
    static_assert(is_duration<time_quant>::value, "template arg must be duration!");
    return std::chrono::duration_cast<time_quant>(std::chrono::steady_clock::now() - last);
  }

 private:
  std::chrono::steady_clock::time_point last;
};
