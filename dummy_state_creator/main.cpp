
#include <cstddef>
#include <png_creator.hpp>

#define LAMBDA_DECL [](size_t x, size_t y, uint32_t curr_pix) -> uint32_t

#define DRAW_UNDER \
  { return (x < y) ? 0xFFFF00FF : curr_pix; }

#define DRAW_LINE1 \
  { return (x == y) ? 0xFFFFFFFF : curr_pix; }

int main(int argc, char** argv) {
  png_creator creator_test(256, 256);
  creator_test.process([](size_t x, size_t y, uint32_t curr_pix) -> uint32_t {
    return (y == x && (x % 5 == 0)) ? 0xFFFF00FF : curr_pix;
  });

  creator_test.process([](size_t x, size_t y, uint32_t curr_pix) -> uint32_t {
    return ((x > 50 && x < 100) && (y > 20 && y < 70)) ? 0xFFFF00FF : curr_pix;
  });

  creator_test.process([](size_t x, size_t y, uint32_t curr_pix) -> uint32_t {
    return ((x > 80 && x < 130) && (y > 100 && y < 150)) ? 0xFFFF00FF : curr_pix;
  });

  // creator_test.process([](size_t x, size_t y, uint32_t curr_pix) -> uint32_t {
  //   return 0xFFFF00FF;
  // });
  // creator_test.process( LAMBDA_DECL DRAW_UNDER);
  // creator_test.process(LAMBDA_DECL DRAW_LINE1);
  creator_test.save("dummy_test_pix_drawer.png");
}
