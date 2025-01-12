
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
    return (y == -x + 250) ? 0xFFFF00FF : curr_pix;
  });
  // creator_test.process( LAMBDA_DECL DRAW_UNDER);
  creator_test.process(LAMBDA_DECL DRAW_LINE1);
  creator_test.save("dummy_test_pix_drawer.png");
}
