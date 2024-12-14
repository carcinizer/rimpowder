
add_executable(png_creator main.cpp png_creator.cpp)

target_include_directories(png_creator PRIVATE
  ${PROJECT_SOURCE_DIR}/deps/stb
  ${PROJECT_SOURCE_DIR}/dummy_state_creator
)

target_link_libraries(png_creator PUBLIC m)