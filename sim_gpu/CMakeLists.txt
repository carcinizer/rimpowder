project(sim_gpu LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(sim_gpu main.cpp gpu_sim.cu cuda_sim.cu)

target_include_directories(sim_gpu PRIVATE
  ../sim_gpu
  ../deps/stb
  ../dummy_state_creator
  ../include
)

target_include_directories(sim_gpu PUBLIC ../../cuda-samples/Common)
target_compile_definitions(sim_gpu PUBLIC SDL_IMPL=true)
target_link_libraries(sim_gpu PUBLIC m rimpowder_graphic)
# target_sources(sim_gpu PRIVATE
#   ../src/disp/window.cpp
#   ../src/disp/internal/SDL_window.cpp
#   ../src/drawable/buffer_drawable.cpp
# )

# Fix clangd LSP
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")