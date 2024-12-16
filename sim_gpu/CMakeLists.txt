project(sim_gpu LANGUAGES C CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(sim_gpu main.cu gpu_sim.cu cuda_sim.cu)

target_include_directories(sim_gpu PRIVATE
  ${PROJECT_SOURCE_DIR}/deps/stb
  ${PROJECT_SOURCE_DIR}/dummy_state_creator
)
target_include_directories(sim_gpu PUBLIC include ../../cuda-samples/Common)
target_link_libraries(sim_gpu PUBLIC m)

# Fix clangd LSP
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")