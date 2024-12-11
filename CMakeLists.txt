cmake_minimum_required(VERSION 3.10)
project(rimpowder LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(rimpowder 
    src/main.cu
    src/deps_impl.cu
    src/simulation.cu
)

#target_compile_options(rimpowder PUBLIC cxx_std_20)
target_include_directories(rimpowder PUBLIC include deps/stb)
target_link_libraries(rimpowder PUBLIC m) # libm, required by stb_image

# Fix clangd LSP
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
