cmake_minimum_required(VERSION 3.17)
project(rimpowder LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(dummy_state_creator)

#include(cmakeScripts/setupNVCC.cmake)
add_executable(rimpowder
    src/main.cu
    src/deps_impl.cu
    src/simulation.cu
)

#target_compile_options(rimpowder PUBLIC cxx_std_20)
target_include_directories(rimpowder PUBLIC include deps/stb)
if(MICHAL_INCLUDES)
    message("Mine cmake is on...")
    target_include_directories(rimpowder PRIVATE ../cuda-samples/common)
endif()

target_link_libraries(rimpowder PUBLIC m) # libm, required by stb_image

# Fix clangd LSP
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
