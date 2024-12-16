cmake_minimum_required(VERSION 3.17)
project(rimpowder LANGUAGES C CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmakeScripts/printList.cmake)

file( GLOB_RECURSE
    RIMPOWDER_SOURCES
    CONFIGURE_DEPENDS src/*.cpp src/*.cu src/*.c
)

add_subdirectory(dummy_state_creator)
add_subdirectory(deps/SDL)
add_executable(rimpowder ${RIMPOWDER_SOURCES})

printPrettyList(RIMPOWDER_SOURCES "Sources in target:")

# I use cmake "cmake -B build -DMICHAL_INCLUDES=ON" command.
if(MICHAL_INCLUDES)
message("Mine cmake is on...")
    target_include_directories(rimpowder PRIVATE ../cuda-samples/common)
else()
    message("note there is CMAKE option here to include cuda-samples directory with")
    message("all includes and freeglut.lib.")
endif()

printPrettyList(SDL2_INCLUDE_DIRS "SDL2_INCLUDE")
printPrettyList(SDL2_INCLUDE_DIRS "SDL2_INCLUDE")
target_include_directories(rimpowder PUBLIC ../cuda-samples/Common)
target_include_directories(rimpowder PUBLIC include/ deps/stb)
target_include_directories(rimpowder PRIVATE deps/SDL/include)
#set the SDL window implementation to true ("rn its the only one existing")
target_compile_definitions(rimpowder PUBLIC SDL_IMPL=true)
target_link_libraries(rimpowder PRIVATE SDL3-shared)
# libm, required by stb_image
target_link_libraries(rimpowder PUBLIC m)

# Fix clangd LSP
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
