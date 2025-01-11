file( GLOB_RECURSE
    GRAPHIC_CONTEXT_SOURCES
    CONFIGURE_DEPENDS src/*.cpp src/*.c
)

add_library(rimpowder_graphic STATIC ${GRAPHIC_CONTEXT_SOURCES})
target_include_directories(rimpowder_graphic PUBLIC include)
target_compile_definitions(rimpowder_graphic PUBLIC SDL_IMPL=true)
target_link_libraries(rimpowder_graphic PUBLIC SDL3-shared)
