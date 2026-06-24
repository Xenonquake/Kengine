cmake_minimum_required(VERSION 3.25)
project(Kengine VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerFlags)
include(Dependencies)

find_package(Vulkan REQUIRED)

# ---------------------------------------------------------------------------
# C23 modular libraries (math, spherical harmonics, physics)
# ---------------------------------------------------------------------------
add_library(kengine_c_math STATIC
    c_modules/math/vec4.c
    c_modules/math/mat4.c
    c_modules/math/mat4d.c
    c_modules/math/gauss_legendre.c
)
target_include_directories(kengine_c_math PUBLIC c_modules/math)
kengine_apply_c23_flags(kengine_c_math)

add_library(kengine_c_sh STATIC
    c_modules/sh/sh_basis.c
    c_modules/sh/sh_project.c
    c_modules/sh/sh_eval.c
)
target_include_directories(kengine_c_sh PUBLIC c_modules/sh c_modules/math)
target_link_libraries(kengine_c_sh PUBLIC kengine_c_math)
kengine_apply_c23_flags(kengine_c_sh)

add_library(kengine_c_physics STATIC
    c_modules/physics/verlet_solver.c
    c_modules/physics/spatial_hash.c
)
target_include_directories(kengine_c_physics PUBLIC c_modules/physics c_modules/math)
target_link_libraries(kengine_c_physics PUBLIC kengine_c_math)
kengine_apply_c23_flags(kengine_c_physics)

# ---------------------------------------------------------------------------
# Shaders (SPIR-V)
# ---------------------------------------------------------------------------
set(KENGINE_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(KENGINE_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")

file(GLOB_RECURSE KENGINE_SHADER_SOURCES
    CONFIGURE_DEPENDS
    "${KENGINE_SHADER_DIR}/*.comp"
    "${KENGINE_SHADER_DIR}/*.frag"
    "${KENGINE_SHADER_DIR}/*.vert"
)

set(KENGINE_COMPILED_SHADERS "")
foreach(SHADER ${KENGINE_SHADER_SOURCES})
    file(RELATIVE_PATH REL_PATH "${KENGINE_SHADER_DIR}" "${SHADER}")
    set(SPV "${KENGINE_SHADER_OUTPUT_DIR}/${REL_PATH}.spv")
    get_filename_component(SPV_DIR "${SPV}" DIRECTORY)
    add_custom_command(
        OUTPUT "${SPV}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPV_DIR}"
        COMMAND glslc "${SHADER}" -o "${SPV}"
        DEPENDS "${SHADER}"
        COMMENT "Compiling shader ${REL_PATH}"
    )
    list(APPEND KENGINE_COMPILED_SHADERS "${SPV}")
endforeach()

add_custom_target(kengine_shaders ALL DEPENDS ${KENGINE_COMPILED_SHADERS})

# ---------------------------------------------------------------------------
# C++ engine library
# ---------------------------------------------------------------------------
add_library(kengine STATIC
    src/core/job_system.cpp
    src/core/service_locator.cpp
    src/ecs/registry.cpp
    src/math/vec4d.cpp
    src/physics/physics_world.cpp
    src/world/world.cpp
    src/vulkan/context.cpp
    src/vulkan/swapchain.cpp
    src/vulkan/vma_allocator.cpp
    src/vulkan/shader_module.cpp
    src/vulkan/pipeline_cache.cpp
    src/vulkan/pipeline_builder.cpp
    src/vulkan/dynamic_renderer.cpp
    src/vulkan/gpu_image.cpp
    src/render/scene_targets.cpp
    src/render/retro_pipeline.cpp
    src/render/pipeline_manager.cpp
    src/render/frame_renderer.cpp
    src/render/camera_4d.cpp
    src/render/render_graph.cpp
    src/render/frame_graph.cpp
    src/render/post_process.cpp
    src/render/texture.cpp
    src/lighting/sh_lighting.cpp
    src/app/engine.cpp
)
target_include_directories(kengine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/c_modules/math
    ${CMAKE_CURRENT_SOURCE_DIR}/c_modules/sh
    ${CMAKE_CURRENT_SOURCE_DIR}/c_modules/physics
    ${vma_SOURCE_DIR}/include
    ${stb_SOURCE_DIR}
)
target_link_libraries(kengine PUBLIC
    Vulkan::Vulkan
    glfw
    kengine_c_math
    kengine_c_sh
    kengine_c_physics
)
kengine_apply_compiler_flags(kengine)

target_compile_definitions(kengine PUBLIC
    VK_HPP_DISPATCH_LOADER_DYNAMIC=1
    VK_HPP_RAII_ENABLED=1
    KENGINE_SHADER_DIR="${KENGINE_SHADER_OUTPUT_DIR}"
    KENGINE_CACHE_DIR="${CMAKE_BINARY_DIR}/cache"
)

add_dependencies(kengine kengine_shaders)

# ---------------------------------------------------------------------------
# Demo executable
# ---------------------------------------------------------------------------
add_executable(kengine_demo src/main.cpp)
target_link_libraries(kengine_demo PRIVATE kengine)
kengine_apply_compiler_flags(kengine_demo)