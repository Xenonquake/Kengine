include(FetchContent)

set(FETCHCONTENT_QUIET OFF)

# GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# ------------------------------------------------------------------
# Vulkan Memory Allocator (VMA) - Official integration
#
# The project now prefers a local checkout of the official VMA:
#   ~/Downloads/GitHub_Repos/VulkanMemoryAllocator
#
# You can override with:
#   cmake -DVMA_LOCAL_PATH=/path/to/your/VulkanMemoryAllocator ..
#
# Falls back to FetchContent only if the local header is missing.
# ------------------------------------------------------------------
set(VMA_LOCAL_PATH "$ENV{HOME}/Downloads/GitHub_Repos/VulkanMemoryAllocator"
    CACHE PATH "Path to local official VulkanMemoryAllocator checkout")

if(EXISTS "${VMA_LOCAL_PATH}/include/vk_mem_alloc.h")
    message(STATUS "Using official local VMA from ${VMA_LOCAL_PATH}")
    set(VMA_INCLUDE_DIR "${VMA_LOCAL_PATH}/include" CACHE PATH "VMA include directory")
else()
    message(STATUS "Local VMA not found – falling back to FetchContent")
    FetchContent_Declare(
        vma
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG        v3.1.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(vma)
    set(VMA_INCLUDE_DIR "${vma_SOURCE_DIR}/include")
endif()

# stb_image for texture loading (header only)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(stb)

# cglm - lightweight C math library (SIMD friendly) for hot C23 paths
FetchContent_Declare(
    cglm
    GIT_REPOSITORY https://github.com/recp/cglm.git
    GIT_TAG        v0.9.4
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cglm)

# GLM - header-only C++ math for ergonomics in C++ systems
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glm)

# tinygltf - header-only glTF 2.0 loader for loading .glb models
FetchContent_Declare(
    tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG        v2.9.0
    GIT_SHALLOW    TRUE
)
set(TINYGLTF_HEADER_ONLY ON CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinygltf)