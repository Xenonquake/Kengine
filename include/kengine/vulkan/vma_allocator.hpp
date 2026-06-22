#pragma once

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vk_mem_alloc.h>

namespace kengine {

class GpuAllocator {
public:
    GpuAllocator() = default;
    ~GpuAllocator();

    GpuAllocator(const GpuAllocator&) = delete;
    GpuAllocator& operator=(const GpuAllocator&) = delete;

    void init(const vk::raii::Instance& instance,
              const vk::raii::PhysicalDevice& physical_device,
              const vk::raii::Device& device);

    VmaAllocator handle() const { return allocator_; }

private:
    VmaAllocator allocator_ = VK_NULL_HANDLE;
};

} // namespace kengine