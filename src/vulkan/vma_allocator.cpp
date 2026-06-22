#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#include "kengine/vulkan/vma_allocator.hpp"
#include <stdexcept>

namespace kengine {

GpuAllocator::~GpuAllocator() {
    if (allocator_ != VK_NULL_HANDLE) {
        vmaDestroyAllocator(allocator_);
        allocator_ = VK_NULL_HANDLE;
    }
}

void GpuAllocator::init(const vk::raii::Instance& instance,
                        const vk::raii::PhysicalDevice& physical_device,
                        const vk::raii::Device& device) {
    VmaAllocatorCreateInfo info{};
    info.physicalDevice = *physical_device;
    info.device         = *device;
    info.instance       = *instance;

    VmaVulkanFunctions funcs{};
    funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    funcs.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;
    info.pVulkanFunctions       = &funcs;

    if (vmaCreateAllocator(&info, &allocator_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator");
    }
}

} // namespace kengine