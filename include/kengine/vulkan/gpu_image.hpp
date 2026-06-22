#pragma once

#include "kengine/vulkan/vma_allocator.hpp"
#include <optional>
#include <vulkan/vulkan_raii.hpp>

namespace kengine {

struct GpuImageDesc {
    vk::Extent2D extent{};
    vk::Format   format       = vk::Format::eR16G16B16A16Sfloat;
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment
                              | vk::ImageUsageFlagBits::eSampled
                              | vk::ImageUsageFlagBits::eTransferSrc;
    vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
};

class GpuImage {
public:
    GpuImage() = default;
    GpuImage(const vk::raii::Device& device, VmaAllocator allocator, const GpuImageDesc& desc);
    ~GpuImage();

    GpuImage(GpuImage&&) noexcept;
    GpuImage& operator=(GpuImage&&) noexcept;

    GpuImage(const GpuImage&) = delete;
    GpuImage& operator=(const GpuImage&) = delete;

    void destroy();
    void recreate(const GpuImageDesc& desc);

    vk::Image     image() const { return image_; }
    vk::ImageView view() const { return *view_; }
    vk::Format    format() const { return desc_.format; }
    vk::Extent2D  extent() const { return desc_.extent; }
    bool          valid() const { return image_ != VK_NULL_HANDLE; }

private:
    void create(const GpuImageDesc& desc);

    const vk::raii::Device* device_   = nullptr;
    VmaAllocator            allocator_ = VK_NULL_HANDLE;
    GpuImageDesc            desc_{};

    VkImage         image_      = VK_NULL_HANDLE;
    VmaAllocation   allocation_ = VK_NULL_HANDLE;
    std::optional<vk::raii::ImageView> view_;
};

} // namespace kengine