#include "kengine/vulkan/gpu_image.hpp"
#include <stdexcept>

namespace kengine {

GpuImage::GpuImage(const vk::raii::Device& device, VmaAllocator allocator, const GpuImageDesc& desc)
    : device_(&device), allocator_(allocator) {
    create(desc);
}

GpuImage::~GpuImage() {
    destroy();
}

GpuImage::GpuImage(GpuImage&& other) noexcept {
    *this = std::move(other);
}

GpuImage& GpuImage::operator=(GpuImage&& other) noexcept {
    if (this != &other) {
        destroy();
        device_     = other.device_;
        allocator_  = other.allocator_;
        desc_       = other.desc_;
        image_      = other.image_;
        allocation_ = other.allocation_;
        view_       = std::move(other.view_);
        other.image_      = VK_NULL_HANDLE;
        other.allocation_ = VK_NULL_HANDLE;
        other.device_     = nullptr;
    }
    return *this;
}

void GpuImage::destroy() {
    view_.reset();
    if (image_ != VK_NULL_HANDLE && allocator_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, image_, allocation_);
        image_      = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
    }
}

void GpuImage::recreate(const GpuImageDesc& desc) {
    destroy();
    create(desc);
}

void GpuImage::create(const GpuImageDesc& desc) {
    if (!device_ || allocator_ == VK_NULL_HANDLE) {
        throw std::runtime_error("GpuImage: invalid device or allocator");
    }
    if (desc.extent.width == 0 || desc.extent.height == 0) {
        throw std::runtime_error("GpuImage: invalid extent");
    }

    desc_ = desc;

    VkImageCreateInfo image_info{};
    image_info.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType   = VK_IMAGE_TYPE_2D;
    image_info.format      = static_cast<VkFormat>(desc.format);
    image_info.extent      = {desc.extent.width, desc.extent.height, 1};
    image_info.mipLevels   = 1;
    image_info.arrayLayers = 1;
    image_info.samples     = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling      = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage       = static_cast<VkImageUsageFlags>(desc.usage);
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateImage(allocator_, &image_info, &alloc_info, &image_, &allocation_, nullptr)
        != VK_SUCCESS) {
        throw std::runtime_error("GpuImage: vmaCreateImage failed");
    }

    vk::ImageViewCreateInfo view_info;
    view_info.image    = image_;
    view_info.viewType = vk::ImageViewType::e2D;
    view_info.format   = desc.format;
    view_info.subresourceRange.aspectMask = desc.aspect;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.layerCount = 1;

    view_ = device_->createImageView(view_info);
}

} // namespace kengine