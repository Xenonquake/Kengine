#include <vulkan/vulkan.h>
#include "kengine/vulkan/swapchain.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace kengine {

Swapchain::Swapchain(VulkanContext& ctx, vk::Extent2D desired_extent) : ctx_(ctx) {
    VkSurfaceKHR raw_surface = VK_NULL_HANDLE;
    VkInstance vk_instance = static_cast<VkInstance>(static_cast<vk::Instance>(ctx.instance()));
    if (glfwCreateWindowSurface(vk_instance, ctx.window(), nullptr, &raw_surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface");
    }
    surface_ = vk::raii::SurfaceKHR{ctx.instance(), raw_surface};
    create(desired_extent);
}

void Swapchain::recreate(vk::Extent2D extent) {
    ctx_.device().waitIdle();
    image_views_.clear();
    swapchain_.reset();
    create(extent);
}

void Swapchain::create(vk::Extent2D extent) {
    auto caps = ctx_.physical_device().getSurfaceCapabilitiesKHR(*surface_);
    auto formats = ctx_.physical_device().getSurfaceFormatsKHR(*surface_);
    auto modes   = ctx_.physical_device().getSurfacePresentModesKHR(*surface_);

    auto surface_format = choose_format(formats);
    format_ = surface_format.format;
    auto present_mode = choose_present_mode(modes, true);

    extent.width  = std::clamp(extent.width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
    extent.height = std::clamp(extent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    extent_ = extent;

    std::uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
        image_count = caps.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR sci;
    sci.surface          = *surface_;
    sci.minImageCount    = image_count;
    sci.imageFormat      = format_;
    sci.imageColorSpace  = surface_format.colorSpace;
    sci.imageExtent      = extent_;
    sci.imageArrayLayers = 1;
    sci.imageUsage       = vk::ImageUsageFlagBits::eColorAttachment
                         | vk::ImageUsageFlagBits::eTransferSrc
                         | vk::ImageUsageFlagBits::eTransferDst;
    sci.imageSharingMode = vk::SharingMode::eExclusive;
    sci.preTransform     = caps.currentTransform;
    sci.compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    sci.presentMode      = present_mode;
    sci.clipped          = VK_TRUE;

    swapchain_ = ctx_.device().createSwapchainKHR(sci);
    images_ = swapchain_->getImages();

    image_views_.reserve(images_.size());
    for (auto image : images_) {
        vk::ImageViewCreateInfo ivci;
        ivci.image    = image;
        ivci.viewType = vk::ImageViewType::e2D;
        ivci.format   = format_;
        ivci.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        image_views_.push_back(ctx_.device().createImageView(ivci));
    }
}

vk::SurfaceFormatKHR Swapchain::choose_format(const std::vector<vk::SurfaceFormatKHR>& formats) {
    for (const auto& f : formats) {
        if (f.format == vk::Format::eB8G8R8A8Srgb
            && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return f;
        }
    }
    return formats.front();
}

vk::PresentModeKHR Swapchain::choose_present_mode(
    const std::vector<vk::PresentModeKHR>& modes, bool vsync) {
    if (!vsync) {
        for (auto m : modes) {
            if (m == vk::PresentModeKHR::eMailbox) return m;
        }
        for (auto m : modes) {
            if (m == vk::PresentModeKHR::eImmediate) return m;
        }
    }
    return vk::PresentModeKHR::eFifo;
}

} // namespace kengine