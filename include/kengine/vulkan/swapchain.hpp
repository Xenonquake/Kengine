#pragma once

#include "kengine/vulkan/context.hpp"
#include <vulkan/vulkan_raii.hpp>
#include <vector>

namespace kengine {

class Swapchain {
public:
    Swapchain(VulkanContext& ctx, vk::Extent2D desired_extent);
    ~Swapchain() = default;

    void recreate(vk::Extent2D extent);

    vk::Format image_format() const { return format_; }
    vk::Extent2D extent() const { return extent_; }
    const std::vector<vk::raii::ImageView>& image_views() const { return image_views_; }
    const std::vector<vk::Image>& images() const { return images_; }
    vk::Image image(std::uint32_t index) const { return images_[index]; }
    const vk::raii::SwapchainKHR& handle() const { return *swapchain_; }
    const vk::raii::SurfaceKHR& surface() const { return surface_; }

private:
    void create(vk::Extent2D extent);
    vk::SurfaceFormatKHR choose_format(const std::vector<vk::SurfaceFormatKHR>& formats);
    vk::PresentModeKHR choose_present_mode(const std::vector<vk::PresentModeKHR>& modes, bool vsync);

    VulkanContext& ctx_;
    vk::raii::SurfaceKHR surface_{nullptr};
    std::optional<vk::raii::SwapchainKHR> swapchain_;
    std::vector<vk::Image> images_;
    std::vector<vk::raii::ImageView> image_views_;
    vk::Format format_ = vk::Format::eB8G8R8A8Srgb;
    vk::Extent2D extent_{};
};

} // namespace kengine