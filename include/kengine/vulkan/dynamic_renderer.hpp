#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <array>

namespace kengine {

struct DynamicRenderTarget {
    vk::ImageView color_view;
    vk::ImageView depth_view;
    vk::Extent2D  extent;
    vk::Format    color_format = vk::Format::eB8G8R8A8Srgb;
    vk::Format    depth_format = vk::Format::eD32Sfloat;
    bool          has_depth    = false;
};

struct ClearValues {
    std::array<float, 4> color = {0.02f, 0.02f, 0.06f, 1.0f};
    float depth = 1.0f;
};

class DynamicRenderer {
public:
    static void begin(
        const vk::raii::CommandBuffer& cmd,
        const DynamicRenderTarget& target,
        const ClearValues& clear = {});

    static void end(const vk::raii::CommandBuffer& cmd);

    static void transition_color_to_attachment(
        const vk::raii::CommandBuffer& cmd, vk::Image image);

    static void transition_depth_to_attachment(
        const vk::raii::CommandBuffer& cmd, vk::Image image);

    static void transition_color_to_shader_read(
        const vk::raii::CommandBuffer& cmd, vk::Image image);

    static void transition_depth_to_shader_read(
        const vk::raii::CommandBuffer& cmd, vk::Image image);

    static void transition_to_present(
        const vk::raii::CommandBuffer& cmd, vk::Image image);

    /* Legacy alias */
    static void transition_to_color_attachment(
        const vk::raii::CommandBuffer& cmd, vk::Image image) {
        transition_color_to_attachment(cmd, image);
    }
};

} // namespace kengine