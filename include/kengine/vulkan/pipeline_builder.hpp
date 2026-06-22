#pragma once

#include "kengine/vulkan/shader_module.hpp"
#include <vulkan/vulkan_raii.hpp>
#include <vector>

namespace kengine {

enum class BlendMode {
    Opaque,
    Alpha,
    Additive,
    Premultiplied
};

struct DynamicRenderingFormats {
    vk::Format color_format = vk::Format::eB8G8R8A8Srgb;
    vk::Format depth_format = vk::Format::eD32Sfloat;
    bool       has_depth    = true;
};

struct VertexBindingDesc {
    std::uint32_t binding = 0;
    std::uint32_t stride  = 0;
    vk::VertexInputRate rate = vk::VertexInputRate::eVertex;
};

struct VertexAttribDesc {
    std::uint32_t location = 0;
    std::uint32_t binding  = 0;
    vk::Format    format   = vk::Format::eR32G32Sfloat;
    std::uint32_t offset   = 0;
};

struct GraphicsPipelineDesc {
    const ShaderModule* vertex_shader   = nullptr;
    const ShaderModule* fragment_shader = nullptr;
    DynamicRenderingFormats rendering;
    std::vector<VertexBindingDesc>  bindings;
    std::vector<VertexAttribDesc>   attributes;
    vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList;
    bool depth_test   = true;
    bool depth_write  = true;
    BlendMode blend    = BlendMode::Opaque;
    vk::CullModeFlags cull = vk::CullModeFlagBits::eNone;
    std::uint32_t push_constant_size = 0;
    vk::ShaderStageFlags push_constant_stages = vk::ShaderStageFlagBits::eVertex
                                              | vk::ShaderStageFlagBits::eFragment;
};

class GraphicsPipelineBuilder {
public:
    static vk::raii::Pipeline create(
        const vk::raii::Device& device,
        const vk::raii::PipelineLayout& layout,
        const vk::raii::PipelineCache& cache,
        const GraphicsPipelineDesc& desc);

    static vk::raii::PipelineLayout create_layout(
        const vk::raii::Device& device,
        const std::vector<vk::DescriptorSetLayout>& set_layouts,
        std::uint32_t push_constant_size,
        vk::ShaderStageFlags push_stages);
};

} // namespace kengine