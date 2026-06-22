#include "kengine/vulkan/pipeline_builder.hpp"

namespace kengine {

static vk::PipelineColorBlendAttachmentState make_blend(BlendMode mode) {
    vk::PipelineColorBlendAttachmentState blend;
    blend.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
                         | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

    switch (mode) {
    case BlendMode::Alpha:
        blend.blendEnable         = VK_TRUE;
        blend.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
        blend.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        blend.colorBlendOp        = vk::BlendOp::eAdd;
        blend.srcAlphaBlendFactor = vk::BlendFactor::eOne;
        blend.dstAlphaBlendFactor = vk::BlendFactor::eZero;
        blend.alphaBlendOp        = vk::BlendOp::eAdd;
        break;
    case BlendMode::Additive:
        blend.blendEnable         = VK_TRUE;
        blend.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
        blend.dstColorBlendFactor = vk::BlendFactor::eOne;
        blend.colorBlendOp        = vk::BlendOp::eAdd;
        blend.srcAlphaBlendFactor = vk::BlendFactor::eOne;
        blend.dstAlphaBlendFactor = vk::BlendFactor::eOne;
        blend.alphaBlendOp        = vk::BlendOp::eAdd;
        break;
    case BlendMode::Premultiplied:
        blend.blendEnable         = VK_TRUE;
        blend.srcColorBlendFactor = vk::BlendFactor::eOne;
        blend.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        blend.colorBlendOp        = vk::BlendOp::eAdd;
        blend.srcAlphaBlendFactor = vk::BlendFactor::eOne;
        blend.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        blend.alphaBlendOp        = vk::BlendOp::eAdd;
        break;
    default:
        blend.blendEnable = VK_FALSE;
        break;
    }
    return blend;
}

vk::raii::PipelineLayout GraphicsPipelineBuilder::create_layout(
    const vk::raii::Device& device,
    const std::vector<vk::DescriptorSetLayout>& set_layouts,
    std::uint32_t push_constant_size,
    vk::ShaderStageFlags push_stages) {

    vk::PipelineLayoutCreateInfo info;
    info.setLayoutCount = static_cast<std::uint32_t>(set_layouts.size());
    info.pSetLayouts    = set_layouts.data();

    vk::PushConstantRange pcr;
    pcr.stageFlags = push_stages;
    pcr.offset     = 0;
    pcr.size       = push_constant_size;

    if (push_constant_size > 0) {
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges    = &pcr;
    }

    return device.createPipelineLayout(info);
}

vk::raii::Pipeline GraphicsPipelineBuilder::create(
    const vk::raii::Device& device,
    const vk::raii::PipelineLayout& layout,
    const vk::raii::PipelineCache& cache,
    const GraphicsPipelineDesc& desc) {

    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    if (desc.vertex_shader && desc.vertex_shader->valid()) {
        stages.push_back({{}, vk::ShaderStageFlagBits::eVertex, desc.vertex_shader->handle(), "main"});
    }
    if (desc.fragment_shader && desc.fragment_shader->valid()) {
        stages.push_back({{}, vk::ShaderStageFlagBits::eFragment, desc.fragment_shader->handle(), "main"});
    }

    std::vector<vk::VertexInputBindingDescription> bindings;
    for (const auto& b : desc.bindings) {
        bindings.push_back({b.binding, b.stride, b.rate});
    }

    std::vector<vk::VertexInputAttributeDescription> attribs;
    for (const auto& a : desc.attributes) {
        attribs.push_back({a.location, a.binding, a.format, a.offset});
    }

    vk::PipelineVertexInputStateCreateInfo vertex_input;
    vertex_input.vertexBindingDescriptionCount   = static_cast<std::uint32_t>(bindings.size());
    vertex_input.pVertexBindingDescriptions      = bindings.data();
    vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attribs.size());
    vertex_input.pVertexAttributeDescriptions    = attribs.data();

    vk::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = desc.topology;

    vk::PipelineViewportStateCreateInfo viewport;
    viewport.viewportCount = 1;
    viewport.scissorCount  = 1;

    vk::PipelineRasterizationStateCreateInfo raster;
    raster.polygonMode = vk::PolygonMode::eFill;
    raster.cullMode    = desc.cull;
    raster.frontFace   = vk::FrontFace::eCounterClockwise;
    raster.lineWidth   = 1.0f;

    vk::PipelineMultisampleStateCreateInfo msaa;
    msaa.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::PipelineDepthStencilStateCreateInfo depth;
    depth.depthTestEnable  = desc.depth_test ? VK_TRUE : VK_FALSE;
    depth.depthWriteEnable = desc.depth_write ? VK_TRUE : VK_FALSE;
    depth.depthCompareOp   = vk::CompareOp::eLess;

    auto blend_attachment = make_blend(desc.blend);
    vk::PipelineColorBlendStateCreateInfo blend;
    blend.attachmentCount = 1;
    blend.pAttachments    = &blend_attachment;

    std::vector<vk::DynamicState> dynamic_states = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };
    vk::PipelineDynamicStateCreateInfo dynamic;
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates    = dynamic_states.data();

    vk::Format color_format = desc.rendering.color_format;
    vk::PipelineRenderingCreateInfo rendering;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &color_format;

    vk::Format depth_format = desc.rendering.depth_format;
    if (desc.rendering.has_depth) {
        rendering.depthAttachmentFormat = depth_format;
    }

    vk::GraphicsPipelineCreateInfo info;
    info.stageCount          = static_cast<std::uint32_t>(stages.size());
    info.pStages             = stages.data();
    info.pVertexInputState   = &vertex_input;
    info.pInputAssemblyState = &input_assembly;
    info.pViewportState      = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState   = &msaa;
    info.pDepthStencilState  = &depth;
    info.pColorBlendState    = &blend;
    info.pDynamicState       = &dynamic;
    info.layout              = *layout;
    info.pNext               = &rendering;

    auto result = device.createGraphicsPipeline(cache, info);
#if defined(VK_HPP_NO_EXCEPTIONS)
    return std::move(result.value);
#else
    return std::move(result);
#endif
}

} // namespace kengine