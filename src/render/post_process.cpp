#include "kengine/render/post_process.hpp"
#include "kengine/vulkan/dynamic_renderer.hpp"
#include "kengine/vulkan/pipeline_builder.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace kengine {

PostProcessPipeline::PostProcessPipeline(VulkanContext& ctx, PipelineCache& cache)
    : ctx_(ctx), cache_(cache) {}

void PostProcessPipeline::init(const std::filesystem::path& shader_dir,
                               vk::Format swapchain_format,
                               std::uint32_t frames_in_flight) {
    shader_dir_ = shader_dir;
    shader_fullscreen_vert_.emplace(ctx_.device(),
        shader_dir / "common/fullscreen.vert.spv");
    shader_tonemap_frag_.emplace(ctx_.device(),
        shader_dir / "post/tonemap_present.frag.spv");

    create_descriptor_resources(frames_in_flight);
    create_pipeline(swapchain_format);
    initialized_ = true;
}

void PostProcessPipeline::recreate(vk::Format swapchain_format) {
    pipeline_.reset();
    create_pipeline(swapchain_format);
}

void PostProcessPipeline::create_descriptor_resources(std::uint32_t frames_in_flight) {
    /* Future: add bindless descriptor indexing slot at binding 2+ */
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = vk::ShaderStageFlagBits::eFragment;

    bindings[1].binding         = 1;
    bindings[1].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutCreateInfo layout_info;
    layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layout_info.pBindings    = bindings.data();
    descriptor_layout_ = ctx_.device().createDescriptorSetLayout(layout_info);

    vk::DescriptorPoolSize pool_size;
    pool_size.type            = vk::DescriptorType::eCombinedImageSampler;
    pool_size.descriptorCount = frames_in_flight * 2;

    vk::DescriptorPoolCreateInfo pool_info;
    pool_info.maxSets       = frames_in_flight;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes    = &pool_size;
    descriptor_pool_ = ctx_.device().createDescriptorPool(pool_info);

    std::vector<vk::DescriptorSetLayout> layouts(frames_in_flight, *descriptor_layout_);
    vk::DescriptorSetAllocateInfo alloc_info;
    alloc_info.descriptorPool     = *descriptor_pool_;
    alloc_info.descriptorSetCount = frames_in_flight;
    alloc_info.pSetLayouts        = layouts.data();
    descriptor_sets_ = ctx_.device().allocateDescriptorSets(alloc_info);

    vk::SamplerCreateInfo sampler_info;
    sampler_info.magFilter    = vk::Filter::eLinear;
    sampler_info.minFilter    = vk::Filter::eLinear;
    sampler_info.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    sampler_info.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    sampler_ = ctx_.device().createSampler(sampler_info);
}

void PostProcessPipeline::create_pipeline(vk::Format swapchain_format) {
    pipeline_layout_ = GraphicsPipelineBuilder::create_layout(
        ctx_.device(), {*descriptor_layout_}, sizeof(TonemapPushConstants),
        vk::ShaderStageFlagBits::eFragment);

    GraphicsPipelineDesc desc;
    desc.vertex_shader   = &*shader_fullscreen_vert_;
    desc.fragment_shader = &*shader_tonemap_frag_;
    desc.rendering.color_format = swapchain_format;
    desc.rendering.has_depth    = false;
    desc.depth_test  = false;
    desc.depth_write = false;
    desc.blend       = BlendMode::Opaque;
    desc.push_constant_size = sizeof(TonemapPushConstants);
    desc.push_constant_stages = vk::ShaderStageFlagBits::eFragment;

    pipeline_ = GraphicsPipelineBuilder::create(
        ctx_.device(), *pipeline_layout_, cache_.handle(), desc);
}

void PostProcessPipeline::update_descriptors(std::uint32_t frame_index,
                                             vk::ImageView scene_color,
                                             vk::Format scene_color_format,
                                             vk::ImageView scene_depth) {
    if (frame_index >= descriptor_sets_.size()) return;

    vk::DescriptorImageInfo color_info;
    color_info.sampler     = *sampler_;
    color_info.imageView   = scene_color;
    color_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::DescriptorImageInfo depth_info;
    depth_info.sampler     = *sampler_;
    depth_info.imageView   = scene_depth;
    depth_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    std::array<vk::WriteDescriptorSet, 2> writes{};
    writes[0].dstSet          = *descriptor_sets_[frame_index];
    writes[0].dstBinding      = 0;
    writes[0].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo        = &color_info;

    writes[1].dstSet          = *descriptor_sets_[frame_index];
    writes[1].dstBinding      = 1;
    writes[1].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo        = &depth_info;

    ctx_.device().updateDescriptorSets(writes, {});

    (void)scene_color_format;
}

void PostProcessPipeline::record_present(
    const vk::raii::CommandBuffer& cmd, std::uint32_t frame_index,
    vk::ImageView swapchain_view, vk::Extent2D extent,
    const TonemapPushConstants& pc) {

    if (!pipeline_ || frame_index >= descriptor_sets_.size()) return;

    DynamicRenderTarget target;
    target.color_view   = swapchain_view;
    target.extent       = extent;
    target.has_depth    = false;

    ClearValues clear;
    DynamicRenderer::begin(cmd, target, clear);

    vk::Viewport viewport{0, 0, static_cast<float>(extent.width),
                          static_cast<float>(extent.height), 0, 1};
    vk::Rect2D scissor{{0, 0}, extent};
    cmd.setViewport(0, viewport);
    cmd.setScissor(0, scissor);

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline_);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_layout_,
                           0, {*descriptor_sets_[frame_index]}, {});

    cmd.pushConstants<TonemapPushConstants>(
        *pipeline_layout_, vk::ShaderStageFlagBits::eFragment, 0, pc);

    cmd.draw(3, 1, 0, 0);
    DynamicRenderer::end(cmd);
}

void PostProcessPipeline::configure(const FrameGraphConfig& config) {
    config_ = config;
}

void PostProcessPipeline::register_passes(FrameGraph& /*frame_graph*/) {}

// Simple helper to produce tonemap constants for testing the post chain (bloom/dof/taa/sharpen + final).
// Called by frame renderer; values chosen to visibly exercise current effects.
TonemapPushConstants PostProcessPipeline::make_test_tonemap_constants(float time, float w_morph, float scan) const {
    TonemapPushConstants pc;
    // Exposure breathes a little to show hdr -> tonemap
    pc.exposure = 1.15f + 0.25f * sinf(time * 0.7f) * (0.5f + 0.5f * w_morph);
    pc.scanline_strength = scan;
    pc.w_morph = w_morph;
    pc.time = time;
    // Future: feed config_.bloom_*, dof_* into other passes when implemented.
    (void)config_;
    return pc;
}

} // namespace kengine