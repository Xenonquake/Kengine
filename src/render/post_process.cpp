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

    shader_bloom_threshold_.emplace(ctx_.device(),
        shader_dir / "post/bloom_threshold.comp.spv");
    shader_bloom_blur_.emplace(ctx_.device(),
        shader_dir / "post/bloom_blur.comp.spv");

    create_descriptor_resources(frames_in_flight);
    create_pipeline(swapchain_format);
    create_bloom_compute_pipelines();
    // bloom storage set allocation happens inside create_descriptor_resources after layout is ready? 
    // (moved creation of bloom storage set to after bloom layout in practice)
    initialized_ = true;
}

void PostProcessPipeline::recreate(vk::Format swapchain_format) {
    pipeline_.reset();
    create_pipeline(swapchain_format);

    // Recreate intermediate images at new size (caller should have waited idle)
    if (current_extent_.width > 0 && current_extent_.height > 0) {
        destroy_post_resources();
        create_post_resources(current_extent_);
    }
}

void PostProcessPipeline::create_descriptor_resources(std::uint32_t frames_in_flight) {
    /* Binding 0: scene color, 1: scene depth, 2: bloom (for compositing) */
    std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = vk::ShaderStageFlagBits::eFragment;

    bindings[1].binding         = 1;
    bindings[1].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = vk::ShaderStageFlagBits::eFragment;

    bindings[2].binding         = 2;
    bindings[2].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags      = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutCreateInfo layout_info;
    layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layout_info.pBindings    = bindings.data();
    descriptor_layout_ = ctx_.device().createDescriptorSetLayout(layout_info);

    vk::DescriptorPoolSize pool_size;
    pool_size.type            = vk::DescriptorType::eCombinedImageSampler;
    pool_size.descriptorCount = frames_in_flight * 3;

    vk::DescriptorPoolCreateInfo pool_info;
    pool_info.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
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

    /* Begin bindless prep for sprite texture atlases (future use with retro sprites too).
       A separate large array binding for combined samplers using descriptor indexing. */
    if (true) {  // guarded by ext already requested at device level
        vk::DescriptorSetLayoutBinding bindlessBind{};
        bindlessBind.binding = 2;
        bindlessBind.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        bindlessBind.descriptorCount = 1024;  // large bindless capacity
        bindlessBind.stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex;
        bindlessBind.pImmutableSamplers = nullptr;

        vk::DescriptorSetLayoutCreateInfo blInfo;
        blInfo.bindingCount = 1;
        blInfo.pBindings = &bindlessBind;

        // Flag for variable count / update after bind if driver supports (optional for starters)
        vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
        vk::DescriptorBindingFlags bindFlag = vk::DescriptorBindingFlagBits::ePartiallyBound
                                            | vk::DescriptorBindingFlagBits::eVariableDescriptorCount;
        bindingFlagsInfo.bindingCount = 1;
        bindingFlagsInfo.pBindingFlags = &bindFlag;
        blInfo.pNext = &bindingFlagsInfo;

        // Note: we don't create/keep the layout here yet; this demonstrates the structure.
        // A real implementation will keep a bindless set + pool sized for many atlases.
        (void)blInfo;
    }
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
                                             vk::ImageView scene_depth,
                                             vk::ImageView bloom) {
    if (frame_index >= descriptor_sets_.size()) return;

    vk::DescriptorImageInfo color_info;
    color_info.sampler     = *sampler_;
    color_info.imageView   = scene_color;
    color_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::DescriptorImageInfo depth_info;
    depth_info.sampler     = *sampler_;
    depth_info.imageView   = scene_depth;
    depth_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::DescriptorImageInfo bloom_info;
    bloom_info.sampler     = *sampler_;
    bloom_info.imageView   = bloom ? bloom : scene_color;  // fallback if no bloom
    bloom_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    std::array<vk::WriteDescriptorSet, 3> writes{};
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

    writes[2].dstSet          = *descriptor_sets_[frame_index];
    writes[2].dstBinding      = 2;
    writes[2].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
    writes[2].descriptorCount = 1;
    writes[2].pImageInfo        = &bloom_info;

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

void PostProcessPipeline::on_resize(vk::Extent2D extent) {
    if (extent.width == current_extent_.width && extent.height == current_extent_.height) return;
    destroy_post_resources();
    create_post_resources(extent);
}

void PostProcessPipeline::register_passes(FrameGraph& /*frame_graph*/) {}

void PostProcessPipeline::create_bloom_compute_pipelines() {
    if (!shader_bloom_threshold_ || !shader_bloom_blur_) return;

    // Simple layout for storage image compute (no descriptor set yet, we will bind via descriptor or use push for now)
    // For these shaders, they use binding 0 and 1 as storage images - we will create a minimal layout.
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = vk::DescriptorType::eStorageImage;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = vk::ShaderStageFlagBits::eCompute;

    bindings[1].binding = 1;
    bindings[1].descriptorType = vk::DescriptorType::eStorageImage;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = vk::ShaderStageFlagBits::eCompute;

    vk::DescriptorSetLayoutCreateInfo layout_info;
    layout_info.bindingCount = 2;
    layout_info.pBindings = bindings.data();
    bloom_compute_desc_layout_ = ctx_.device().createDescriptorSetLayout(layout_info);

    bloom_compute_layout_ = GraphicsPipelineBuilder::create_layout(
        ctx_.device(), {*bloom_compute_desc_layout_}, sizeof(float) * 2 /* direction or threshold+intensity via push */,
        vk::ShaderStageFlagBits::eCompute);

    // We will manage a small descriptor pool for bloom compute later if needed.
    // For initial implementation we create pipelines directly and will update descriptors at record time.

    vk::ComputePipelineCreateInfo ci_threshold;
    ci_threshold.stage = {{}, vk::ShaderStageFlagBits::eCompute, shader_bloom_threshold_->handle(), "main"};
    ci_threshold.layout = *bloom_compute_layout_;
    bloom_threshold_pipeline_ = ctx_.device().createComputePipeline(nullptr, ci_threshold);

    vk::ComputePipelineCreateInfo ci_blur;
    ci_blur.stage = {{}, vk::ShaderStageFlagBits::eCompute, shader_bloom_blur_->handle(), "main"};
    ci_blur.layout = *bloom_compute_layout_;
    bloom_blur_pipeline_ = ctx_.device().createComputePipeline(nullptr, ci_blur);

    // Allocate bloom storage descriptor set now that layout exists
    if (bloom_compute_desc_layout_) {
        vk::DescriptorPoolSize storage_size;
        storage_size.type = vk::DescriptorType::eStorageImage;
        storage_size.descriptorCount = 4;

        vk::DescriptorPoolCreateInfo bpool;
        bpool.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        bpool.maxSets = 2;
        bpool.poolSizeCount = 1;
        bpool.pPoolSizes = &storage_size;
        bloom_storage_pool_ = ctx_.device().createDescriptorPool(bpool);

        std::vector<vk::DescriptorSetLayout> ls{*bloom_compute_desc_layout_};
        vk::DescriptorSetAllocateInfo ba;
        ba.descriptorPool = *bloom_storage_pool_;
        ba.descriptorSetCount = 1;
        ba.pSetLayouts = ls.data();
        auto sets = ctx_.device().allocateDescriptorSets(ba);
        if (!sets.empty()) bloom_storage_set_ = std::move(sets.front());
    }
}

void PostProcessPipeline::create_post_resources(vk::Extent2D extent) {
    if (extent.width == 0 || extent.height == 0) return;
    current_extent_ = extent;

    // Bloom buffers (half res is common, but for simplicity use full for now; can optimize later)
    GpuImageDesc desc;
    desc.extent = extent;
    desc.format = vk::Format::eR8G8B8A8Unorm;
    desc.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
    desc.aspect = vk::ImageAspectFlagBits::eColor;

    bloom_input_ = GpuImage(ctx_.device(), ctx_.allocator().handle(), desc);
    bloom_output_ = GpuImage(ctx_.device(), ctx_.allocator().handle(), desc);
}

void PostProcessPipeline::destroy_post_resources() {
    bloom_input_ = {};
    bloom_output_ = {};
    current_extent_ = vk::Extent2D{};
}

void PostProcessPipeline::record_bloom_threshold(const vk::raii::CommandBuffer& cmd, std::uint32_t /*frame_index*/, vk::Extent2D extent) {
    if (!bloom_threshold_pipeline_ || !bloom_input_.valid() || !bloom_output_.valid() || !bloom_storage_set_) {
        if (extent.width > 0) create_post_resources(extent);
        return;
    }

    // Transition images for storage access
    DynamicRenderer::transition_to_storage_general(cmd, bloom_input_.image());
    DynamicRenderer::transition_to_storage_general(cmd, bloom_output_.image());

    // Update the storage descriptor set: binding 0 = input (read), binding 1 = output (write)
    vk::DescriptorImageInfo in_info;
    in_info.imageView = bloom_input_.view();
    in_info.imageLayout = vk::ImageLayout::eGeneral;

    vk::DescriptorImageInfo out_info;
    out_info.imageView = bloom_output_.view();
    out_info.imageLayout = vk::ImageLayout::eGeneral;

    std::array<vk::WriteDescriptorSet, 2> writes{};
    writes[0].dstSet = *bloom_storage_set_;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = vk::DescriptorType::eStorageImage;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &in_info;

    writes[1].dstSet = *bloom_storage_set_;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = vk::DescriptorType::eStorageImage;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &out_info;

    ctx_.device().updateDescriptorSets(writes, {});

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *bloom_threshold_pipeline_);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *bloom_compute_layout_, 0, {*bloom_storage_set_}, {});

    struct BloomThresholdPC { float threshold; float intensity; } pc{config_.bloom_threshold, config_.bloom_intensity};
    cmd.pushConstants<BloomThresholdPC>(*bloom_compute_layout_, vk::ShaderStageFlagBits::eCompute, 0, pc);

    uint32_t groupsX = (extent.width + 15) / 16;
    uint32_t groupsY = (extent.height + 15) / 16;
    cmd.dispatch(groupsX, groupsY, 1);

    // Transition back for potential later sampled use
    DynamicRenderer::transition_storage_to_shader_read(cmd, bloom_output_.image());
}

void PostProcessPipeline::record_bloom_blur(const vk::raii::CommandBuffer& cmd, std::uint32_t /*frame_index*/, vk::Extent2D extent) {
    if (!bloom_blur_pipeline_ || !bloom_input_.valid() || !bloom_output_.valid() || !bloom_storage_set_) return;

    auto dispatch_blur = [&](float dx, float dy) {
        DynamicRenderer::transition_to_storage_general(cmd, bloom_input_.image());
        DynamicRenderer::transition_to_storage_general(cmd, bloom_output_.image());

        // Update descriptors (ping-pong: treat output as write target, input as read)
        vk::DescriptorImageInfo in_info{ nullptr, bloom_input_.view(), vk::ImageLayout::eGeneral };
        vk::DescriptorImageInfo out_info{ nullptr, bloom_output_.view(), vk::ImageLayout::eGeneral };

        std::array<vk::WriteDescriptorSet, 2> writes{};
        writes[0].dstSet = *bloom_storage_set_; writes[0].dstBinding = 0;
        writes[0].descriptorType = vk::DescriptorType::eStorageImage; writes[0].descriptorCount = 1; writes[0].pImageInfo = &in_info;
        writes[1].dstSet = *bloom_storage_set_; writes[1].dstBinding = 1;
        writes[1].descriptorType = vk::DescriptorType::eStorageImage; writes[1].descriptorCount = 1; writes[1].pImageInfo = &out_info;
        ctx_.device().updateDescriptorSets(writes, {});

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *bloom_blur_pipeline_);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *bloom_compute_layout_, 0, {*bloom_storage_set_}, {});

        struct BlurPC { float direction[2]; } pc{{dx, dy}};
        cmd.pushConstants<BlurPC>(*bloom_compute_layout_, vk::ShaderStageFlagBits::eCompute, 0, pc);

        uint32_t gx = (extent.width + 15) / 16;
        uint32_t gy = (extent.height + 15) / 16;
        cmd.dispatch(gx, gy, 1);

        DynamicRenderer::transition_storage_to_shader_read(cmd, bloom_output_.image());
    };

    // Two pass separable (note: for real ping-pong we should swap input/output between passes)
    dispatch_blur(1.0f, 0.0f);   // horizontal
    // Swap roles for second pass (simple: rebind by swapping which we treat as input)
    std::swap(bloom_input_, bloom_output_);
    dispatch_blur(0.0f, 1.0f);   // vertical
    std::swap(bloom_input_, bloom_output_);
}

// Simple helper to produce tonemap constants for testing the post chain (bloom/dof/taa/sharpen + final).
// Called by frame renderer; values chosen to visibly exercise current effects.
TonemapPushConstants PostProcessPipeline::make_test_tonemap_constants(float time, float w_morph, float scan) const {
    TonemapPushConstants pc;
    // Exposure breathes a little to show scene -> tonemap
    pc.exposure = 1.15f + 0.25f * sinf(time * 0.7f) * (0.5f + 0.5f * w_morph);
    pc.scanline_strength = scan;
    pc.w_morph = w_morph;
    pc.time = time;
    // Future: feed config_.bloom_*, dof_* into other passes when implemented.
    (void)config_;
    return pc;
}

} // namespace kengine