#include "kengine/vulkan/dynamic_renderer.hpp"

namespace kengine {

namespace {

void image_barrier(
    const vk::raii::CommandBuffer& cmd,
    vk::Image image,
    vk::ImageAspectFlags aspect,
    vk::PipelineStageFlags2 src_stage,
    vk::PipelineStageFlags2 dst_stage,
    vk::AccessFlags2 src_access,
    vk::AccessFlags2 dst_access,
    vk::ImageLayout old_layout,
    vk::ImageLayout new_layout) {

    vk::ImageMemoryBarrier2 barrier;
    barrier.image                       = image;
    barrier.srcStageMask                = src_stage;
    barrier.dstStageMask                = dst_stage;
    barrier.srcAccessMask               = src_access;
    barrier.dstAccessMask               = dst_access;
    barrier.oldLayout                   = old_layout;
    barrier.newLayout                   = new_layout;
    barrier.subresourceRange.aspectMask = aspect;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;

    vk::DependencyInfo dep;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &barrier;
    cmd.pipelineBarrier2(dep);
}

} // namespace

void DynamicRenderer::begin(
    const vk::raii::CommandBuffer& cmd,
    const DynamicRenderTarget& target,
    const ClearValues& clear) {

    vk::RenderingAttachmentInfo color_att;
    color_att.imageView   = target.color_view;
    color_att.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    color_att.loadOp      = vk::AttachmentLoadOp::eClear;
    color_att.storeOp     = vk::AttachmentStoreOp::eStore;
    color_att.clearValue  = vk::ClearColorValue(clear.color);

    vk::RenderingInfo rendering;
    rendering.renderArea   = vk::Rect2D{{0, 0}, target.extent};
    rendering.layerCount   = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments  = &color_att;

    vk::RenderingAttachmentInfo depth_att;
    if (target.has_depth && target.depth_view) {
        depth_att.imageView   = target.depth_view;
        depth_att.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        depth_att.loadOp      = vk::AttachmentLoadOp::eClear;
        depth_att.storeOp     = vk::AttachmentStoreOp::eStore;
        depth_att.clearValue  = vk::ClearDepthStencilValue(clear.depth, 0);
        rendering.pDepthAttachment = &depth_att;
    }

    cmd.beginRendering(rendering);
}

void DynamicRenderer::end(const vk::raii::CommandBuffer& cmd) {
    cmd.endRendering();
}

void DynamicRenderer::transition_color_to_attachment(
    const vk::raii::CommandBuffer& cmd, vk::Image image) {
    image_barrier(cmd, image, vk::ImageAspectFlagBits::eColor,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        {}, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
}

void DynamicRenderer::transition_depth_to_attachment(
    const vk::raii::CommandBuffer& cmd, vk::Image image) {
    image_barrier(cmd, image, vk::ImageAspectFlagBits::eDepth,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests,
        {}, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
}

void DynamicRenderer::transition_color_to_shader_read(
    const vk::raii::CommandBuffer& cmd, vk::Image image) {
    image_barrier(cmd, image, vk::ImageAspectFlagBits::eColor,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal);
}

void DynamicRenderer::transition_depth_to_shader_read(
    const vk::raii::CommandBuffer& cmd, vk::Image image) {
    image_barrier(cmd, image, vk::ImageAspectFlagBits::eDepth,
        vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::ImageLayout::eDepthStencilAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal);
}

void DynamicRenderer::transition_to_present(
    const vk::raii::CommandBuffer& cmd, vk::Image image) {
    image_barrier(cmd, image, vk::ImageAspectFlagBits::eColor,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eBottomOfPipe,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        {},
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR);
}

void DynamicRenderer::transition_to_storage_general(
    const vk::raii::CommandBuffer& cmd, vk::Image image) {
    image_barrier(cmd, image, vk::ImageAspectFlagBits::eColor,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
        vk::ImageLayout::eUndefined,   // don't care previous for post passes
        vk::ImageLayout::eGeneral);
}

void DynamicRenderer::transition_storage_to_shader_read(
    const vk::raii::CommandBuffer& cmd, vk::Image image) {
    image_barrier(cmd, image, vk::ImageAspectFlagBits::eColor,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eShaderWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eShaderReadOnlyOptimal);
}

} // namespace kengine