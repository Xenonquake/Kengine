#include "kengine/render/frame_renderer.hpp"
#include "kengine/vulkan/dynamic_renderer.hpp"
#include <cstring>
#include <stdexcept>

namespace kengine {

FrameRenderer::FrameRenderer(VulkanContext& ctx, Swapchain& swapchain,
                             PipelineManager& pipelines, PostProcessPipeline& post,
                             FrameGraph& frame_graph)
    : ctx_(ctx), swapchain_(swapchain), pipelines_(pipelines), post_(post),
      frame_graph_(frame_graph), frames_in_flight_(ctx.frames_in_flight()) {
    create_sync_objects();
    create_command_buffers();
    create_geometry_buffers();
    last_extent_ = swapchain.extent();
}

FrameRenderer::~FrameRenderer() {
    ctx_.device().waitIdle();
}

void FrameRenderer::bind_frame_graph_passes() {
    frame_graph_.set_pass_execute("geometry", [this]() {
        record_scene_pass(active_sync_index_, pending_time_, pending_state_);
    });
    frame_graph_.set_pass_execute("present", [this]() {
        record_present_pass(active_sync_index_, active_image_index_,
                           pending_time_, pending_state_);
    });
}

bool FrameRenderer::needs_resize(vk::Extent2D extent) const {
    return extent.width != last_extent_.width || extent.height != last_extent_.height;
}

void FrameRenderer::on_resize(vk::Extent2D extent) {
    last_extent_ = extent;
}

void FrameRenderer::create_sync_objects() {
    vk::SemaphoreCreateInfo sem_info;
    vk::FenceCreateInfo fence_info;
    fence_info.flags = vk::FenceCreateFlagBits::eSignaled;

    for (std::uint32_t i = 0; i < frames_in_flight_; ++i) {
        image_available_.push_back(ctx_.device().createSemaphore(sem_info));
        render_finished_.push_back(ctx_.device().createSemaphore(sem_info));
        in_flight_.push_back(ctx_.device().createFence(fence_info));
    }
}

void FrameRenderer::create_command_buffers() {
    vk::CommandBufferAllocateInfo alloc;
    alloc.commandPool        = *ctx_.command_pool();
    alloc.level              = vk::CommandBufferLevel::ePrimary;
    alloc.commandBufferCount = frames_in_flight_;
    command_buffers_ = ctx_.device().allocateCommandBuffers(alloc);
}

void FrameRenderer::create_geometry_buffers() {
    std::vector<RetroVertex4D> verts = {
        {{-0.6f, -0.3f, 0, 0}, {0, 1}, 0xFF00FFFF},
        {{ 0.0f,  0.3f, 0, 0}, {0.5f, 0}, 0xFFFF00FF},
        {{ 0.6f, -0.3f, 0, 0}, {1, 1}, 0xFF00FFFF},
        {{-0.3f, 0.5f, 0, 0.5f}, {0, 1}, 0xFFFF44FF},
        {{ 0.3f, 0.5f, 0, 0.5f}, {1, 0}, 0xFFFF44FF},
        {{ 0.0f, 0.9f, 0, 1.0f}, {0.5f, 0.5f}, 0xFFFFFFFF},
    };
    vertex_count_ = static_cast<std::uint32_t>(verts.size());

    vk::DeviceSize size = sizeof(RetroVertex4D) * verts.size();
    vk::BufferCreateInfo buf_info;
    buf_info.size  = size;
    buf_info.usage = vk::BufferUsageFlagBits::eVertexBuffer;
    vertex_buffer_ = ctx_.device().createBuffer(buf_info);

    auto mem_req = vertex_buffer_.getMemoryRequirements();
    auto mem_props = ctx_.physical_device().getMemoryProperties();

    std::uint32_t mem_type = UINT32_MAX;
    for (std::uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((mem_req.memoryTypeBits & (1u << i))
            && (mem_props.memoryTypes[i].propertyFlags
                & vk::MemoryPropertyFlagBits::eHostVisible)) {
            mem_type = i;
            break;
        }
    }
    if (mem_type == UINT32_MAX) throw std::runtime_error("No host-visible memory");

    vk::MemoryAllocateInfo alloc_info;
    alloc_info.allocationSize  = mem_req.size;
    alloc_info.memoryTypeIndex = mem_type;
    vertex_memory_ = ctx_.device().allocateMemory(alloc_info);
    vertex_buffer_.bindMemory(*vertex_memory_, 0);

    void* mapped = vertex_memory_.mapMemory(0, size);
    std::memcpy(mapped, verts.data(), static_cast<std::size_t>(size));
    vertex_memory_.unmapMemory();
}

void FrameRenderer::record_scene_pass(std::uint32_t sync_index, float time,
                                      const RetroPipelineState& state) {
    auto& cmd = command_buffers_[sync_index];
    auto extent = pipelines_.scene_targets().extent();
    auto& targets = pipelines_.scene_targets().frame(sync_index);

    DynamicRenderer::transition_color_to_attachment(cmd, targets.color.image());
    DynamicRenderer::transition_depth_to_attachment(cmd, targets.depth.image());

    DynamicRenderTarget target;
    target.color_view   = targets.color.view();
    target.depth_view   = targets.depth.view();
    target.extent       = extent;
    target.color_format = SceneTargetSet::kSceneColorFormat;
    target.depth_format = SceneTargetSet::kSceneDepthFormat;
    target.has_depth    = true;

    ClearValues clear;
    clear.color = {0.02f, 0.02f, 0.06f, 1.0f};
    DynamicRenderer::begin(cmd, target, clear);

    vk::Viewport viewport{0, 0, static_cast<float>(extent.width),
                          static_cast<float>(extent.height), 0, 1};
    vk::Rect2D scissor{{0, 0}, extent};
    cmd.setViewport(0, viewport);
    cmd.setScissor(0, scissor);

    auto pc = pipelines_.make_push_constants(state, time, extent);

    pipelines_.bind_scene(cmd, RetroPipelineKind::RetroComposite);
    pipelines_.push_scene_state(cmd, RetroPipelineKind::RetroComposite, pc);
    cmd.draw(3, 1, 0, 0);

    pipelines_.bind_scene(cmd, RetroPipelineKind::Sprite4D);
    pipelines_.push_scene_state(cmd, RetroPipelineKind::Sprite4D, pc);
    vk::DeviceSize offset = 0;
    cmd.bindVertexBuffers(0, *vertex_buffer_, offset);
    cmd.draw(vertex_count_, 1, 0, 0);

    if (state.style == RetroStyle::GeometryWars || state.w_morph > 0.3f) {
        pipelines_.bind_scene(cmd, RetroPipelineKind::VectorGlow);
        pipelines_.push_scene_state(cmd, RetroPipelineKind::VectorGlow, pc);
        cmd.draw(3, 1, 0, 0);
    }

    DynamicRenderer::end(cmd);

    DynamicRenderer::transition_color_to_shader_read(cmd, targets.color.image());
    DynamicRenderer::transition_depth_to_shader_read(cmd, targets.depth.image());

    post_.update_descriptors(sync_index,
        targets.color.view(), targets.color.format(),
        targets.depth.view());
}

void FrameRenderer::record_present_pass(std::uint32_t sync_index,
                                        std::uint32_t image_index,
                                        float time, const RetroPipelineState& state) {
    auto& cmd = command_buffers_[sync_index];
    auto extent = swapchain_.extent();

    auto swapchain_image = swapchain_.image(image_index);
    DynamicRenderer::transition_color_to_attachment(cmd, swapchain_image);

    TonemapPushConstants tonemap;
    tonemap.exposure          = 1.2f;
    tonemap.scanline_strength = state.scanline_strength;
    tonemap.w_morph           = state.w_morph;
    tonemap.time              = time;

    post_.record_present(cmd, sync_index,
        *swapchain_.image_views()[image_index], extent, tonemap);

    DynamicRenderer::transition_to_present(cmd, swapchain_image);
}

void FrameRenderer::render_frame(float time, const RetroPipelineState& state) {
    const std::uint32_t sync_index = sync_index_;
    auto& fence = in_flight_[sync_index];
    auto& cmd   = command_buffers_[sync_index];

    (void)ctx_.device().waitForFences(*fence, VK_TRUE, UINT64_MAX);
    ctx_.device().resetFences(*fence);

    auto [result, image_index] = swapchain_.handle().acquireNextImage(
        UINT64_MAX, *image_available_[sync_index], nullptr);

    if (result == vk::Result::eErrorOutOfDateKHR) return;

    pending_time_        = time;
    pending_state_       = state;
    active_sync_index_   = sync_index;
    active_image_index_  = image_index;

    cmd.reset();
    cmd.begin(vk::CommandBufferBeginInfo{});
    frame_graph_.execute();
    cmd.end();

    vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    vk::SubmitInfo submit;
    submit.waitSemaphoreCount   = 1;
    submit.pWaitSemaphores      = &*image_available_[sync_index];
    submit.pWaitDstStageMask    = &wait_stage;
    submit.commandBufferCount   = 1;
    submit.pCommandBuffers      = &*cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores    = &*render_finished_[sync_index];

    ctx_.graphics_queue().submit(submit, *fence);

    vk::PresentInfoKHR present;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores    = &*render_finished_[sync_index];
    present.swapchainCount     = 1;
    auto sc = static_cast<vk::SwapchainKHR>(*swapchain_.handle());
    present.pSwapchains        = &sc;
    present.pImageIndices      = &image_index;
    (void)ctx_.present_queue().presentKHR(present);

    sync_index_ = (sync_index + 1) % frames_in_flight_;
}

} // namespace kengine