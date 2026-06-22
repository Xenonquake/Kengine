#include "kengine/render/frame_renderer.hpp"
#include "kengine/vulkan/dynamic_renderer.hpp"
#include <cmath>
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
        record_scene_pass(active_sync_index_, pending_time_, pending_state_, pending_cam_);
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
    // Recreate per-swapchain-image sync objects in case image count changed
    recreate_swapchain_sync();
}

void FrameRenderer::recreate_swapchain_sync() {
    // Wait to be safe (caller usually does device waitIdle)
    render_finished_semaphores_.clear();
    present_fences_.clear();

    swapchain_image_count_ = swapchain_.image_count();

    vk::SemaphoreCreateInfo sem_info;
    for (std::uint32_t i = 0; i < swapchain_image_count_; ++i) {
        render_finished_semaphores_.push_back(ctx_.device().createSemaphore(sem_info));

        vk::FenceCreateInfo fence_info;
        fence_info.flags = vk::FenceCreateFlagBits::eSignaled;
        present_fences_.push_back(ctx_.device().createFence(fence_info));
    }
}

void FrameRenderer::create_sync_objects() {
    vk::SemaphoreCreateInfo sem_info;
    vk::FenceCreateInfo fence_info;
    fence_info.flags = vk::FenceCreateFlagBits::eSignaled;

    // Per frame-in-flight for acquire + command buffer lifetime
    for (std::uint32_t i = 0; i < frames_in_flight_; ++i) {
        acquire_semaphores_.push_back(ctx_.device().createSemaphore(sem_info));
        in_flight_.push_back(ctx_.device().createFence(fence_info));
    }

    // Per swapchain image (for present side + maintenance1)
    swapchain_image_count_ = swapchain_.image_count();
    for (std::uint32_t i = 0; i < swapchain_image_count_; ++i) {
        render_finished_semaphores_.push_back(ctx_.device().createSemaphore(sem_info));

        vk::FenceCreateInfo present_fence_info;
        present_fence_info.flags = vk::FenceCreateFlagBits::eSignaled;
        present_fences_.push_back(ctx_.device().createFence(present_fence_info));
    }
}

void FrameRenderer::create_command_buffers() {
    vk::CommandBufferAllocateInfo alloc;
    alloc.commandPool        = *ctx_.command_pool();
    alloc.level              = vk::CommandBufferLevel::ePrimary;
    alloc.commandBufferCount = frames_in_flight_;
    command_buffers_ = ctx_.device().allocateCommandBuffers(alloc);
}

static std::uint32_t pick_host_visible_mem_type(const vk::raii::PhysicalDevice& phys, std::uint32_t bits) {
    auto mem_props = phys.getMemoryProperties();
    for (std::uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((bits & (1u << i))
            && (mem_props.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)) {
            return i;
        }
    }
    return UINT32_MAX;
}

static void upload_host_buffer(vk::raii::DeviceMemory& mem,
                               const void* data,
                               vk::DeviceSize size) {
    void* mapped = mem.mapMemory(0, size);
    std::memcpy(mapped, data, static_cast<std::size_t>(size));
    mem.unmapMemory();
}

static vk::raii::Buffer create_vb(const vk::raii::Device& dev, vk::DeviceSize size) {
    vk::BufferCreateInfo b;
    b.size  = size;
    b.usage = vk::BufferUsageFlagBits::eVertexBuffer;
    return dev.createBuffer(b);
}

static vk::raii::DeviceMemory alloc_and_bind_host(const vk::raii::Device& dev,
                                                  const vk::raii::PhysicalDevice& phys,
                                                  vk::raii::Buffer& buf) {
    auto mr = buf.getMemoryRequirements();
    std::uint32_t mt = pick_host_visible_mem_type(phys, mr.memoryTypeBits);
    if (mt == UINT32_MAX) throw std::runtime_error("No host-visible memory");
    vk::MemoryAllocateInfo ai;
    ai.allocationSize  = mr.size;
    ai.memoryTypeIndex = mt;
    auto memory = dev.allocateMemory(ai);
    buf.bindMemory(*memory, 0);
    return memory;
}

static std::vector<RetroVertex4D> make_sprite_quad(float x, float y, float z, float w,
                                                   float sx, float sy, std::uint32_t color) {
    // tiny quad as two tris, uv for the sprite shader
    std::vector<RetroVertex4D> q(6);
    q[0] = {{-sx + x, -sy + y, z, w}, {0, 1}, color};
    q[1] = {{ sx + x, -sy + y, z, w}, {1, 1}, color};
    q[2] = {{ sx + x,  sy + y, z, w}, {1, 0}, color};
    q[3] = {{-sx + x, -sy + y, z, w}, {0, 1}, color};
    q[4] = {{ sx + x,  sy + y, z, w}, {1, 0}, color};
    q[5] = {{-sx + x,  sy + y, z, w}, {0, 0}, color};
    return q;
}

void FrameRenderer::create_geometry_buffers() {
    // --- Stars and rocks as Sprite4D (small quads in 4D) ---
    std::vector<RetroVertex4D> sprite_verts;
    // Stars (background, small, various w)
    for (int i = 0; i < 18; ++i) {
        float t = i * 1.7f;
        float x = sinf(t * 0.7f) * (1.8f + (i % 3) * 0.6f);
        float y = cosf(t * 1.1f) * (1.0f + (i % 2) * 0.8f);
        float z = -1.5f - (i % 5) * 0.8f;
        float w = sinf(t * 0.4f) * 1.2f;
        float s = 0.018f + (i % 4) * 0.004f;
        std::uint32_t col = (i % 3 == 0) ? 0xFFFFFF88u : 0xFFCCEEFFu;
        auto q = make_sprite_quad(x, y, z, w, s, s, col);
        sprite_verts.insert(sprite_verts.end(), q.begin(), q.end());
    }
    // Rocks (mid/fg, chunkier)
    for (int i = 0; i < 7; ++i) {
        float t = i * 2.3f + 0.5f;
        float x = cosf(t) * (2.2f - i * 0.15f);
        float y = sinf(t * 0.8f) * 1.6f - 0.4f;
        float z = 0.2f + (i % 3) * 0.6f;
        float w = cosf(t * 0.9f) * 0.8f - 0.3f;
        float s = 0.09f + ((i + 2) % 3) * 0.025f;
        std::uint32_t col = (i % 2 == 0) ? 0xFF88AACCu : 0xFF66BB99u;
        auto q = make_sprite_quad(x, y, z, w, s, s * 0.7f, col);
        sprite_verts.insert(sprite_verts.end(), q.begin(), q.end());
    }

    sprite_vertex_count_ = static_cast<std::uint32_t>(sprite_verts.size());
    vk::DeviceSize sprite_size = sizeof(RetroVertex4D) * sprite_verts.size();

    sprite_vertex_buffer_ = create_vb(ctx_.device(), sprite_size);
    sprite_vertex_memory_ = alloc_and_bind_host(ctx_.device(), ctx_.physical_device(), sprite_vertex_buffer_);
    upload_host_buffer(sprite_vertex_memory_, sprite_verts.data(), sprite_size);

    // --- Vector ship (LineStrip via VectorGlow) base shape (will be transformed each frame) ---
    // Simple classic vector ship pointing up, centered near origin
    ship_base_shape_ = {
        {{ 0.00f,  0.18f, 0.02f, 0.05f}, {0.5f, 0.0f}, 0xFFFFFFFFu}, // nose
        {{-0.12f, -0.08f, 0.01f, 0.04f}, {0.0f, 1.0f}, 0xFFEEFFFFu}, // left wing
        {{-0.04f, -0.02f, 0.00f, 0.03f}, {0.25f, 0.6f}, 0xFFCCEEFFu},
        {{ 0.04f, -0.02f, 0.00f, 0.03f}, {0.75f, 0.6f}, 0xFFCCEEFFu},
        {{ 0.12f, -0.08f, 0.01f, 0.04f}, {1.0f, 1.0f}, 0xFFEEFFFFu}, // right wing
        {{ 0.00f,  0.18f, 0.02f, 0.05f}, {0.5f, 0.0f}, 0xFFFFFFFFu}, // close nose
    };
    ship_vertex_count_ = static_cast<std::uint32_t>(ship_base_shape_.size());

    vk::DeviceSize ship_size = sizeof(RetroVertex4D) * ship_base_shape_.size();
    ship_vertex_buffer_ = create_vb(ctx_.device(), ship_size);
    ship_vertex_memory_ = alloc_and_bind_host(ctx_.device(), ctx_.physical_device(), ship_vertex_buffer_);
    // initial upload (will be overwritten on first update)
    upload_host_buffer(ship_vertex_memory_, ship_base_shape_.data(), ship_size);
}

static RetroVertex4D transform_ship_vert(const RetroVertex4D& base, float cx, float cy, float cz, float cw,
                                         float yaw, float pitch_w) {
    // very simple 2D-ish yaw in xy + slight w modulation
    float c = cosf(yaw), s = sinf(yaw);
    float x = base.pos[0] * c - base.pos[1] * s;
    float y = base.pos[0] * s + base.pos[1] * c;
    float z = base.pos[2] + (base.pos[1] * 0.15f) * sinf(pitch_w);
    float w = base.pos[3] + cw + cosf(pitch_w) * 0.03f;
    RetroVertex4D v = base;
    v.pos[0] = x + cx;
    v.pos[1] = y + cy;
    v.pos[2] = z + cz;
    v.pos[3] = w;
    return v;
}

void FrameRenderer::update_ship_geometry(float time) {
    if (ship_base_shape_.empty() || ship_vertex_count_ == 0) return;

    // Zig zag ship in foreground: figure-8-ish path near viewer
    float phase = time * 1.6f;
    float zig_x = sinf(phase) * 1.6f + sinf(phase * 2.3f) * 0.4f;
    float zig_y = cosf(phase * 0.7f) * 0.9f - 0.3f;
    float zig_z = 2.4f + sinf(phase * 1.1f) * 0.25f; // closer
    float zig_w = sinf(phase * 0.4f) * 0.35f + 0.05f;

    float yaw   = sinf(phase * 1.3f) * 0.6f + sinf(phase) * 0.2f;
    float pw    = sinf(phase * 2.0f) * 1.2f;

    std::vector<RetroVertex4D> live(ship_base_shape_.size());
    for (size_t i = 0; i < live.size(); ++i) {
        live[i] = transform_ship_vert(ship_base_shape_[i], zig_x, zig_y, zig_z, zig_w, yaw, pw);
    }

    vk::DeviceSize sz = sizeof(RetroVertex4D) * live.size();
    upload_host_buffer(ship_vertex_memory_, live.data(), sz);
    ship_vertex_count_ = static_cast<std::uint32_t>(live.size());
}

void FrameRenderer::record_scene_pass(std::uint32_t sync_index, float time,
                                      const RetroPipelineState& state, const Camera4D& cam) {
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
    clear.color = {0.0f, 0.0f, 0.0f, 1.0f};  // solid black for Space Arcade background
    DynamicRenderer::begin(cmd, target, clear);

    vk::Viewport viewport{0, 0, static_cast<float>(extent.width),
                          static_cast<float>(extent.height), 0, 1};
    vk::Rect2D scissor{{0, 0}, extent};
    cmd.setViewport(0, viewport);
    cmd.setScissor(0, scissor);

    auto pc = pipelines_.make_push_constants(state, cam, time, extent);

    pipelines_.bind_scene(cmd, RetroPipelineKind::RetroComposite);
    pipelines_.push_scene_state(cmd, RetroPipelineKind::RetroComposite, pc);
    cmd.draw(3, 1, 0, 0);

    // Update animated ship verts (zig zagging foreground vector ship)
    update_ship_geometry(time);

    // Stars + rocks
    pipelines_.bind_scene(cmd, RetroPipelineKind::Sprite4D);
    pipelines_.push_scene_state(cmd, RetroPipelineKind::Sprite4D, pc);
    vk::DeviceSize off = 0;
    cmd.bindVertexBuffers(0, *sprite_vertex_buffer_, off);
    cmd.draw(sprite_vertex_count_, 1, 0, 0);

    // Vector line ship
    if (state.style == RetroStyle::GeometryWars || state.w_morph > 0.1f) {
        pipelines_.bind_scene(cmd, RetroPipelineKind::VectorGlow);
        pipelines_.push_scene_state(cmd, RetroPipelineKind::VectorGlow, pc);
        cmd.bindVertexBuffers(0, *ship_vertex_buffer_, off);
        cmd.draw(ship_vertex_count_, 1, 0, 0);
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

    auto tonemap = post_.make_test_tonemap_constants(time, state.w_morph, state.scanline_strength);

    post_.record_present(cmd, sync_index,
        *swapchain_.image_views()[image_index], extent, tonemap);

    DynamicRenderer::transition_to_present(cmd, swapchain_image);
}

void FrameRenderer::render_frame(float time, const RetroPipelineState& state, const Camera4D& cam) {
    const std::uint32_t frame = sync_index_;
    auto& fence = in_flight_[frame];
    auto& cmd   = command_buffers_[frame];

    (void)ctx_.device().waitForFences(*fence, VK_TRUE, UINT64_MAX);
    ctx_.device().resetFences(*fence);

    // Acquire using per-frame acquire semaphore
    auto [result, image_index] = swapchain_.handle().acquireNextImage(
        UINT64_MAX, *acquire_semaphores_[frame], nullptr);

    if (result == vk::Result::eErrorOutOfDateKHR) return;

    pending_time_        = time;
    pending_state_       = state;
    pending_cam_         = cam;
    active_sync_index_   = frame;
    active_image_index_  = image_index;

    cmd.reset();
    cmd.begin(vk::CommandBufferBeginInfo{});
    frame_graph_.execute();
    cmd.end();

    vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    vk::SubmitInfo submit;
    submit.waitSemaphoreCount   = 1;
    submit.pWaitSemaphores      = &*acquire_semaphores_[frame];
    submit.pWaitDstStageMask    = &wait_stage;
    submit.commandBufferCount   = 1;
    submit.pCommandBuffers      = &*cmd;
    // Signal the *per-image* render finished semaphore
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores    = &*render_finished_semaphores_[image_index];

    ctx_.graphics_queue().submit(submit, *fence);

    // Present: wait on the per-image semaphore.
    // (When VK_KHR_swapchain_maintenance1 is enabled and supported we can also attach
    //  VkSwapchainPresentFenceInfoKHR with a per-image fence for cleaner present sync.)
    vk::PresentInfoKHR present;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores    = &*render_finished_semaphores_[image_index];
    present.swapchainCount     = 1;
    auto sc = static_cast<vk::SwapchainKHR>(*swapchain_.handle());
    present.pSwapchains        = &sc;
    present.pImageIndices      = &image_index;

    (void)ctx_.present_queue().presentKHR(present);

    sync_index_ = (frame + 1) % frames_in_flight_;
}

} // namespace kengine