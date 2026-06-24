#include "kengine/render/frame_renderer.hpp"
#include "kengine/render/texture.hpp"
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
    post_.on_resize(last_extent_);
}

FrameRenderer::~FrameRenderer() {
    ctx_.device().waitIdle();
}

void FrameRenderer::bind_frame_graph_passes() {
    frame_graph_.set_pass_execute("geometry", [this]() {
        record_scene_pass(active_sync_index_, pending_time_, pending_state_, pending_cam_);
    });

    // Bloom now executed manually inside geometry pass for correct command buffer state.
    // (graph passes left for future when full post recording is restructured)
    // frame_graph_.set_pass_execute("bloom_threshold", ...);
    // frame_graph_.set_pass_execute("bloom_blur", ...);

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
    post_.on_resize(extent);
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
                                                   float sx, float sy, std::uint32_t color, std::uint32_t texIndex = 0,
                                                   float vx = 0.0f, float vy = 0.0f, float vz = 0.0f) {
    // tiny quad as two tris, uv for the sprite shader
    std::vector<RetroVertex4D> q(6);
    q[0] = {{-sx + x, -sy + y, z, w}, {vx, vy, vz}, {0, 1}, color, texIndex};
    q[1] = {{ sx + x, -sy + y, z, w}, {vx, vy, vz}, {1, 1}, color, texIndex};
    q[2] = {{ sx + x,  sy + y, z, w}, {vx, vy, vz}, {0, 0}, color, texIndex};
    q[3] = {{-sx + x, -sy + y, z, w}, {vx, vy, vz}, {0, 1}, color, texIndex};
    q[4] = {{ sx + x,  sy + y, z, w}, {vx, vy, vz}, {1, 0}, color, texIndex};
    q[5] = {{-sx + x,  sy + y, z, w}, {vx, vy, vz}, {0, 0}, color, texIndex};
    return q;
}

void FrameRenderer::create_geometry_buffers() {
    // --- Moving 2D stars for flying-through-space effect (small sprites coming towards viewer, using 4D flare via ParticleAdditive) ---
    moving_stars_.clear();
    for (int i = 0; i < 60; ++i) {
        float r1 = (i * 0.37f);
        float r2 = (i * 0.73f);
        MovingStar s;
        s.base_x = sinf(r1) * (0.9f + (i % 5) * 0.1f);
        s.base_y = cosf(r2) * 0.9f;
        s.depth = -2.5f - (i % 7) * 0.3f; // far negative
        s.speed = 1.8f + (i % 4) * 0.3f;
        s.size = 0.012f + (i % 3) * 0.003f;
        s.color = 0xFFFFFFFFu;  // pure white (no neon/purple tint from palette)
        moving_stars_.push_back(s);
    }

    // initial build of sprite buffer (will be updated every frame)
    std::vector<RetroVertex4D> initial_sprites;
    for (auto& s : moving_stars_) {
        float z = s.depth;
        // forward vel for motion effects
        auto q = make_sprite_quad(s.base_x, s.base_y, z, 0.0f, s.size, s.size, s.color, 1, 0.f, 0.f, s.speed);
        initial_sprites.insert(initial_sprites.end(), q.begin(), q.end());
    }
    // room for exhaust + bullets (generous for Phase 1)
    initial_sprites.resize(initial_sprites.size() + 120);

    sprite_vertex_count_ = static_cast<std::uint32_t>(initial_sprites.size());
    vk::DeviceSize sprite_size = sizeof(RetroVertex4D) * initial_sprites.size();

    sprite_vertex_buffer_ = create_vb(ctx_.device(), sprite_size);
    sprite_vertex_memory_ = alloc_and_bind_host(ctx_.device(), ctx_.physical_device(), sprite_vertex_buffer_);
    upload_host_buffer(sprite_vertex_memory_, initial_sprites.data(), sprite_size);

    // 3D edgy green neon angular retro spaceship is built procedurally in update_ship_geometry
    // (wireframe lines using eLineList topology for angular look)
    ship_vertex_count_ = 64;
    vk::DeviceSize ship_size = sizeof(RetroVertex4D) * ship_vertex_count_;
    ship_vertex_buffer_ = create_vb(ctx_.device(), ship_size);
    ship_vertex_memory_ = alloc_and_bind_host(ctx_.device(), ctx_.physical_device(), ship_vertex_buffer_);

    // Demo texture for bindless (small white 2x2 to test sampling on stars)
    {
        uint8_t whiteData[4 * 4] = {
            255,255,255,255,  255,255,255,255,
            255,255,255,255,  255,255,255,255
        };
        demoTexture_.emplace(ctx_.device(), ctx_.allocator().handle(),
                             ctx_.graphics_queue(), ctx_.command_pool(),
                             whiteData, 2, 2);
    }
    // Register demo texture at bindless index 1
    if (demoTexture_) {
        auto& rs = pipelines_.scene();
        rs.updateTexture(1, demoTexture_->view(), demoTexture_->sampler());
    }
}

void FrameRenderer::update_background_flow(float ship_vx, float ship_vy, float ship_vz, float dt) {
    float responsiveness = 0.65f;
    background_flow_x_ = background_flow_x_ * (1.0f - responsiveness) + (-ship_vx * 0.85f) * responsiveness;
    background_flow_y_ = background_flow_y_ * (1.0f - responsiveness) + (-ship_vy * 0.65f) * responsiveness;
    background_flow_z_ = 1.0f + ship_vz * 0.45f;
}

void FrameRenderer::update_ship_geometry(float time) {
    // Build edgy 3D green neon angular forward-pointed retro spaceship (wireframe)
    std::vector<RetroVertex4D> live;

    float vx = ship_vel_x_, vy = ship_vel_y_, vz = ship_vel_z_;
    float flen = sqrtf(vx*vx + vy*vy + vz*vz);
    float fx = (flen > 0.01f) ? vx / flen : 0.0f;
    float fy = (flen > 0.01f) ? vy / flen : -1.0f;
    float fz = (flen > 0.01f) ? vz / flen : 0.0f;

    // orthonormal basis (local +z = forward)
    float rx = fy * 0.0f - fz * 1.0f;
    float ry = fz * 0.0f - fx * 0.0f;
    float rz = fx * 1.0f - fy * 0.0f;
    float rlen = sqrtf(rx*rx + ry*ry + rz*rz);
    if (rlen > 0.001f) { rx /= rlen; ry /= rlen; rz /= rlen; } else { rx=1.0f; ry=0; rz=0; }

    float ux = ry * fz - rz * fy;
    float uy = rz * fx - rx * fz;
    float uz = rx * fy - ry * fx;

    float model_scale = 2.0f;  // make the 3D ship larger and more visible

    auto add_line = [&](float lx1, float ly1, float lz1, float lx2, float ly2, float lz2, uint32_t col) {
        lx1 *= model_scale; ly1 *= model_scale; lz1 *= model_scale;
        lx2 *= model_scale; ly2 *= model_scale; lz2 *= model_scale;
        float wx1 = lx1*rx + ly1*ux + lz1*fx + ship_x_;
        float wy1 = lx1*ry + ly1*uy + lz1*fy + ship_y_;
        float wz1 = lx1*rz + ly1*uz + lz1*fz + ship_z_;
        float ww1 = ship_w_;

        float wx2 = lx2*rx + ly2*ux + lz2*fx + ship_x_;
        float wy2 = lx2*ry + ly2*uy + lz2*fy + ship_y_;
        float wz2 = lx2*rz + ly2*uz + lz2*fz + ship_z_;
        float ww2 = ship_w_;

        live.push_back({{wx1, wy1, wz1, ww1}, {vx, vy, vz}, {0.0f, 0.0f}, col, 0});
        live.push_back({{wx2, wy2, wz2, ww2}, {vx, vy, vz}, {0.0f, 1.0f}, col, 0});
    };

    const uint32_t neon_green = 0xFF00FF88;
    const uint32_t dark_green = 0xFF008844;
    const uint32_t accent    = 0xFF44FFAA;

    // Angular forward pointed retro 3D ship (green neon wireframe)
    // local: +z forward, +y up, +x right
    // Nose
    add_line( 0.000f,  0.025f,  0.26f,   -0.070f,  0.010f,  0.12f , neon_green);
    add_line( 0.000f,  0.025f,  0.26f,    0.070f,  0.010f,  0.12f , neon_green);
    // Left wing
    add_line(-0.070f,  0.010f,  0.12f ,  -0.220f,  0.000f,  0.03f , neon_green);
    add_line(-0.220f,  0.000f,  0.03f ,  -0.090f,  0.015f, -0.09f , dark_green);
    // Right wing
    add_line( 0.070f,  0.010f,  0.12f ,   0.220f,  0.000f,  0.03f , neon_green);
    add_line( 0.220f,  0.000f,  0.03f ,   0.090f,  0.015f, -0.09f , dark_green);
    // Fuselage spine
    add_line( 0.000f,  0.025f,  0.26f ,   0.000f,  0.040f, -0.05f , accent);
    add_line( 0.000f,  0.040f, -0.05f ,   0.000f,  0.030f, -0.18f , neon_green);
    // Cockpit / angular top
    add_line(-0.030f,  0.035f,  0.10f ,   0.030f,  0.035f,  0.10f , dark_green);
    add_line(-0.030f,  0.035f,  0.10f ,   0.000f,  0.055f,  0.02f , accent);
    add_line( 0.030f,  0.035f,  0.10f ,   0.000f,  0.055f,  0.02f , accent);
    // Vertical stabilizer (edgy fin)
    add_line( 0.000f,  0.030f, -0.05f ,   0.000f,  0.130f, -0.10f , neon_green);
    add_line( 0.000f,  0.130f, -0.10f ,   0.000f,  0.045f, -0.20f , dark_green);
    // Lower body details for 3D volume
    add_line(-0.040f,  0.005f,  0.08f ,   0.040f,  0.005f,  0.08f , dark_green);
    add_line(-0.040f,  0.005f,  0.08f ,  -0.060f, -0.010f, -0.06f , neon_green);
    add_line( 0.040f,  0.005f,  0.08f ,   0.060f, -0.010f, -0.06f , neon_green);

    ship_vertex_count_ = static_cast<std::uint32_t>(live.size());
    vk::DeviceSize sz = sizeof(RetroVertex4D) * live.size();
    upload_host_buffer(ship_vertex_memory_, live.data(), sz);
}

void FrameRenderer::update_moving_stars(float dt) {
    if (moving_stars_.empty()) return;

    // Flow members are updated from engine via update_background_flow (ship-relative)

    std::vector<RetroVertex4D> sprite_verts;

    for (auto& s : moving_stars_) {
        // Apply ship-relative flow + depth (parallax)
        float effective_speed = s.speed * background_flow_z_;
        s.depth += effective_speed * dt;

        // Lateral streaming (stronger parallax on closer stars)
        s.base_x += background_flow_x_ * dt * (1.6f + (s.depth * 0.35f));
        s.base_y += background_flow_y_ * dt * (1.3f + (s.depth * 0.25f));

        if (s.depth > 1.8f) {
            // respawn far away, random lateral position
            s.depth = -2.8f - (rand() % 5) * 0.15f;
            s.base_x = ((rand() % 2000) / 1000.0f - 1.0f) * 0.95f;
            s.base_y = ((rand() % 2000) / 1000.0f - 1.0f) * 0.7f;

            // Inherit a bit of current flow for continuity
            s.base_x += background_flow_x_ * 0.35f;
            s.base_y += background_flow_y_ * 0.35f;
        }

        float sz = s.size * (1.0f + (s.depth + 2.8f) * 0.4f); // grow as it approaches

        // Pass full velocity (for shader glow trails / motion blur)
        float vx = background_flow_x_ * 0.75f;
        float vy = background_flow_y_ * 0.75f;
        float vz = effective_speed;

        auto q = make_sprite_quad(s.base_x, s.base_y, s.depth, 0.0f, sz, sz, s.color, 1, vx, vy, vz);
        sprite_verts.insert(sprite_verts.end(), q.begin(), q.end());
    }

    // === IMPROVED EXHAUST — tied to actual ship thrust + lateral velocity ===
    float thrust = std::max(0.0f, ship_vel_z_ * 0.9f + 0.35f);
    float exhaust_y = ship_y_ - 0.18f;
    for (int e = 0; e < 5; ++e) {
        float ex = ship_x_ + (e - 2) * 0.015f * (1.0f + e * 0.2f) + background_flow_x_ * 0.025f;
        float ey = exhaust_y - e * 0.035f;
        float ez = ship_z_ - e * 0.02f;
        float es = (0.022f - e * 0.002f) * (0.65f + thrust * 0.9f);
        if (es < 0.005f) es = 0.005f;

        std::uint32_t ecol = (e < 2) ? 0xFFFFFFFFu : 0xFFEEFFFFu;

        // Exhaust velocity reacts to ship's lateral + forward thrust
        float evx = background_flow_x_ * 0.45f;
        float evy = -0.9f + background_flow_y_ * 0.35f;
        float evz = -thrust * 0.7f;

        auto q = make_sprite_quad(ex, ey, ez, 0.0f, es, es * 0.7f, ecol, 1, evx, evy, evz);
        sprite_verts.insert(sprite_verts.end(), q.begin(), q.end());
    }

    // === BULLETS (small fast glowing quads with velocity for shader trails) ===
    for (const auto& b : bullets_) {
        float bs = b.size * 0.9f;
        auto q = make_sprite_quad(b.x, b.y, b.z, b.w, bs, bs * 0.35f, b.color, 1,
                                  b.vx, b.vy, b.vz);
        sprite_verts.insert(sprite_verts.end(), q.begin(), q.end());
    }

    // pad if needed
    while (sprite_verts.size() < sprite_vertex_count_) {
        sprite_verts.push_back({{0,0,0,0}, {0,0,0}, {0,0}, 0, 0});
    }
    if (sprite_verts.size() > sprite_vertex_count_) sprite_verts.resize(sprite_vertex_count_);

    current_sprite_draw_count_ = static_cast<std::uint32_t>(sprite_verts.size());

    vk::DeviceSize sz = sizeof(RetroVertex4D) * sprite_verts.size();
    upload_host_buffer(sprite_vertex_memory_, sprite_verts.data(), sz);
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

    // Update moving stars (coming towards perspective) and ship trail
    static float last_t = time;
    float dt = time - last_t;
    if (dt <= 0.0f || dt > 0.1f) dt = 0.016f;
    last_t = time;
    update_moving_stars(dt);

    // Update ship position (Space Invaders bottom style)
    update_ship_geometry(time);

    // Stars + exhaust (white + 4D radial flare/glow from particle shader; post-bloom will further enhance trail)
    pipelines_.bind_scene(cmd, RetroPipelineKind::ParticleAdditive);
    pipelines_.push_scene_state(cmd, RetroPipelineKind::ParticleAdditive, pc);

    // Bind bindless texture set if available (for future textured sprites/atlases)
    auto& retroScene = pipelines_.scene();
    if (auto* ds = retroScene.bindlessSet()) {
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                               retroScene.layout(RetroPipelineKind::ParticleAdditive),
                               0, {**ds}, {});
    }

    vk::DeviceSize off = 0;
    cmd.bindVertexBuffers(0, *sprite_vertex_buffer_, off);
    cmd.draw(current_sprite_draw_count_ > 0 ? current_sprite_draw_count_ : sprite_vertex_count_, 1, 0, 0);

    // Vector line ship (no rocks)
    pipelines_.bind_scene(cmd, RetroPipelineKind::VectorGlow);
    pipelines_.push_scene_state(cmd, RetroPipelineKind::VectorGlow, pc);
    cmd.bindVertexBuffers(0, *ship_vertex_buffer_, off);
    cmd.draw(ship_vertex_count_, 1, 0, 0);

    DynamicRenderer::end(cmd);

    // Feed the rendered scene into bloom input for post processing passes
    if (post_.bloom_input().valid()) {
        // Transition scene to transfer src for the copy (bloom to dst)
        {
            vk::ImageMemoryBarrier2 bar{};
            bar.image = targets.color.image();
            bar.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
            bar.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
            bar.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
            bar.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
            bar.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
            bar.newLayout = vk::ImageLayout::eTransferSrcOptimal;
            bar.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            vk::DependencyInfo dep{};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &bar;
            cmd.pipelineBarrier2(dep);
        }

        // bloom input to transfer dst
        {
            vk::ImageMemoryBarrier2 bar{};
            bar.image = post_.bloom_input().image();
            bar.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
            bar.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
            bar.srcAccessMask = {};
            bar.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
            bar.oldLayout = vk::ImageLayout::eUndefined;
            bar.newLayout = vk::ImageLayout::eTransferDstOptimal;
            bar.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            vk::DependencyInfo dep{};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &bar;
            cmd.pipelineBarrier2(dep);
        }

        vk::ImageCopy cpy{};
        cpy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        cpy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        cpy.extent = vk::Extent3D{extent.width, extent.height, 1};
        cmd.copyImage(targets.color.image(), vk::ImageLayout::eTransferSrcOptimal,
                      post_.bloom_input().image(), vk::ImageLayout::eTransferDstOptimal, cpy);

        // Put scene back to shader read for later use (descriptors, present, dof etc)
        {
            vk::ImageMemoryBarrier2 bar{};
            bar.image = targets.color.image();
            bar.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
            bar.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
            bar.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
            bar.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
            bar.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
            bar.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            bar.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            vk::DependencyInfo dep{};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &bar;
            cmd.pipelineBarrier2(dep);
        }

        // Put bloom input to general for compute threshold
        {
            vk::ImageMemoryBarrier2 bar{};
            bar.image = post_.bloom_input().image();
            bar.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
            bar.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
            bar.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
            bar.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
            bar.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            bar.newLayout = vk::ImageLayout::eGeneral;
            bar.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            vk::DependencyInfo dep{};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &bar;
            cmd.pipelineBarrier2(dep);
        }
    }

    // Note: bloom compute temporarily disabled to resolve cb invalid state.
    // The copy populates bloom_input, which is bound and added in tonemap for visual composite.
    // TODO: re-enable record_bloom_threshold + blur once command buffer state for mixed render/compute is stable.
    // post_.record_bloom_threshold(cmd, sync_index, extent);
    // post_.record_bloom_blur(cmd, sync_index, extent);

    DynamicRenderer::transition_depth_to_shader_read(cmd, targets.depth.image());

    post_.update_descriptors(sync_index,
        targets.color.view(), targets.color.format(),
        targets.depth.view(),
        {});  // bloom bound later before present
}

void FrameRenderer::record_present_pass(std::uint32_t sync_index,
                                        std::uint32_t image_index,
                                        float time, const RetroPipelineState& state) {
    auto& cmd = command_buffers_[sync_index];
    auto extent = swapchain_.extent();

    auto& scene_targets = pipelines_.scene_targets();
    auto& scene = scene_targets.frame(sync_index);

    auto swapchain_image = swapchain_.image(image_index);
    DynamicRenderer::transition_color_to_attachment(cmd, swapchain_image);

    // Bind bloom for compositing (using the populated input for now; compute will produce proper blurred output later)
    vk::ImageView bloom_view{};
    if (post_.bloom_input().valid()) {
        // Ensure layout for sampling
        vk::ImageMemoryBarrier2 bar{};
        bar.image = post_.bloom_input().image();
        bar.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
        bar.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
        bar.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
        bar.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        bar.oldLayout = vk::ImageLayout::eGeneral;
        bar.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        bar.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        vk::DependencyInfo dep{};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &bar;
        cmd.pipelineBarrier2(dep);

        bloom_view = post_.bloom_input().view();
    }

    post_.update_descriptors(sync_index,
        scene.color.view(), scene.color.format(),
        scene.depth.view(),
        bloom_view);

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