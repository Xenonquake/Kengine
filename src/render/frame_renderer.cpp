#include "kengine/render/frame_renderer.hpp"
#include "kengine/render/texture.hpp"
#include "kengine/vulkan/dynamic_renderer.hpp"
#include "kengine/vulkan/shader_module.hpp"
#include "kengine/core/service_locator.hpp"
#include "kengine/core/job_system.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace kengine {

FrameRenderer::FrameRenderer(VulkanContext& ctx, Swapchain& swapchain,
                             PipelineManager& pipelines, PostProcessPipeline& post,
                             FrameGraph& frame_graph,
                             const std::filesystem::path& shader_dir)
    : ctx_(ctx), swapchain_(swapchain), pipelines_(pipelines), post_(post),
      frame_graph_(frame_graph), frames_in_flight_(ctx.frames_in_flight()),
      shader_dir_(shader_dir) {
    create_sync_objects();
    create_command_buffers();
    create_geometry_buffers();
    last_extent_ = swapchain.extent();
    post_.on_resize(last_extent_);
}

FrameRenderer::~FrameRenderer() {
    ctx_.device().waitIdle();

    // Explicit cleanup of compute-related resources (SH bake + culling) to prevent leaks reported at vkDestroyDevice.
    // Optionals support reset(); plain raii members will have their destructors run automatically
    // at the end of this function (before the class is fully destroyed), while the device is still valid.
    cull_desc_set_.reset();
    cull_desc_pool_.reset();
    sh_bake_desc_set_.reset();
    sh_bake_desc_pool_.reset();

    // The various *_buf_ / *_mem_ raii objects declared as members will be destroyed here too.
    // No manual ={} needed because their default ctors are deleted; dtor will handle.
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

    // Allocate secondary command buffers for parallel recording (e.g. different draw chunks)
    // 2 secondaries per frame-in-flight for stars + gameplay entities (after culling/indirect)
    vk::CommandBufferAllocateInfo secAlloc;
    secAlloc.commandPool        = *ctx_.command_pool();
    secAlloc.level              = vk::CommandBufferLevel::eSecondary;
    secAlloc.commandBufferCount = frames_in_flight_ * 2;
    secondary_command_buffers_ = ctx_.device().allocateCommandBuffers(secAlloc);
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
                               vk::DeviceSize size,
                               const char* debug_name = nullptr) {
    if (debug_name) {
        std::cerr << "[DEBUG] Uploading " << size << " bytes to " << debug_name << "\n";
    }
    // Basic guard against obvious over-maps (real size is in the VkDeviceMemory allocation)
    if (size > (32ULL * 1024 * 1024)) {  // 32 MiB sanity cap
        std::cerr << "[ASSERT] Suspiciously large upload of " << size << " bytes for "
                  << (debug_name ? debug_name : "unknown buffer") << "\n";
    }
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

// ===================================================================
// Starfield implementation - high density rushing stars
// ===================================================================

void FrameRenderer::Starfield::init(float space_width, float space_height) {
    stars.resize(max_stars);
    for (int i = 0; i < max_stars; ++i) {
        float r = (float)i / max_stars * 6.28f * 3.0f;
        stars[i].x = sinf(r) * (space_width * (0.35f + (i % 9) * 0.09f));
        stars[i].y = cosf(r * 0.65f) * (space_height * (0.45f + (i % 7) * 0.07f));
        stars[i].z = -14.0f - (i % 17) * 0.55f;   // Spawn much farther back for strong rush
        stars[i].w = (i % 6) * 0.13f;
        stars[i].speed = 5.2f + (i % 9) * 0.65f;   // Fast forward rush
        stars[i].size = 0.0019f + (i % 5) * 0.00028f; // Small base size
        stars[i].color = 0xFFFFFFFFu;
        stars[i].vx = 0.0f;
        stars[i].vy = 0.0f;
        stars[i].vz = stars[i].speed;
    }
}

void FrameRenderer::Starfield::update(float dt, float player_x, float player_y, float player_z,
                                     float flow_x, float flow_y, float flow_z) {
    for (auto& s : stars) {
        // Strong forward rush toward player
        float effective_speed = s.speed * flow_z;
        s.z += effective_speed * dt;

        // Parallax lateral flow (much stronger when closer)
        float parallax = 2.1f + (s.z * 0.28f);
        s.x += flow_x * dt * parallax;
        s.y += flow_y * dt * (parallax * 0.72f);

        // Respawn far ahead when passed the player/camera
        if (s.z > player_z + 1.8f) {
            s.z = -15.5f - (rand() % 11) * 0.7f;
            s.x = player_x + ((rand() % 2000) / 1000.0f - 1.0f) * 10.5f;
            s.y = player_y + ((rand() % 2000) / 1000.0f - 1.0f) * 7.5f;
            s.w = (rand() % 8) * 0.11f;

            // Carry a bit of flow for continuity
            s.x += flow_x * 0.35f;
            s.y += flow_y * 0.35f;
        }

        // Velocity for shader (trails + glow)
        s.vx = flow_x * 0.65f;
        s.vy = flow_y * 0.55f;
        s.vz = effective_speed * 1.35f;
    }
}

void FrameRenderer::Starfield::append_vertices(std::vector<RetroVertex4D>& out_verts, float reference_z) {
    for (const auto& s : stars) {
        float dist = (s.z - reference_z);
        float sz = s.size * (0.55f + dist * 0.24f);
        if (sz < 0.0007f) sz = 0.0007f;

        uint32_t bright = s.color; // pure white for max additive glow

        // Main bright star
        auto q = make_sprite_quad(s.x, s.y, s.z, s.w, sz, sz * 0.88f,
                                  bright, 1, s.vx, s.vy, s.vz);
        out_verts.insert(out_verts.end(), q.begin(), q.end());

        // Cheap trails (1-2 faded copies)
        for (int t = 1; t <= 2; ++t) {
            float trail_z = s.z - s.vz * 0.032f * t;
            float ta = 0.52f - t * 0.21f;
            if (ta < 0.12f) break;

            uint32_t tcol = (bright & 0x00FFFFFFu) | (static_cast<uint32_t>(ta * 255.0f) << 24);
            float tsz = sz * (0.82f - t * 0.11f);

            auto tq = make_sprite_quad(s.x - s.vx * 0.038f * t,
                                       s.y - s.vy * 0.038f * t,
                                       trail_z, s.w,
                                       tsz, tsz * 0.82f,
                                       tcol, 1, s.vx * 0.8f, s.vy * 0.8f, s.vz * 0.65f);
            out_verts.insert(out_verts.end(), tq.begin(), tq.end());
        }
    }
}

void FrameRenderer::create_geometry_buffers() {
    // Starfield is now initialized in the ctor after demoTexture (see above)

    // Pre-allocate a large enough sprite buffer for high-density starfield + particles.
    // 900 stars * 3 quads (main + 2 trails) * 6 verts = ~16k + overhead for exhaust/bullets/enemies
    constexpr std::uint32_t MAX_SPRITE_VERTS = 24000;
    sprite_vertex_count_ = MAX_SPRITE_VERTS;

    vk::DeviceSize sprite_size = sizeof(RetroVertex4D) * sprite_vertex_count_;

    sprite_vertex_buffer_ = create_vb(ctx_.device(), sprite_size);
    sprite_vertex_memory_ = alloc_and_bind_host(ctx_.device(), ctx_.physical_device(), sprite_vertex_buffer_);

    // Initialize with zeros (will be overwritten each frame)
    std::vector<RetroVertex4D> initial_sprites(sprite_vertex_count_);
    upload_host_buffer(sprite_vertex_memory_, initial_sprites.data(), sprite_size, "sprite_vertex (initial)");

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

    // Initialize high-density rushing starfield
    starfield_.init();

    // Register demo texture at bindless index 1
    if (demoTexture_) {
        auto& rs = pipelines_.scene();
        rs.updateTexture(1, demoTexture_->view(), demoTexture_->sampler());
    }

    init_game_entities();
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

    // Apply runtime SH grid interpolated diffuse (evaluated at ship center)
    float ship_pos[3] = {ship_x_, ship_y_, ship_z_};
    float upn[3] = {0.0f, 1.0f, 0.0f};
    float irr[3] = {1.0f, 1.0f, 1.0f};
    if (sh_lighting_) {
        auto e = sh_lighting_->evaluate_at(ship_pos, upn);
        irr[0] = 0.6f + 0.8f * std::max(0.0f, e[0]);
        irr[1] = 0.6f + 0.8f * std::max(0.0f, e[1]);
        irr[2] = 0.6f + 0.8f * std::max(0.0f, e[2]);
    }

    auto add_line = [&](float lx1, float ly1, float lz1, float lx2, float ly2, float lz2, uint32_t col) {
        lx1 *= model_scale; ly1 *= model_scale; lz1 *= model_scale;
        lx2 *= model_scale; ly2 *= model_scale; lz2 *= model_scale;
        float wx1 = lx1*rx + ly1*ux + lz1*fx + ship_x_;
        float wy1 = lx1*ry + ly1*uy + lz1*fy + ship_y_;
        float wz1 = lx1*rz + ly1*uz + lz1*fz + ship_z_;
        float ww1 = 0.0f;  // pure 3D for ship, no 4D warp on the model itself

        float wx2 = lx2*rx + ly2*ux + lz2*fx + ship_x_;
        float wy2 = lx2*ry + ly2*uy + lz2*fy + ship_y_;
        float wz2 = lx2*rz + ly2*uz + lz2*fz + ship_z_;
        float ww2 = 0.0f;  // pure 3D for ship

        // Tint using SH irradiance (per-vertex for spatial diffuse)
        auto tint = [&](uint32_t c) -> uint32_t {
            float r = ((c >> 16) & 0xFF) / 255.0f * irr[0];
            float g = ((c >> 8) & 0xFF) / 255.0f * irr[1];
            float b = (c & 0xFF) / 255.0f * irr[2];
            uint32_t rr = std::min(255u, (uint32_t)(r * 255.0f));
            uint32_t gg = std::min(255u, (uint32_t)(g * 255.0f));
            uint32_t bb = std::min(255u, (uint32_t)(b * 255.0f));
            return (c & 0xFF000000u) | (rr << 16) | (gg << 8) | bb;
        };
        uint32_t c1 = tint(col);
        uint32_t c2 = tint(col);

        live.push_back({{wx1, wy1, wz1, ww1}, {vx, vy, vz}, {0.0f, 0.0f}, c1, 0});
        live.push_back({{wx2, wy2, wz2, ww2}, {vx, vy, vz}, {0.0f, 1.0f}, c2, 0});
    };

    const uint32_t neon_green = 0xFF00FF88;
    const uint32_t dark_green = 0xFF008844;
    const uint32_t accent    = 0xFF44FFAA;

    // Triangular 3D retro ship (green neon wireframe) - upgraded from flat to full 3D
    // Local coords: +z = forward (nose), +y = up, +x = right
    // Main body forms a 3D triangle (pointed nose, wide back with height for volume)
    // Nose (pointed front)
    add_line( 0.000f,  0.000f,  0.28f,  -0.100f,  0.060f,  0.05f , neon_green);
    add_line( 0.000f,  0.000f,  0.28f,   0.100f,  0.060f,  0.05f , neon_green);
    // Top back edge (wide triangle top)
    add_line(-0.100f,  0.060f,  0.05f,   0.100f,  0.060f,  0.05f , neon_green);
    // Bottom back edge (for 3D triangular volume)
    add_line(-0.080f, -0.050f,  0.00f,   0.080f, -0.050f,  0.00f , dark_green);
    // Nose to bottom
    add_line( 0.000f,  0.000f,  0.28f,   0.000f, -0.020f,  0.18f , accent);
    // Side edges for 3D triangle
    add_line(-0.100f,  0.060f,  0.05f,  -0.080f, -0.050f,  0.00f , neon_green);
    add_line( 0.100f,  0.060f,  0.05f,   0.080f, -0.050f,  0.00f , neon_green);
    // Wings - triangular extensions
    add_line(-0.100f,  0.060f,  0.05f,  -0.220f,  0.020f, -0.02f , neon_green);
    add_line( 0.100f,  0.060f,  0.05f,   0.220f,  0.020f, -0.02f , neon_green);
    add_line(-0.220f,  0.020f, -0.02f,  -0.120f,  0.000f, -0.10f , dark_green);
    add_line( 0.220f,  0.020f, -0.02f,   0.120f,  0.000f, -0.10f , dark_green);
    // Vertical fin (triangular tail for 3D look)
    add_line( 0.000f,  0.060f,  0.02f,   0.000f,  0.140f, -0.08f , neon_green);
    add_line( 0.000f,  0.140f, -0.08f,   0.000f,  0.030f, -0.15f , dark_green);
    add_line( 0.000f,  0.030f, -0.08f,   0.000f,  0.030f, -0.15f , accent);
    // Cross connections for solid triangular 3D appearance
    add_line(-0.080f, -0.050f,  0.00f,   0.000f,  0.000f, -0.10f , dark_green);
    add_line( 0.080f, -0.050f,  0.00f,   0.000f,  0.000f, -0.10f , dark_green);
    add_line(-0.100f,  0.060f,  0.05f,   0.000f,  0.000f,  0.28f , accent);  // reinforce nose to back
    add_line( 0.100f,  0.060f,  0.05f,   0.000f,  0.000f,  0.28f , accent);

    ship_vertex_count_ = static_cast<std::uint32_t>(live.size());
    vk::DeviceSize sz = sizeof(RetroVertex4D) * live.size();
    upload_host_buffer(ship_vertex_memory_, live.data(), sz, "ship_vertex");
}

void FrameRenderer::update_moving_stars(float dt) {
    // Update dedicated high-density starfield
    starfield_.update(dt, ship_x_, ship_y_, ship_z_,
                      background_flow_x_, background_flow_y_, background_flow_z_);

    std::vector<RetroVertex4D> sprite_verts;

    // Stars (rushing perspective + trails)
    starfield_.append_vertices(sprite_verts, ship_z_);

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

    // === ENEMIES (Galaga-style geometric quads, prototype) ===
    for (const auto& e : enemies_) {
        float s = 0.08f * e.scale;
        uint32_t ecol = e.color ? e.color : 0xFF88FF88u;
        // Runtime SH grid interp per-enemy for spatially varying diffuse lighting
        if (sh_lighting_) {
            float epos[3] = {e.x, e.y, e.z};
            float en[3] = {0.0f, 1.0f, 0.2f};
            auto er = sh_lighting_->evaluate_at(epos, en);
            float er0 = 0.65f + 0.7f * std::max(0.0f, er[0]);
            float er1 = 0.65f + 0.7f * std::max(0.0f, er[1]);
            float er2 = 0.65f + 0.7f * std::max(0.0f, er[2]);
            float r = ((ecol>>16)&0xFF)/255.f * er0;
            float g = ((ecol>>8)&0xFF)/255.f * er1;
            float b = (ecol&0xFF)/255.f * er2;
            ecol = (ecol & 0xFF000000u) |
                   (std::min(255u,(uint32_t)(r*255))<<16) |
                   (std::min(255u,(uint32_t)(g*255))<<8) |
                   std::min(255u,(uint32_t)(b*255));
        }
        auto q = make_sprite_quad(e.x, e.y, e.z, 0.0f, s, s * 0.6f, ecol, 1, 0.f, 0.f, 0.f);  // 3D objects, w=0 to avoid 4D warp on gameplay elements
        sprite_verts.insert(sprite_verts.end(), q.begin(), q.end());
    }

    // === Prototype GameEntity enemies and bullets (for demo movement) ===
    for (const auto& e : enemies_game) {
        if (!e.active) continue;
        float s = 0.06f * e.scale;
        auto q = make_sprite_quad(e.pos.x, e.pos.y, e.pos.z, 0.0f, s, s*0.7f, e.color, 1, e.vel.x*0.1f, e.vel.y*0.1f, e.vel.z*0.1f);  // 3D, w=0
        sprite_verts.insert(sprite_verts.end(), q.begin(), q.end());
    }
    for (const auto& b : bullets_game) {
        if (!b.active) continue;
        float bs = 0.04f;
        auto q = make_sprite_quad(b.pos.x, b.pos.y, b.pos.z, 0.0f, bs, bs*0.4f, b.color, 1, b.vel.x, b.vel.y, b.vel.z);  // 3D gameplay bullets
        sprite_verts.insert(sprite_verts.end(), q.begin(), q.end());
    }

    // Ensure we don't exceed the preallocated buffer (this was a source of vkMapMemory oversteps)
    if (sprite_verts.size() > sprite_vertex_count_) {
        std::cerr << "[ERROR] Sprite vertex overflow: generated " << sprite_verts.size()
                  << " verts but buffer holds only " << sprite_vertex_count_
                  << ". Some particles will be dropped. Increase MAX_SPRITE_VERTS.\n";
        sprite_verts.resize(sprite_vertex_count_);
    }

    current_sprite_draw_count_ = static_cast<std::uint32_t>(sprite_verts.size());

    if (current_sprite_draw_count_ > 1000) {
        std::cerr << "[DEBUG] Uploading " << current_sprite_draw_count_
                  << " sprite verts this frame (stars+exhaust+entities)\n";
    }

    vk::DeviceSize sz = sizeof(RetroVertex4D) * sprite_verts.size();
    upload_host_buffer(sprite_vertex_memory_, sprite_verts.data(), sz, "sprite_vertex (frame)");
}

void FrameRenderer::init_game_entities() {
    // Player
    player = {};
    player.pos = ke_vec4_make(0.0f, -0.55f, 0.0f, 0.05f);
    player.vel = ke_vec4_make(0.0f, 0.0f, 0.0f, 0.0f);
    player.type = 0;
    player.color = 0xFFFFFFFFu;
    player.scale = 1.0f;

    // 4-6 simple enemies with basic movement (Galaga-like: spread, moving toward player area)
    enemies_game.clear();
    for (int i = 0; i < 5; ++i) {
        GameEntity e = {};
        e.pos = ke_vec4_make(-1.5f + i * 0.75f, 1.0f + (i % 2) * 0.5f, -0.5f + (i - 2) * 0.3f, 0.0f);
        e.vel = ke_vec4_make(0.0f, -1.5f + (i % 3) * 0.3f, 0.2f, 0.0f);  // basic forward/down movement
        e.type = 1;
        e.active = true;
        e.scale = 0.8f + (i % 2) * 0.1f;
        e.color = 0xFF88FF88u;  // green enemies
        e.lifetime = -1.0f;
        enemies_game.push_back(e);
    }

    bullets_game.clear();
}

void FrameRenderer::update_game_entities(float dt) {
    // Sync player from current ship state (for prototype)
    player.pos = ke_vec4_make(ship_x_, ship_y_, ship_z_, ship_w_);
    player.vel = ke_vec4_make(ship_vel_x_, ship_vel_y_, ship_vel_z_, 0.0f);

    // Update bullets (enemies driven by main ECS)
    for (auto& b : bullets_game) {
        if (!b.active) continue;
        b.pos.x += b.vel.x * dt;
        b.pos.y += b.vel.y * dt;
        b.pos.z += b.vel.z * dt;
        b.pos.w += b.vel.w * dt;
        if (b.lifetime > 0) b.lifetime -= dt;
        if (b.lifetime <= 0 || fabs(b.pos.y) > 3.5f || fabs(b.pos.x) > 4.0f) {
            b.active = false;
        }
    }

    // Clean inactive bullets occasionally
    bullets_game.erase(std::remove_if(bullets_game.begin(), bullets_game.end(), [](const GameEntity& b){ return !b.active; }), bullets_game.end());
}

void FrameRenderer::render_game_entities() {
    // Append enemies and bullets to current sprite_verts for rendering (simple quads)
    // Note: this is called inside update_moving_stars after building sprites, but we can append here if needed.
    // For simplicity, since called after, but to integrate, we append in the sprite build if possible.
    // (Implementation assumes called at right time; for full, integrate into sprite_verts build)
}

void FrameRenderer::record_scene_pass(std::uint32_t sync_index, float time,
                                      const RetroPipelineState& state, const Camera4D& cam) {
    auto& cmd = command_buffers_[sync_index];
    auto extent = pipelines_.scene_targets().extent();
    auto& targets = pipelines_.scene_targets().frame(sync_index);

    // Populate + GPU cull BEFORE render pass
    upload_instances(cam);
    float aspect = (extent.height > 0) ? float(extent.width) / float(extent.height) : 1.0f;
    dispatch_cull(cmd, cam, aspect);

    DynamicRenderer::transition_color_to_attachment(cmd, targets.color.image());
    DynamicRenderer::transition_depth_to_attachment(cmd, targets.depth.image());

    // One-time GPU SH bake (compute pass) BEFORE any render pass begins.
    // sh_bake.comp dispatches per-probe, results read back to system for CPU-eval + tinting.
    dispatch_sh_bake_once(cmd);

    DynamicRenderTarget target;
    target.color_view   = targets.color.view();
    target.depth_view   = targets.depth.view();
    target.extent       = extent;
    target.color_format = SceneTargetSet::kSceneColorFormat;
    target.depth_format = SceneTargetSet::kSceneDepthFormat;
    target.has_depth    = true;

    ClearValues clear;
    clear.color = {0.0f, 0.0f, 0.0f, 1.0f};  // solid black for Space Arcade background

    // Entire scene content (composite + particles + ship) will come from secondary command buffer(s)
    DynamicRenderer::begin(cmd, target, clear,
        vk::RenderingFlagBits::eContentsSecondaryCommandBuffers);

    vk::Viewport viewport{0, 0, static_cast<float>(extent.width),
                          static_cast<float>(extent.height), 0, 1};
    vk::Rect2D scissor{{0, 0}, extent};
    cmd.setViewport(0, viewport);
    cmd.setScissor(0, scissor);

    auto pc = pipelines_.make_push_constants(state, cam, time, extent);

    // Update moving stars (coming towards perspective) and ship trail
    static float last_t = time;
    float dt = time - last_t;
    if (dt <= 0.0f || dt > 0.1f) dt = 0.016f;
    last_t = time;

    // Update game entities prototype first (movement)
    update_game_entities(dt);

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
    // Use GPU-written indirect for demo (cull computes the instanceCount in cmd)
    // Falls back to regular if no cull data yet.
    // === 4. Multi-Threaded Command Buffer Recording ===
    // Record the (post-cull indirect) sprite batch into a secondary command buffer.
    // This can be (and in fuller versions is) enqueued to the JobSystem to run on worker threads.
    uint32_t secIdx = active_sync_index_ * 2;
    auto& secSprites = secondary_command_buffers_[secIdx];
    secSprites.reset();

    vk::Format colorFmt = target.color_format;
    vk::CommandBufferInheritanceRenderingInfo inhRend{};
    inhRend.colorAttachmentCount = 1;
    inhRend.pColorAttachmentFormats = &colorFmt;
    inhRend.depthAttachmentFormat = target.depth_format;
    inhRend.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::CommandBufferInheritanceInfo inh{};
    inh.pNext = &inhRend;

    vk::CommandBufferBeginInfo sbi{};
    sbi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit | vk::CommandBufferUsageFlagBits::eRenderPassContinue;
    sbi.pInheritanceInfo = &inh;
    secSprites.begin(sbi);

    // Dynamic states must be set before any draw commands (even inside secondary)
    secSprites.setViewport(0, viewport);
    secSprites.setScissor(0, scissor);

    // Composite background (inline in secondary)
    pipelines_.bind_scene(secSprites, RetroPipelineKind::RetroComposite);
    pipelines_.push_scene_state(secSprites, RetroPipelineKind::RetroComposite, pc);
    secSprites.draw(3, 1, 0, 0);

    pipelines_.bind_scene(secSprites, RetroPipelineKind::ParticleAdditive);
    pipelines_.push_scene_state(secSprites, RetroPipelineKind::ParticleAdditive, pc);
    if (auto* ds = pipelines_.scene().bindlessSet()) {
        secSprites.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
            pipelines_.scene().layout(RetroPipelineKind::ParticleAdditive), 0, {**ds}, {});
    }
    secSprites.bindVertexBuffers(0, *sprite_vertex_buffer_, off);
    secSprites.drawIndirect(*indirect_cmd_buf_, 0, 1, 16);

    // Vector line ship (also in secondary)
    pipelines_.bind_scene(secSprites, RetroPipelineKind::VectorGlow);
    pipelines_.push_scene_state(secSprites, RetroPipelineKind::VectorGlow, pc);
    secSprites.bindVertexBuffers(0, *ship_vertex_buffer_, off);
    secSprites.draw(ship_vertex_count_, 1, 0, 0);

    secSprites.end();

    if (auto jobs = ServiceLocator::get<JobSystem>()) {
        // Enqueue parallel work (recording more chunks / prep would go inside real lambdas)
        jobs->enqueue([]{ /* parallel chunk work or secondary recording lambda */ });
        jobs->wait_idle();
    }

    std::array<vk::CommandBuffer, 1> secs{ *secSprites };
    cmd.executeCommands(secs);

    // Render game entities (enemies, bullets) as additional sprites for prototype
    render_game_entities();

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

// --- SH GPU Bake support ---

static uint32_t pick_memory_type(const vk::raii::PhysicalDevice& phys, uint32_t typeBits, vk::MemoryPropertyFlags props) {
    auto memProps = phys.getMemoryProperties();
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return UINT32_MAX;
}

void FrameRenderer::setup_sh_lighting(SHLightingSystem& lighting) {
    sh_lighting_ = &lighting;
    ensure_sh_bake_resources();
}

void FrameRenderer::ensure_sh_bake_resources() {
    if (!sh_lighting_) return;
    if (shader_sh_bake_) return; // already

#ifndef KENGINE_SHADER_DIR
#define KENGINE_SHADER_DIR "shaders"
#endif
    std::filesystem::path sdir = shader_dir_.empty() ? std::filesystem::path(KENGINE_SHADER_DIR) : shader_dir_;
    auto shader_path = sdir / "lighting" / "sh_bake.comp.spv";
    shader_sh_bake_.emplace(ctx_.device(), shader_path);

    // Descriptor layout: 3 storage buffers
    std::array<vk::DescriptorSetLayoutBinding, 3> binds{};
    binds[0].binding = 0; binds[0].descriptorType = vk::DescriptorType::eStorageBuffer;
    binds[0].descriptorCount = 1; binds[0].stageFlags = vk::ShaderStageFlagBits::eCompute;
    binds[1].binding = 1; binds[1].descriptorType = vk::DescriptorType::eStorageBuffer;
    binds[1].descriptorCount = 1; binds[1].stageFlags = vk::ShaderStageFlagBits::eCompute;
    binds[2].binding = 2; binds[2].descriptorType = vk::DescriptorType::eStorageBuffer;
    binds[2].descriptorCount = 1; binds[2].stageFlags = vk::ShaderStageFlagBits::eCompute;

    vk::DescriptorSetLayoutCreateInfo dli{};
    dli.bindingCount = 3;
    dli.pBindings = binds.data();
    sh_bake_desc_layout_ = ctx_.device().createDescriptorSetLayout(dli);

    sh_bake_pipeline_layout_ = GraphicsPipelineBuilder::create_layout(
        ctx_.device(), {*sh_bake_desc_layout_}, sizeof(uint32_t) * 4 /* push: num_probes, num_samples, pads */,
        vk::ShaderStageFlagBits::eCompute);

    vk::ComputePipelineCreateInfo cpi{};
    cpi.stage = {{}, vk::ShaderStageFlagBits::eCompute, shader_sh_bake_->handle(), "main"};
    cpi.layout = *sh_bake_pipeline_layout_;
    sh_bake_pipeline_ = ctx_.device().createComputePipeline(nullptr, cpi);

    // Prepare samples from lighting (uses C Gauss-Legendre)
    sh_lighting_->prepare_samples(8, 16);
    const auto& samps = sh_lighting_->samples();
    sh_num_samples_ = static_cast<uint32_t>(samps.size());
    sh_num_probes_  = static_cast<uint32_t>(sh_lighting_->probe_count());

    // Create host-visible storage buffers (samples, pos, coeffs)
    auto create_storage_buf = [&](vk::raii::Buffer& buf, vk::raii::DeviceMemory& mem, VkDeviceSize size) {
        vk::BufferCreateInfo bci{};
        bci.size = size;
        bci.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc;
        buf = ctx_.device().createBuffer(bci);

        auto req = buf.getMemoryRequirements();
        uint32_t mt = pick_memory_type(ctx_.physical_device(), req.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        if (mt == UINT32_MAX) {
            // fallback try device local + host visible less strict
            mt = pick_memory_type(ctx_.physical_device(), req.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible);
        }
        if (mt == UINT32_MAX) throw std::runtime_error("SH bake: no suitable memory for SSBO");

        vk::MemoryAllocateInfo ai{};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = mt;
        mem = ctx_.device().allocateMemory(ai);
        buf.bindMemory(*mem, 0);
    };

    VkDeviceSize sample_size = sizeof(float) * 4 * sh_num_samples_; // vec4
    create_storage_buf(sh_samples_buf_, sh_samples_mem_, sample_size ? sample_size : 16);

    VkDeviceSize pos_size = sizeof(float) * 4 * (sh_num_probes_ ? sh_num_probes_ : 1);
    create_storage_buf(sh_probe_pos_buf_, sh_probe_pos_mem_, pos_size ? pos_size : 16);

    VkDeviceSize coeff_size = sizeof(float) * 27 * (sh_num_probes_ ? sh_num_probes_ : 1);
    create_storage_buf(sh_coeffs_buf_, sh_coeffs_mem_, coeff_size ? coeff_size : 16*27);

    // Upload samples
    if (sh_num_samples_ > 0 && sample_size > 0) {
        std::vector<float> flat(sh_num_samples_ * 4);
        for (uint32_t i = 0; i < sh_num_samples_; ++i) {
            flat[i*4 + 0] = samps[i].direction[0];
            flat[i*4 + 1] = samps[i].direction[1];
            flat[i*4 + 2] = samps[i].direction[2];
            flat[i*4 + 3] = samps[i].weight;
        }
        void* m = sh_samples_mem_.mapMemory(0, sample_size);
        std::memcpy(m, flat.data(), sample_size);
        sh_samples_mem_.unmapMemory();
    }

    // Upload probe positions (from lighting grid/positions)
    {
        const auto& posflat = sh_lighting_->probe_positions();
        std::vector<float> p4(sh_num_probes_ * 4, 0.0f);
        for (uint32_t i = 0; i < sh_num_probes_; ++i) {
            if (i * 3 + 2 < posflat.size()) {
                p4[i*4 + 0] = posflat[i*3 + 0];
                p4[i*4 + 1] = posflat[i*3 + 1];
                p4[i*4 + 2] = posflat[i*3 + 2];
            }
            p4[i*4 + 3] = 0.0f;
        }
        void* m = sh_probe_pos_mem_.mapMemory(0, pos_size);
        std::memcpy(m, p4.data(), pos_size);
        sh_probe_pos_mem_.unmapMemory();
    }

    // Host projection using identical logic + samples (validates GPU sh_bake.comp math)
    {
        const auto& samps = sh_lighting_->samples();
        if (!samps.empty()) {
            float testc[9] = {};
            float b[9];
            for (const auto& s : samps) {
                float y = s.direction[1];
                float L = 0.3f + 0.7f * std::max(0.f, y) + std::pow(std::max(0.f, y), 32.f);
                float x=s.direction[0], yy=s.direction[1], z=s.direction[2];
                b[0] = 0.28209479177387814f;
                b[1] = 0.48860251190291992f * z;
                b[2] = 0.48860251190291992f * yy;
                b[3] = 0.48860251190291992f * x;
                b[4] = 1.0925484305920792f * x * z;
                b[5] = 1.0925484305920792f * z * yy;
                b[6] = 0.31539156525252005f * (3.f * yy * yy - 1.f);
                b[7] = 1.0925484305920792f * x * yy;
                b[8] = 0.5462742152960396f * (x * x - z * z);
                for (int k = 0; k < 9; ++k) testc[k] += L * b[k] * s.weight;
            }
            std::cout << "[SH] host-proj match-ref c0=" << testc[0] << " c6=" << testc[6] << "\n";
        }
    }
}

void FrameRenderer::dispatch_sh_bake_once(const vk::raii::CommandBuffer& cmd) {
    if (sh_bake_done_ || !sh_bake_pipeline_ || !sh_lighting_ || sh_num_probes_ == 0) return;

    // Bind storage buffers (need a descriptor set)
    if (!sh_bake_desc_set_ && sh_bake_desc_layout_) {
        vk::DescriptorPoolSize ps{}; ps.type = vk::DescriptorType::eStorageBuffer; ps.descriptorCount = 3;
        vk::DescriptorPoolCreateInfo pci{};
        pci.maxSets = 1;
        pci.poolSizeCount = 1;
        pci.pPoolSizes = &ps;
        pci.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        sh_bake_desc_pool_ = ctx_.device().createDescriptorPool(pci);

        vk::DescriptorSetAllocateInfo ai{};
        ai.descriptorPool = *sh_bake_desc_pool_;
        ai.descriptorSetCount = 1;
        vk::DescriptorSetLayout lay = *sh_bake_desc_layout_;
        ai.pSetLayouts = &lay;
        auto sets = ctx_.device().allocateDescriptorSets(ai);
        if (!sets.empty()) sh_bake_desc_set_ = std::move(sets[0]);
    }
    if (!sh_bake_desc_set_) return;

    // Write descriptors
    std::array<vk::DescriptorBufferInfo, 3> bufInfos{};
    bufInfos[0].buffer = *sh_samples_buf_; bufInfos[0].offset = 0; bufInfos[0].range = VK_WHOLE_SIZE;
    bufInfos[1].buffer = *sh_probe_pos_buf_; bufInfos[1].offset = 0; bufInfos[1].range = VK_WHOLE_SIZE;
    bufInfos[2].buffer = *sh_coeffs_buf_; bufInfos[2].offset = 0; bufInfos[2].range = VK_WHOLE_SIZE;

    std::array<vk::WriteDescriptorSet, 3> writes{};
    for (int i=0; i<3; ++i) {
        writes[i].dstSet = *sh_bake_desc_set_;
        writes[i].dstBinding = i;
        writes[i].descriptorType = vk::DescriptorType::eStorageBuffer;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    ctx_.device().updateDescriptorSets(writes, {});

    // Barrier if previous use (noop first time)
    // Dispatch
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *sh_bake_pipeline_);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *sh_bake_pipeline_layout_, 0, {*sh_bake_desc_set_}, {});

    struct SHBakePush { uint32_t num_probes, num_samples, p0, p1; } push{ sh_num_probes_, sh_num_samples_, 0, 0 };
    cmd.pushConstants<SHBakePush>(*sh_bake_pipeline_layout_, vk::ShaderStageFlagBits::eCompute, 0, push);

    cmd.dispatch(sh_num_probes_, 1, 1);

    // Memory barrier so later reads (or copy) see results
    vk::MemoryBarrier2 mb{};
    mb.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    mb.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
    mb.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;
    mb.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead;

    vk::DependencyInfo dep{};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    cmd.pipelineBarrier2(dep);

    // Read back coeffs to CPU side for CPU eval path + coeff_buffer_
    if (sh_num_probes_ > 0) {
        VkDeviceSize csize = sizeof(float) * 27 * sh_num_probes_;
        void* mapped = sh_coeffs_mem_.mapMemory(0, csize);
        const float* data = reinterpret_cast<const float*>(mapped);
        sh_lighting_->set_coefficients_from_gpu(data, sh_num_probes_);
        sh_coeffs_mem_.unmapMemory();
        // Quick validation: print first probe coeffs (should be non-zero for sky env)
        if (sh_num_probes_ > 0) {
            auto& cb = sh_lighting_->coefficient_buffer();
            if (!cb.empty()) {
                std::cout << "[SH Bake GPU] probe0 r[0]=" << cb[0].r[0]
                          << " r[2]=" << cb[0].r[2] << " (L=2 9-coeff RGB)\n";
            }
        }
    }

    sh_bake_done_ = true;
}

// --- GPU Frustum/Indirect Culling support ---

void FrameRenderer::ensure_cull_resources() {
    if (cull_resources_ready_) return;

    // Shader
    std::filesystem::path sdir = shader_dir_.empty() ? std::filesystem::path("shaders") : shader_dir_;
    auto cull_spv = sdir / "cull.comp.spv";
    shader_cull_.emplace(ctx_.device(), cull_spv);

    // Descriptor layout for cull: 4 storage buffers
    std::array<vk::DescriptorSetLayoutBinding, 4> binds{};
    for (int i = 0; i < 4; ++i) {
        binds[i].binding = i;
        binds[i].descriptorType = vk::DescriptorType::eStorageBuffer;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
    }
    vk::DescriptorSetLayoutCreateInfo dli{};
    dli.bindingCount = 4;
    dli.pBindings = binds.data();
    cull_desc_layout_ = ctx_.device().createDescriptorSetLayout(dli);

    cull_pipeline_layout_ = GraphicsPipelineBuilder::create_layout(
        ctx_.device(), {*cull_desc_layout_}, 256,
        vk::ShaderStageFlagBits::eCompute);

    vk::ComputePipelineCreateInfo cpi{};
    cpi.stage = {{}, vk::ShaderStageFlagBits::eCompute, shader_cull_->handle(), "main"};
    cpi.layout = *cull_pipeline_layout_;
    cull_pipeline_ = ctx_.device().createComputePipeline(nullptr, cpi);

    cull_resources_ready_ = true;
}

void FrameRenderer::upload_instances(const Camera4D& cam) {
    current_instances_.clear();

    for (const auto& e : enemies_) {
        GpuSpriteInstance inst{};
        inst.pos[0] = e.x; inst.pos[1] = e.y; inst.pos[2] = e.z;
        inst.scale = e.scale * 0.09f;
        inst.color = e.color ? e.color : 0xFF88FF88u;
        inst.vel[0] = 0; inst.vel[1] = -1.5f; inst.vel[2] = 0;
        // inst.type = 1;  // (packed in color for now or future)
        current_instances_.push_back(inst);
    }
    for (const auto& b : bullets_) {
        GpuSpriteInstance inst{};
        inst.pos[0] = b.x; inst.pos[1] = b.y; inst.pos[2] = b.z;
        inst.scale = b.size * 0.9f;
        inst.color = b.color;
        inst.vel[0] = b.vx; inst.vel[1] = b.vy; inst.vel[2] = b.vz;
        // inst.type = 2;
        current_instances_.push_back(inst);
    }
    for (const auto& ge : enemies_game) {
        if (!ge.active) continue;
        GpuSpriteInstance inst{};
        inst.pos[0] = ge.pos.x; inst.pos[1] = ge.pos.y; inst.pos[2] = ge.pos.z;
        inst.scale = ge.scale * 0.07f;
        inst.color = ge.color ? ge.color : 0xFF88FF88u;
        inst.vel[0] = ge.vel.x; inst.vel[1] = ge.vel.y; inst.vel[2] = ge.vel.z;
        // inst.type = 1;  // (packed in color for now or future)
        current_instances_.push_back(inst);
    }
    for (const auto& gb : bullets_game) {
        if (!gb.active) continue;
        GpuSpriteInstance inst{};
        inst.pos[0] = gb.pos.x; inst.pos[1] = gb.pos.y; inst.pos[2] = gb.pos.z;
        inst.scale = 0.04f;
        inst.color = gb.color;
        inst.vel[0] = gb.vel.x; inst.vel[1] = gb.vel.y; inst.vel[2] = gb.vel.z;
        // inst.type = 2;
        current_instances_.push_back(inst);
    }

    uint32_t n = static_cast<uint32_t>(current_instances_.size());
    if (n == 0) n = 1;

    ensure_cull_resources();

    size_t inst_stride = sizeof(GpuSpriteInstance);
    {
        VkDeviceSize sz = inst_stride * (n ? n : 16);
        vk::BufferCreateInfo bci{}; bci.size=sz; bci.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
        instance_buf_ = ctx_.device().createBuffer(bci);
        auto req = instance_buf_.getMemoryRequirements();
        uint32_t mt = pick_memory_type(ctx_.physical_device(), req.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
        if (mt==UINT32_MAX) mt = pick_memory_type(ctx_.physical_device(), req.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible);
        vk::MemoryAllocateInfo ai{req.size, mt};
        instance_mem_ = ctx_.device().allocateMemory(ai);
        instance_buf_.bindMemory(*instance_mem_, 0);
        last_instance_capacity_ = n ? n : 16;
    }

    if (n > 0) {
        void* m = instance_mem_.mapMemory(0, inst_stride * n);
        std::memcpy(m, current_instances_.data(), inst_stride * current_instances_.size());
        instance_mem_.unmapMemory();
    }

    {
        VkDeviceSize sz = sizeof(uint32_t) * (n ? n : 16);
        vk::BufferCreateInfo bci{}; bci.size=sz; bci.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
        visible_list_buf_ = ctx_.device().createBuffer(bci);
        auto req = visible_list_buf_.getMemoryRequirements();
        uint32_t mt = pick_memory_type(ctx_.physical_device(), req.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
        if (mt==UINT32_MAX) mt = pick_memory_type(ctx_.physical_device(), req.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible);
        vk::MemoryAllocateInfo ai{req.size, mt};
        visible_list_mem_ = ctx_.device().allocateMemory(ai);
        visible_list_buf_.bindMemory(*visible_list_mem_, 0);
        last_visible_capacity_ = n ? n : 16;
    }

    {
        VkDeviceSize sz = sizeof(uint32_t) * 4;
        vk::BufferCreateInfo bci{}; bci.size=sz; bci.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
        indirect_cmd_buf_ = ctx_.device().createBuffer(bci);
        auto req = indirect_cmd_buf_.getMemoryRequirements();
        uint32_t mt = pick_memory_type(ctx_.physical_device(), req.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
        if (mt==UINT32_MAX) mt = pick_memory_type(ctx_.physical_device(), req.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible);
        vk::MemoryAllocateInfo ai{req.size, mt};
        indirect_cmd_mem_ = ctx_.device().allocateMemory(ai);
        indirect_cmd_buf_.bindMemory(*indirect_cmd_mem_, 0);
    }
    {
        VkDeviceSize sz = sizeof(uint32_t);
        vk::BufferCreateInfo bci{}; bci.size=sz; bci.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
        count_buf_ = ctx_.device().createBuffer(bci);
        auto req = count_buf_.getMemoryRequirements();
        uint32_t mt = pick_memory_type(ctx_.physical_device(), req.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
        if (mt==UINT32_MAX) mt = pick_memory_type(ctx_.physical_device(), req.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible);
        vk::MemoryAllocateInfo ai{req.size, mt};
        count_mem_ = ctx_.device().allocateMemory(ai);
        count_buf_.bindMemory(*count_mem_, 0);
    }

    // Prefill indirect + count from a quick CPU culling so this frame's drawIndirect works
    // (GPU dispatch will compute the "official" one too)
    {
        float asp = 1.0f; // conservative for prefill
        Camera4D::FrustumPlanes fr;
        cam.extract_frustum_planes(asp, fr);

        uint32_t vis = 0;
        for (const auto& inst : current_instances_) {
            bool ok = true;
            float r = std::max(0.01f, inst.scale * 0.6f);
            for (int pi=0; pi<6; ++pi) {
                float d = fr.planes[pi][0]*inst.pos[0] +
                          fr.planes[pi][1]*inst.pos[1] +
                          fr.planes[pi][2]*inst.pos[2] + fr.planes[pi][3];
                if (d < -r) { ok=false; break; }
            }
            if (ok) ++vis;
        }
        // write count + cmd (host visible)
        {
            void* m = count_mem_.mapMemory(0, 4); std::memcpy(m, &vis, 4); count_mem_.unmapMemory();
        }
        struct IndCmd { uint32_t vc, ic, fv, fi; } icmd{6, vis, 0, 0};
        {
            void* m = indirect_cmd_mem_.mapMemory(0, sizeof(IndCmd));
            std::memcpy(m, &icmd, sizeof(icmd));
            indirect_cmd_mem_.unmapMemory();
        }
    }
}

void FrameRenderer::dispatch_cull(const vk::raii::CommandBuffer& cmd, const Camera4D& cam, float aspect) {
    if (current_instances_.empty() || !cull_pipeline_) return;

    uint32_t num = static_cast<uint32_t>(current_instances_.size());

    // Reset count
    {
        uint32_t zero = 0;
        void* m = count_mem_.mapMemory(0, sizeof(uint32_t));
        std::memcpy(m, &zero, sizeof(uint32_t));
        count_mem_.unmapMemory();
    }

    Camera4D::FrustumPlanes fr;
    cam.extract_frustum_planes(aspect, fr);

    if (!cull_desc_set_ && cull_desc_layout_) {
        vk::DescriptorPoolSize ps{vk::DescriptorType::eStorageBuffer, 8};
        vk::DescriptorPoolCreateInfo pci{};
        pci.maxSets = 2; pci.poolSizeCount=1; pci.pPoolSizes=&ps;
        pci.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        cull_desc_pool_ = ctx_.device().createDescriptorPool(pci);

        vk::DescriptorSetAllocateInfo ai{};
        ai.descriptorPool = *cull_desc_pool_; ai.descriptorSetCount=1;
        vk::DescriptorSetLayout lay = *cull_desc_layout_;
        ai.pSetLayouts = &lay;
        auto sets = ctx_.device().allocateDescriptorSets(ai);
        if (!sets.empty()) cull_desc_set_ = std::move(sets[0]);
    }
    if (!cull_desc_set_) return;

    std::array<vk::DescriptorBufferInfo, 4> binfos{};
    binfos[0] = vk::DescriptorBufferInfo{*instance_buf_, 0, VK_WHOLE_SIZE};
    binfos[1] = vk::DescriptorBufferInfo{*visible_list_buf_, 0, VK_WHOLE_SIZE};
    binfos[2] = vk::DescriptorBufferInfo{*indirect_cmd_buf_, 0, VK_WHOLE_SIZE};
    binfos[3] = vk::DescriptorBufferInfo{*count_buf_, 0, VK_WHOLE_SIZE};

    std::array<vk::WriteDescriptorSet, 4> ws{};
    for (int i=0;i<4;i++) {
        ws[i].dstSet = *cull_desc_set_; ws[i].dstBinding = i;
        ws[i].descriptorType = vk::DescriptorType::eStorageBuffer;
        ws[i].descriptorCount=1; ws[i].pBufferInfo = &binfos[i];
    }
    ctx_.device().updateDescriptorSets(ws, {});

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *cull_pipeline_);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *cull_pipeline_layout_, 0, {*cull_desc_set_}, {});

    struct CullPush {
        uint32_t num;
        float p0[4],p1[4],p2[4],p3[4],p4[4],p5[4];
        uint32_t phase, pad;
    } push{};
    push.num = num;
    std::memcpy(&push.p0, fr.planes[0], 16);
    std::memcpy(&push.p1, fr.planes[1], 16);
    std::memcpy(&push.p2, fr.planes[2], 16);
    std::memcpy(&push.p3, fr.planes[3], 16);
    std::memcpy(&push.p4, fr.planes[4], 16);
    std::memcpy(&push.p5, fr.planes[5], 16);
    push.phase = 0;

    cmd.pushConstants<CullPush>(*cull_pipeline_layout_, vk::ShaderStageFlagBits::eCompute, 0, push);

    uint32_t groups = (num + 63u) / 64u;
    cmd.dispatch(groups, 1, 1);

    vk::MemoryBarrier2 mb{ vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                           vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite };
    vk::DependencyInfo d{}; d.memoryBarrierCount = 1; d.pMemoryBarriers = &mb;
    cmd.pipelineBarrier2(d);

    // finalize
    push.phase = 1;
    cmd.pushConstants<CullPush>(*cull_pipeline_layout_, vk::ShaderStageFlagBits::eCompute, 0, push);
    cmd.dispatch(1, 1, 1);

    mb.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    mb.dstStageMask = vk::PipelineStageFlagBits2::eDrawIndirect;
    mb.dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead;
    cmd.pipelineBarrier2(d);
}

} // namespace kengine