#pragma once

#include "kengine/render/camera_4d.hpp"
#include "kengine/render/frame_graph.hpp"
#include "kengine/render/pipeline_manager.hpp"
#include "kengine/render/post_process.hpp"
#include "kengine/render/retro_pipeline.hpp"
#include "kengine/render/texture.hpp"
#include "kengine/vulkan/context.hpp"
#include "kengine/vulkan/swapchain.hpp"
#include "kengine/game/game_entity.hpp"
#include "kengine/lighting/spherical_harmonics.hpp"
#include "kengine/render/retro_types.hpp"
#include <cstdint>
#include <vulkan/vulkan_raii.hpp>
#include <vector>
#include <optional>
#include <filesystem>

namespace kengine {

class FrameRenderer {
public:
    FrameRenderer(VulkanContext& ctx, Swapchain& swapchain,
                  PipelineManager& pipelines, PostProcessPipeline& post,
                  FrameGraph& frame_graph,
                  const std::filesystem::path& shader_dir = {});
    ~FrameRenderer();

    void bind_frame_graph_passes();
    void render_frame(float time, const RetroPipelineState& state, const Camera4D& cam);

    bool needs_resize(vk::Extent2D extent) const;
    void on_resize(vk::Extent2D extent);
    void recreate_swapchain_sync();  // recreate per-image semaphores/fences when swapchain image count changes

    void set_ship_position(float x, float y) { ship_x_ = x; ship_y_ = y; }
    void set_ship_velocity(float vx, float vy, float vz = 0.0f) { ship_vel_x_ = vx; ship_vel_y_ = vy; ship_vel_z_ = vz; }
    void set_ship_zw(float z, float w) { ship_z_ = z; ship_w_ = w; }

    // Access for GameEntity prototype spawning
    void add_bullet(const GameEntity& b) { bullets_game.push_back(b); }
    std::vector<GameEntity>& get_bullets_game() { return bullets_game; }

    // Lightweight bullet data for renderer (populated from ECS)
    struct BulletVisual {
        float x, y, z, w;
        float vx, vy, vz;
        float size;
        std::uint32_t color;
    };

    // Receive current live bullets from game logic (for rendering + trails)
    void set_bullets(const std::vector<BulletVisual>& bullets) { bullets_ = bullets; }

    // Enemy visuals for prototype (Galaga formations)
    struct EnemyVisual {
        float x, y, z, w;
        float yaw;
        float scale;
        std::uint32_t color;
    };

    // Receive enemies
    void set_enemies(const std::vector<EnemyVisual>& e) { enemies_ = e; }

    // Update starfield flow from ship's velocity (call every frame after input/physics)
    void update_background_flow(float ship_vx, float ship_vy, float ship_vz, float dt);

    // 3D star model support
    bool loadStarMesh();
    void updateStarInstances();

    // GPU SH lighting setup + one-time bake compute dispatch support (env bake via sh_bake.comp)
    void setup_sh_lighting(SHLightingSystem& lighting);
    void ensure_sh_bake_resources();
    void dispatch_sh_bake_once(const vk::raii::CommandBuffer& cmd);

    // GPU frustum culling + indirect draw support
    void ensure_cull_resources();
    void upload_instances(const Camera4D& cam);
    void dispatch_cull(const vk::raii::CommandBuffer& cmd, const Camera4D& cam, float aspect);

private:
    void create_sync_objects();
    void create_command_buffers();
    void create_geometry_buffers();
    void update_ship_geometry(float time);
    void update_moving_stars(float dt);
    void record_scene_pass(std::uint32_t sync_index, float time,
                           const RetroPipelineState& state, const Camera4D& cam);
    void record_present_pass(std::uint32_t sync_index, std::uint32_t image_index,
                             float time, const RetroPipelineState& state);

    VulkanContext& ctx_;
    Swapchain& swapchain_;
    PipelineManager& pipelines_;
    PostProcessPipeline& post_;
    FrameGraph& frame_graph_;
    std::filesystem::path shader_dir_;

    std::uint32_t sync_index_ = 0;
    std::uint32_t active_sync_index_ = 0;
    std::uint32_t active_image_index_ = 0;
    float pending_time_ = 0.0f;
    RetroPipelineState pending_state_{};
    Camera4D pending_cam_{};
    vk::Extent2D last_extent_{};

    std::uint32_t frames_in_flight_;
    std::uint32_t swapchain_image_count_ = 0;

    // Acquire semaphores are indexed by frame-in-flight (used for acquire + submit wait)
    std::vector<vk::raii::Semaphore> acquire_semaphores_;
    // Render finished / present semaphores indexed by swapchain image (per-image to avoid reuse issues)
    std::vector<vk::raii::Semaphore> render_finished_semaphores_;
    // Present fences (via VK_KHR_swapchain_maintenance1) also per swapchain image
    std::vector<vk::raii::Fence> present_fences_;
    // CPU fences for command buffer reuse, indexed by frame-in-flight
    std::vector<vk::raii::Fence> in_flight_;
    std::vector<vk::raii::CommandBuffer> command_buffers_;
    std::vector<vk::raii::CommandBuffer> secondary_command_buffers_;  // for multi-threaded recording

    vk::raii::Buffer sprite_vertex_buffer_{nullptr};
    vk::raii::DeviceMemory sprite_vertex_memory_{nullptr};
    std::uint32_t sprite_vertex_count_ = 0;
    std::uint32_t current_sprite_draw_count_ = 0; // actual number to draw this frame (stars + exhaust + bullets)

    vk::raii::Buffer ship_vertex_buffer_{nullptr};
    vk::raii::DeviceMemory ship_vertex_memory_{nullptr};
    std::uint32_t ship_vertex_count_ = 0;

    // CPU-side ship shape base (rebuilt each update)
    std::vector<RetroVertex4D> ship_base_shape_;

    float ship_x_ = 0.0f;
    float ship_y_ = -0.65f;
    float ship_vel_x_ = 0.0f;
    float ship_vel_y_ = 0.0f;
    float ship_vel_z_ = 0.0f;
    float ship_z_ = 1.6f;
    float ship_w_ = 0.05f;

    // Ship-relative background flow for responsive starfield (Phase 1 polish)
    float background_flow_x_ = 0.0f;
    float background_flow_y_ = 0.0f;
    float background_flow_z_ = 1.0f;

    // Dedicated high-density starfield (separate from gameplay entities)
    struct Star {
        float x, y, z;      // z negative = far ahead (in front of player FOV), increases toward camera
        float w;            // light 4D layer variation
        float speed;
        float size;
        std::uint32_t color;
        float vx, vy, vz;   // for shader trails / motion glow
    };

    struct Starfield {
        std::vector<Star> stars;
        int max_stars = 256; // reduced for performance and density control

        void init(float space_width = 9.0f, float space_height = 6.5f);
        void update(float dt, float player_x, float player_y, float player_z,
                    float flow_x, float flow_y, float flow_z);
        void append_vertices(std::vector<RetroVertex4D>& out_verts, float reference_z);
    };

    Starfield starfield_;

    std::optional<Texture> demoTexture_;  // for bindless demo on stars

    std::vector<BulletVisual> bullets_;
    std::vector<EnemyVisual> enemies_;

    // Prototype game entities (fast path before full ECS integration)
    GameEntity player;
    std::vector<GameEntity> enemies_game;
    std::vector<GameEntity> bullets_game;

    void init_game_entities();
    void update_game_entities(float dt);
    void render_game_entities();  // called from update_moving_stars or similar

    // --- GPU SH Bake (L=2 probe grid) ---
    SHLightingSystem* sh_lighting_ = nullptr;
    bool sh_bake_done_ = false;

    std::optional<ShaderModule> shader_sh_bake_;

    std::optional<vk::raii::DescriptorSetLayout> sh_bake_desc_layout_;
    std::optional<vk::raii::DescriptorPool> sh_bake_desc_pool_;
    std::optional<vk::raii::DescriptorSet> sh_bake_desc_set_;
    std::optional<vk::raii::PipelineLayout> sh_bake_pipeline_layout_;
    std::optional<vk::raii::Pipeline> sh_bake_pipeline_;

    // SSBOs for bake (host-visible for easy sync/readback on small data)
    vk::raii::Buffer sh_samples_buf_{nullptr};
    vk::raii::DeviceMemory sh_samples_mem_{nullptr};
    vk::raii::Buffer sh_probe_pos_buf_{nullptr};
    vk::raii::DeviceMemory sh_probe_pos_mem_{nullptr};
    vk::raii::Buffer sh_coeffs_buf_{nullptr};
    vk::raii::DeviceMemory sh_coeffs_mem_{nullptr};

    uint32_t sh_num_probes_ = 0;
    uint32_t sh_num_samples_ = 0;

    // --- GPU Culling + Indirect ---
    bool cull_resources_ready_ = false;

    // Input instances for this frame (enemies + bullets + dynamic sprites)
    std::vector<GpuSpriteInstance> current_instances_;

    // GPU buffers (host visible for simplicity, like SH bake)
    vk::raii::Buffer instance_buf_{nullptr};
    vk::raii::DeviceMemory instance_mem_{nullptr};
    vk::raii::Buffer visible_list_buf_{nullptr};   // compacted uint indices of visible
    vk::raii::DeviceMemory visible_list_mem_{nullptr};
    vk::raii::Buffer indirect_cmd_buf_{nullptr};   // VkDrawIndirectCommand (or more)
    vk::raii::DeviceMemory indirect_cmd_mem_{nullptr};
    vk::raii::Buffer count_buf_{nullptr};          // uint32 atomic count
    vk::raii::DeviceMemory count_mem_{nullptr};

    // Base unit quad for instanced drawing (local offsets)
    vk::raii::Buffer unit_quad_buf_{nullptr};
    vk::raii::DeviceMemory unit_quad_mem_{nullptr};
    std::uint32_t unit_quad_vertex_count_ = 6;

    // Cull compute
    std::optional<ShaderModule> shader_cull_;
    std::optional<vk::raii::DescriptorSetLayout> cull_desc_layout_;
    std::optional<vk::raii::DescriptorPool> cull_desc_pool_;
    std::optional<vk::raii::DescriptorSet> cull_desc_set_;
    std::optional<vk::raii::PipelineLayout> cull_pipeline_layout_;
    std::optional<vk::raii::Pipeline> cull_pipeline_;

    uint32_t last_instance_capacity_ = 0;
    uint32_t last_visible_capacity_ = 0;

    // 3D star mesh loaded from StarPoint.glb
    struct StarMeshData {
        vk::raii::Buffer vertexBuf{nullptr};
        vk::raii::DeviceMemory vertexMem{nullptr};
        vk::raii::Buffer indexBuf{nullptr};
        vk::raii::DeviceMemory indexMem{nullptr};
        uint32_t indexCount = 0;
        uint32_t vertexCount = 0;
    } starMesh_;

    // SSBO for star instances (pos + scale)
    vk::raii::Buffer starInstanceBuf_{nullptr};
    vk::raii::DeviceMemory starInstanceMem_{nullptr};
    uint32_t starInstanceCount_ = 0;
};

} // namespace kengine