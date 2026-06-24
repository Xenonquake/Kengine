#pragma once

#include "kengine/core/job_system.hpp"
#include "kengine/ecs/entity.hpp"
#include "kengine/ecs/registry.hpp"
#include "kengine/lighting/spherical_harmonics.hpp"
#include "kengine/physics/physics_world.hpp"
#include "kengine/world/world.hpp"
#include "kengine/render/camera_4d.hpp"
#include "kengine/render/frame_graph.hpp"
#include "kengine/render/frame_renderer.hpp"
#include "kengine/render/pipeline_manager.hpp"
#include "kengine/render/post_process.hpp"
#include "kengine/render/retro_types.hpp"
#include "kengine/vulkan/context.hpp"
#include "kengine/vulkan/swapchain.hpp"
#include <filesystem>
#include <memory>

#ifndef KENGINE_CACHE_DIR
#define KENGINE_CACHE_DIR "build/cache"
#endif

namespace kengine {

struct EngineConfig {
    int window_width  = 1476;
    int window_height = 830;
    const char* title = "Kengine";
    bool vsync        = true;
    FrameGraphConfig frame_graph;
    std::filesystem::path pipeline_cache_path =
        std::filesystem::path(KENGINE_CACHE_DIR) / "kengine_pipeline_cache.bin";
};

class Engine {
public:
    explicit Engine(const EngineConfig& config = {});
    ~Engine();

    bool init();
    void run();
    void shutdown();

    Registry& registry() { return world_->registry(); }
    PhysicsWorld& physics() { return world_->physics(); }
    SHLightingSystem& lighting() { return *lighting_; }
    PipelineManager& pipelines() { return *pipelines_; }

    World& world() { return *world_; }

    void set_retro_style(RetroStyle style);

private:
    void update(float dt);
    void render();

    EngineConfig config_;
    std::shared_ptr<World> world_;
    std::shared_ptr<JobSystem> jobs_;
    std::unique_ptr<VulkanContext> vulkan_;
    std::unique_ptr<Swapchain> swapchain_;
    std::unique_ptr<SHLightingSystem> lighting_;
    std::unique_ptr<FrameGraph> frame_graph_;
    std::unique_ptr<PostProcessPipeline> post_process_;
    std::unique_ptr<PipelineManager> pipelines_;
    std::unique_ptr<FrameRenderer> frame_renderer_;
    RetroVisualState retro_state_;
    Camera4D camera_;
    float anim_time_ = 0.0f;
    bool running_ = false;

    // Demo: Space Invaders style ship at bottom + basic L/R/U/D controls
    float ship_x_ = 0.0f;
    float ship_y_ = -0.65f;
    float ship_speed_ = 1.4f;
    float ship_vel_x_ = 0.0f;
    float ship_vel_y_ = 0.0f;
    float ship_vel_z_ = 0.0f;
    float ship_z_ = 1.6f;
    float ship_w_ = 0.05f;

    // Player entity driven by 3D input -> physics velocity
    Entity player_entity_;
    int    player_body_index_ = -1;

    // AI drones that steer toward player
    std::vector<Entity> drones_;

    // Fire rate limiter for demo projectiles
    float fire_cooldown_ = 0.0f;

    // Active bullets (Phase 1)
    struct ActiveBullet {
        Entity entity;
        float lifetime = 1.1f;
    };
    std::vector<ActiveBullet> active_bullets_;
};

} // namespace kengine
