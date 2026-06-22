#include "kengine/app/engine.hpp"
#include "kengine/core/service_locator.hpp"
#include "kengine/render/camera_4d.hpp"
#include <GLFW/glfw3.h>
#include <cmath>
#include <cstdlib>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef KENGINE_CACHE_DIR
#define KENGINE_CACHE_DIR "build/cache"
#endif

namespace kengine {

static float demo_env_radiance(const float dir[3], void* /*userdata*/) {
    float sun = powf(fmaxf(0.0f, dir[1]), 32.0f);
    float sky = 0.3f + 0.7f * fmaxf(0.0f, dir[1]);
    return sun + sky;
}

Engine::Engine(const EngineConfig& config) : config_(config) {
    retro_state_.style = RetroStyle::GeometryCore;
}

Engine::~Engine() {
    shutdown();
}

bool Engine::init() {
    jobs_    = std::make_shared<JobSystem>();
    vulkan_  = std::make_unique<VulkanContext>(VulkanConfig{.vsync = config_.vsync});
    physics_ = std::make_unique<PhysicsWorld>();
    lighting_ = std::make_unique<SHLightingSystem>(16);

    ServiceLocator::register_instance<JobSystem>(jobs_);

    try {
        vulkan_->init_window(config_.window_width, config_.window_height, config_.title);
        vulkan_->init_vulkan();

        auto extent = vulkan_->framebuffer_size();
        swapchain_  = std::make_unique<Swapchain>(*vulkan_, extent);

        pipelines_ = std::make_unique<PipelineManager>(
            *vulkan_, *swapchain_, KENGINE_SHADER_DIR, config_.pipeline_cache_path);
        pipelines_->warmup(extent);

        post_process_ = std::make_unique<PostProcessPipeline>(*vulkan_, pipelines_->cache());
        post_process_->init(KENGINE_SHADER_DIR, swapchain_->image_format(),
                            vulkan_->frames_in_flight());
        post_process_->configure(config_.frame_graph);

        frame_graph_ = std::make_unique<FrameGraph>(config_.frame_graph);
        frame_graph_->build(extent.width, extent.height);

        frame_renderer_ = std::make_unique<FrameRenderer>(
            *vulkan_, *swapchain_, *pipelines_, *post_process_, *frame_graph_);
        frame_renderer_->bind_frame_graph_passes();

        lighting_->bake_all(demo_env_radiance, nullptr);

        auto entity = registry_.create();
        registry_.emplace<TransformComponent>(entity);
        auto& camera = registry_.emplace<CameraComponent>(entity);
        camera.primary = true;
        auto& retro = registry_.emplace<RetroVisualComponent>(entity);
        retro.visual = retro_state_;

        ke_phys_body body{};
        body.position = ke_vec4_make(0, 5, 0, 0);
        body.mass     = 1.0f;
        physics_->add_body(body);

        std::cout << "Kengine: basic Full HD offscreen + depth + pipeline cache ready\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Engine init failed: " << e.what() << '\n';
        return false;
    }
}

void Engine::set_retro_style(RetroStyle style) {
    retro_state_.style = style;
}

void Engine::run() {
    running_ = true;
    double last_time = glfwGetTime();

    while (running_ && !glfwWindowShouldClose(vulkan_->window())) {
        glfwPollEvents();

        double now = glfwGetTime();
        float dt   = static_cast<float>(now - last_time);
        last_time  = now;

        update(dt);
        render();
    }
}

void Engine::shutdown() {
    if (vulkan_) {
        vulkan_->device().waitIdle();
    }
    if (pipelines_) {
        pipelines_->save_cache();
    }
    frame_renderer_.reset();
    post_process_.reset();
    pipelines_.reset();
    frame_graph_.reset();
    swapchain_.reset();
    lighting_.reset();
    physics_.reset();
    if (vulkan_) vulkan_->shutdown();
    vulkan_.reset();
    jobs_.reset();
    ServiceLocator::unregister<JobSystem>();
}

void Engine::update(float dt) {
    anim_time_ += dt;

    GLFWwindow* win = vulkan_ ? vulkan_->window() : nullptr;

    // Fix camera for Space Invaders bottom view + flying stars (no 4D fwd/back)
    camera_.eye[0] = 0.0f;
    camera_.eye[1] = 0.15f;
    camera_.eye[2] = 3.2f;
    camera_.target[0] = 0.0f;
    camera_.target[1] = -0.3f;
    camera_.target[2] = 0.0f;
    camera_.w_slice = 0.0f;
    std::fill(std::begin(camera_.hyper_rot), std::end(camera_.hyper_rot), 0.0f);

    // Basic left/right/up/down ship controls (Space Invaders style at bottom)
    if (win) {
        float move = ship_speed_ * dt;
        if (glfwGetKey(win, GLFW_KEY_LEFT) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) ship_x_ -= move;
        if (glfwGetKey(win, GLFW_KEY_RIGHT) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) ship_x_ += move;
        if (glfwGetKey(win, GLFW_KEY_UP) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) ship_y_ += move * 0.6f;
        if (glfwGetKey(win, GLFW_KEY_DOWN) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) ship_y_ -= move * 0.6f;

        // clamp roughly to view
        ship_x_ = std::max(-0.82f, std::min(0.82f, ship_x_));
        ship_y_ = std::max(-0.82f, std::min(-0.25f, ship_y_));
    }

    retro_state_.w_morph = 0.15f;  // mild perspective for depth feel, no heavy 4D
    retro_state_.w_slice = 0.0f;

    physics_->step(dt);

    registry_.view<RetroVisualComponent>([this](Entity /*e*/, RetroVisualComponent& r) {
        r.visual = retro_state_;
    });
}

void Engine::render() {
    auto extent = vulkan_->framebuffer_size();
    if (extent.width == 0 || extent.height == 0) return;

    if (frame_renderer_->needs_resize(extent)) {
        vulkan_->device().waitIdle();
        swapchain_->recreate(extent);
        pipelines_->recreate(extent);
        post_process_->recreate(swapchain_->image_format());
        frame_renderer_->on_resize(extent);
    }

    frame_renderer_->set_ship_position(ship_x_, ship_y_);
    frame_renderer_->render_frame(anim_time_, retro_state_, camera_);
}

} // namespace kengine