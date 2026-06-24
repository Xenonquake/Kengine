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
    world_   = std::make_shared<World>();
    lighting_ = std::make_unique<SHLightingSystem>(16);

    // Player entity (kinematic for responsive 3D arcade controls)
    {
        player_entity_ = world_->registry().create();
        world_->registry().emplace<TransformComponent>(player_entity_);
        if (auto* tc = world_->registry().get<TransformComponent>(player_entity_)) {
            tc->position[0] = 0.0f;
            tc->position[1] = -0.55f;   // near classic ship start area
            tc->position[2] = 0.0f;
            tc->position[3] = 0.05f;
        }
        auto& pc = world_->registry().emplace<PhysicsComponent>(player_entity_);
        pc.mass = 1.0f;
        pc.kinematic = true;            // velocity directly driven by input
        ke_phys_body b{};
        if (auto* tc = world_->registry().get<TransformComponent>(player_entity_)) {
            b.position = ke_vec4_make(tc->position[0], tc->position[1], tc->position[2], tc->position[3]);
        }
        b.mass = pc.mass;
        player_body_index_ = world_->physics().create_body(b);
        pc.body_index = player_body_index_;
    }

    // Spawn 5-10 physics bodies for demo (player + enemies/pickups). They use Verlet + spatial hash queries for interactions.
    for (int i = 0; i < 7; ++i) {
        auto de = world_->registry().create();
        world_->registry().emplace<TransformComponent>(de);
        if (auto* tc = world_->registry().get<TransformComponent>(de)) {
            tc->position[0] = -2.0f + (i % 5) * 0.9f;
            tc->position[1] =  0.6f + (i / 3) * 0.8f - (i % 2) * 0.4f;
            tc->position[2] =  0.4f * ((i % 3) - 1);
            tc->position[3] =  0.05f * (i - 3);
        }
        auto& pc = world_->registry().emplace<PhysicsComponent>(de);
        pc.mass = 0.6f + 0.1f * (i % 3);
        pc.kinematic = false;
        ke_phys_body b{};
        if (auto* tc = world_->registry().get<TransformComponent>(de)) {
            b.position = ke_vec4_make(tc->position[0], tc->position[1], tc->position[2], tc->position[3]);
            float ivx = 1.2f * sinf(i * 1.3f);
            float ivy = -0.6f + (i % 3) * 0.3f;
            float ivz = 0.8f * cosf(i * 0.7f);
            b.velocity = ke_vec4_make(ivx, ivy, ivz, 0.1f * (i-3));
        }
        b.mass = pc.mass;
        pc.body_index = world_->physics().create_body(b);
        drones_.push_back(de);
    }

    ServiceLocator::register_instance<JobSystem>(jobs_);
    ServiceLocator::register_instance<World>(world_);

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

        auto entity = world_->registry().create();
        world_->registry().emplace<TransformComponent>(entity);
        auto& camera = world_->registry().emplace<CameraComponent>(entity);
        camera.primary = true;
        auto& retro = world_->registry().emplace<RetroVisualComponent>(entity);
        retro.visual = retro_state_;

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

        // Minimal playable loop (via World + systems in update):
        //   input (forces/vel on kinematic bodies) →
        //   physics step (C hot path) →
        //   sync transforms (ECS) →
        //   render with current Camera4D (w_slice, hyper_rot, morph)
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
    world_.reset();
    if (vulkan_) vulkan_->shutdown();
    vulkan_.reset();
    jobs_.reset();
    ServiceLocator::unregister<JobSystem>();
    ServiceLocator::unregister<World>();
}

void Engine::update(float dt) {
    anim_time_ += dt;

    GLFWwindow* win = vulkan_ ? vulkan_->window() : nullptr;

    // One-time starting camera for the ship/arena view. User can fly camera with IJKL+U/O.
    static bool camera_set = false;
    if (!camera_set) {
        camera_.eye[0]   = 0.0f;
        camera_.eye[1]   = 0.15f;
        camera_.eye[2]   = 3.2f;
        camera_.target[0]= 0.0f;
        camera_.target[1]= -0.3f;
        camera_.target[2]= 0.0f;
        camera_.w_slice  = 0.0f;
        std::fill(std::begin(camera_.hyper_rot), std::end(camera_.hyper_rot), 0.0f);
        camera_set = true;
    }

    retro_state_.w_morph = 0.15f;  // mild perspective for depth feel, no heavy 4D
    retro_state_.w_slice = 0.0f;

    // ------------------------------------------------------------------
    // Player 3D movement: WASD (xy) + QE (z) + RF (w)  --> set kinematic vel
    // ------------------------------------------------------------------
    if (win && player_entity_.valid()) {
        if (auto* pc = world_->registry().get<PhysicsComponent>(player_entity_)) {
            if (pc->kinematic) {
                const float ps = 3.8f;  // player speed (units/sec)
                float vx = 0, vy = 0, vz = 0, vw = 0;

                bool left  = (glfwGetKey(win, GLFW_KEY_LEFT ) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS);
                bool right = (glfwGetKey(win, GLFW_KEY_RIGHT) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS);
                bool up    = (glfwGetKey(win, GLFW_KEY_UP   ) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS);
                bool down  = (glfwGetKey(win, GLFW_KEY_DOWN ) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS);

                if (left)  vx -= ps;
                if (right) vx += ps;
                if (up)    vy += ps * 0.75f;
                if (down)  vy -= ps * 0.75f;

                // Z (depth/height) thrust
                if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) vz -= ps * 0.7f;
                if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) vz += ps * 0.7f;

                // W-dimension (4th) thrust
                if (glfwGetKey(win, GLFW_KEY_R) == GLFW_PRESS) vw -= ps * 0.5f;
                if (glfwGetKey(win, GLFW_KEY_F) == GLFW_PRESS) vw += ps * 0.5f;

                pc->velocity[0] = vx;
                pc->velocity[1] = vy;
                pc->velocity[2] = vz;
                pc->velocity[3] = vw;
            }
        }
    }

    // Demo fire (space) - spawns fast kinematic projectiles
    fire_cooldown_ -= dt;
    if (win && glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS && fire_cooldown_ <= 0.0f && player_entity_.valid()) {
        if (auto* ptc = world_->registry().get<TransformComponent>(player_entity_)) {
            if (auto* ppc = world_->registry().get<PhysicsComponent>(player_entity_)) {
                auto proj = world_->registry().create();
                auto& pt = world_->registry().emplace<TransformComponent>(proj);
                pt.position[0] = ptc->position[0];
                pt.position[1] = ptc->position[1] + 0.12f;
                pt.position[2] = ptc->position[2];
                pt.position[3] = ptc->position[3];

                auto& pc = world_->registry().emplace<PhysicsComponent>(proj);
                pc.mass = 0.08f;
                pc.kinematic = true;

                ke_phys_body pb{};
                pb.position = ke_vec4_make(pt.position[0], pt.position[1], pt.position[2], pt.position[3]);
                float bs = 11.0f;
                // inherit some player vel + strong forward bias in view
                pb.velocity = ke_vec4_make(
                    ppc->velocity[0] * 0.3f,
                    (ppc->velocity[1] < 0.5f ? -bs : ppc->velocity[1] * 0.8f),
                    ppc->velocity[2] * 0.4f + 1.5f,
                    ppc->velocity[3] * 0.2f
                );
                pb.mass = pc.mass;
                pc.body_index = world_->physics().create_body(pb);
                fire_cooldown_ = 0.09f;
            }
        }
    }

    // Runtime gravity control (1/2 for y-grav slice feel, 3/4/5 for w gravity)
    if (win) {
        if (auto* raw = world_->physics().raw()) {
            if (glfwGetKey(win, GLFW_KEY_1) == GLFW_PRESS) { raw->gravity[1] = -6.5f; raw->gravity[3] = 0.0f; }
            if (glfwGetKey(win, GLFW_KEY_2) == GLFW_PRESS) { raw->gravity[1] =  0.0f; raw->gravity[3] = 0.0f; }
            if (glfwGetKey(win, GLFW_KEY_3) == GLFW_PRESS) raw->gravity[3] =  0.9f;
            if (glfwGetKey(win, GLFW_KEY_4) == GLFW_PRESS) raw->gravity[3] = -0.9f;
            if (glfwGetKey(win, GLFW_KEY_5) == GLFW_PRESS) raw->gravity[3] =  0.0f;
        }
    }

    // Camera free-flight controls (fly through the 3D/4D scene)
    if (win) {
        float cms = camera_.move_speed * dt;
        if (glfwGetKey(win, GLFW_KEY_J) == GLFW_PRESS) camera_.move(-cms, 0, 0);
        if (glfwGetKey(win, GLFW_KEY_L) == GLFW_PRESS) camera_.move( cms, 0, 0);
        if (glfwGetKey(win, GLFW_KEY_I) == GLFW_PRESS) camera_.move(0, 0,  cms);
        if (glfwGetKey(win, GLFW_KEY_K) == GLFW_PRESS) camera_.move(0, 0, -cms);
        if (glfwGetKey(win, GLFW_KEY_U) == GLFW_PRESS) camera_.move(0,  cms, 0);
        if (glfwGetKey(win, GLFW_KEY_O) == GLFW_PRESS) camera_.move(0, -cms, 0);

        // Live 4D slice / rotation scrubbing (reveals structure while objects move in 3D)
        float sms = camera_.slice_speed * dt;
        if (glfwGetKey(win, GLFW_KEY_Z) == GLFW_PRESS) camera_.adjust_slice(-sms);
        if (glfwGetKey(win, GLFW_KEY_X) == GLFW_PRESS) camera_.adjust_slice( sms);
        if (glfwGetKey(win, GLFW_KEY_7) == GLFW_PRESS) camera_.adjust_hyper(0, -sms);
        if (glfwGetKey(win, GLFW_KEY_8) == GLFW_PRESS) camera_.adjust_hyper(0,  sms);
        if (glfwGetKey(win, GLFW_KEY_9) == GLFW_PRESS) camera_.adjust_hyper(2, -sms * 0.6f);
        if (glfwGetKey(win, GLFW_KEY_0) == GLFW_PRESS) camera_.adjust_hyper(2,  sms * 0.6f);
    }

    // --- Playable loop core (input already handled above for player vel/forces) ---
    // 2. Physics step (C hotpath in World) + 3. sync back to ECS Transforms
    world_->registry().view<PhysicsComponent>([this](Entity e, PhysicsComponent& pc) {
        if (pc.body_index < 0) return;
        ke_phys_body* body = world_->physics().get_body(pc.body_index);
        if (!body) return;
        if (auto* tc = world_->registry().get<TransformComponent>(e)) {
            body->position = ke_vec4_make(tc->position[0], tc->position[1], tc->position[2], tc->position[3]);
            if (pc.kinematic) {
                body->velocity = ke_vec4_make(pc.velocity[0], pc.velocity[1], pc.velocity[2], pc.velocity[3]);
            }
        }
    });

    world_->step_physics(dt);  // C hot path + spatial rebuild

    // Post-step: physics body -> TransformComponent (for rendering) + mirror velocity
    world_->registry().view<PhysicsComponent>([this](Entity e, PhysicsComponent& pc) {
        if (pc.body_index < 0) return;
        ke_phys_body* body = world_->physics().get_body(pc.body_index);
        if (!body) return;
        if (auto* tc = world_->registry().get<TransformComponent>(e)) {
            tc->position[0] = body->position.x;
            tc->position[1] = body->position.y;
            tc->position[2] = body->position.z;
            tc->position[3] = body->position.w;
        }
        pc.velocity[0] = body->velocity.x;
        pc.velocity[1] = body->velocity.y;
        pc.velocity[2] = body->velocity.z;
        pc.velocity[3] = body->velocity.w;
    });

    // Drive retro ship visual from the player physics entity's position
    if (player_entity_.valid()) {
        if (auto* tc = world_->registry().get<TransformComponent>(player_entity_)) {
            ship_x_ = tc->position[0];
            ship_y_ = tc->position[1];
            // (z/w of player could influence ship_z / ship_w in a future visual pass)
        }
        if (auto* pc = world_->registry().get<PhysicsComponent>(player_entity_)) {
            ship_vel_x_ = pc->velocity[0];
            ship_vel_y_ = pc->velocity[1];
            ship_vel_z_ = pc->velocity[2];
        }
        if (auto* tc = world_->registry().get<TransformComponent>(player_entity_)) {
            ship_z_ = tc->position[2] + 1.6f;  // visual offset + player z
            ship_w_ = tc->position[3];
        }
    }

    // Camera follow mode: track player body's position in the projected 3D space
    if (player_entity_.valid()) {
        if (auto* tc = world_->registry().get<TransformComponent>(player_entity_)) {
            float follow = 0.15f;
            camera_.target[0] = (1.0f - follow) * camera_.target[0] + follow * tc->position[0];
            camera_.target[1] = (1.0f - follow) * camera_.target[1] + follow * (tc->position[1] - 0.25f);
            camera_.target[2] = (1.0f - follow) * camera_.target[2] + follow * tc->position[2];
            // pull camera eye to follow from behind/side
            camera_.eye[2] = (1.0f - follow * 0.6f) * camera_.eye[2] + (follow * 0.6f) * (tc->position[2] + 3.2f);
        }
    }

    // Simple 3D steering for drones (xyz homing + Verlet/damping does the motion)
    if (player_entity_.valid()) {
        if (auto* ptc = world_->registry().get<TransformComponent>(player_entity_)) {
            ke_vec3 ppos = ke_vec3_make(ptc->position[0], ptc->position[1], ptc->position[2]);
            for (auto& de : drones_) {
                auto* dpc = world_->registry().get<PhysicsComponent>(de);
                if (!dpc || dpc->body_index < 0 || dpc->kinematic) continue;
                auto* body = world_->physics().get_body(dpc->body_index);
                if (!body) continue;

                ke_vec3 epos = ke_vec3_make(body->position.x, body->position.y, body->position.z);
                ke_vec3 delta = ke_vec3_sub(ppos, epos);
                if (ke_vec3_dot(delta, delta) > 0.0001f) {
                    ke_vec3 dir = ke_vec3_normalize(delta);
                    const float steer = 2.6f;
                    // exponential lerp on velocity (xy z), w left alone
                    body->velocity.x = body->velocity.x * 0.82f + dir.x * steer;
                    body->velocity.y = body->velocity.y * 0.82f + dir.y * steer;
                    body->velocity.z = body->velocity.z * 0.82f + dir.z * steer;

                    // soft speed cap in 3D
                    float sp2 = body->velocity.x*body->velocity.x + body->velocity.y*body->velocity.y + body->velocity.z*body->velocity.z;
                    if (sp2 > 36.0f) {
                        float s = 6.0f / std::sqrt(sp2);
                        body->velocity.x *= s;
                        body->velocity.y *= s;
                        body->velocity.z *= s;
                    }
                }
            }
        }
    }

    // Leverage the spatial hash (3D xyz broadphase) — example: simple separation for bodies near player
    if (player_entity_.valid()) {
        if (auto* ptc = world_->registry().get<TransformComponent>(player_entity_)) {
            float px = ptc->position[0], py = ptc->position[1], pz = ptc->position[2];
            int32_t cands[24];
            int nc = world_->physics().query_radius(px, py, pz, 2.2f, cands, 24);
            for (int i = 0; i < nc; ++i) {
                int bi = cands[i];
                if (bi == player_body_index_) continue;
                auto* b = world_->physics().get_body(bi);
                if (!b) continue;
                float dx = b->position.x - px;
                float dy = b->position.y - py;
                float dz = b->position.z - pz;
                float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 > 0.0001f && d2 < 1.6f * 1.6f) {
                    float inv = 1.0f / sqrtf(d2);
                    float sep = 0.8f;
                    // push this body away (affects next step via verlet)
                    b->velocity.x += dx * inv * sep;
                    b->velocity.y += dy * inv * sep;
                    b->velocity.z += dz * inv * sep;
                }
            }
        }
    }

    // Environment: simple arena bounds + bounce/clamp (xyz walls, soft w limits)
    // Applied post-physics so Verlet + hash interactions still feel natural.
    world_->registry().view<PhysicsComponent>([this](Entity /*e*/, PhysicsComponent& pc) {
        if (pc.body_index < 0) return;
        auto* b = world_->physics().get_body(pc.body_index);
        if (!b) return;
        const float limx = 2.6f, limy = 2.0f, limz = 1.4f;
        auto reflect = [](float& p, float& v, float lim, float damp = 0.65f) {
            if (p > lim) { p = lim; v = -fabs(v) * damp; }
            else if (p < -lim) { p = -lim; v = fabs(v) * damp; }
        };
        reflect(b->position.x, b->velocity.x, limx);
        reflect(b->position.y, b->velocity.y, limy);
        reflect(b->position.z, b->velocity.z, limz);
        // w soft limits (4D slice feel)
        if (b->position.w > 1.5f) { b->position.w = 1.5f; if (!pc.kinematic) b->velocity.w *= 0.4f; }
        if (b->position.w < -1.5f) { b->position.w = -1.5f; if (!pc.kinematic) b->velocity.w *= 0.4f; }
    });

    world_->registry().view<RetroVisualComponent>([this](Entity /*e*/, RetroVisualComponent& r) {
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
    frame_renderer_->set_ship_velocity(ship_vel_x_, ship_vel_y_, ship_vel_z_);
    frame_renderer_->set_ship_zw(ship_z_, ship_w_);
    frame_renderer_->render_frame(anim_time_, retro_state_, camera_);
}

} // namespace kengine