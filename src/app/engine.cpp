#include "kengine/app/engine.hpp"
#include "kengine/core/service_locator.hpp"
#include "kengine/game/game_entity.hpp"
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
    // Setup a coarse 3D light probe grid across the play space for runtime spatial interpolation
    {
        float origin[3] = {-6.0f, -4.0f, -4.0f};
        float spacing[3] = {4.0f, 4.0f, 4.0f};
        lighting_->setup_probe_grid(4, 2, 2, origin, spacing);
    }

    // Early bake + verify for grid + spatial SH (CPU path)
    lighting_->bake_all(demo_env_radiance, nullptr);
    {
        float tp[3] = {0.f, 0.f, 0.f}; float tn[3]={0,1,0};
        auto v = lighting_->evaluate_at(tp, tn);
        std::cout << "[SH] early after-bake grid interp -> " << v[0] << " " << v[1] << " " << v[2] << "\n";
        // host proj ref
        lighting_->prepare_samples(8,16);
        const auto& sm = lighting_->samples();
        if(!sm.empty()){
            float tc[9]={}; 
            for(auto& s:sm){ float y=s.direction[1]; float L=0.3f+0.7f*std::max(0.f,y)+std::pow(std::max(0.f,y),32.f);
                float x=s.direction[0],yy=s.direction[1],z=s.direction[2];
                float b[9]; b[0]=0.28209479f; b[1]=0.48860251f*z; b[2]=0.48860251f*yy; b[3]=0.48860251f*x;
                b[4]=1.09254843f*x*z; b[5]=1.09254843f*z*yy; b[6]=0.31539157f*(3*yy*yy-1); b[7]=1.09254843f*x*yy; b[8]=0.54627422f*(x*x-z*z);
                for(int k=0;k<9;++k) tc[k] += L * b[k] * s.weight;
            }
            std::cout << "[SH] host-proj-ref c0=" << tc[0] << " c6=" << tc[6] << " (GPU sh_bake.comp target)\n";
        }
    }

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
            *vulkan_, *swapchain_, *pipelines_, *post_process_, *frame_graph_, KENGINE_SHADER_DIR);
        frame_renderer_->bind_frame_graph_passes();

        // Wire GPU SH resources (probe grid already configured); enables compute bake dispatch
        frame_renderer_->setup_sh_lighting(*lighting_);

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

    // Clean exit on ESC for graceful shutdown
    if (win && glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        running_ = false;
        glfwSetWindowShouldClose(win, GLFW_TRUE);
        return;  // skip remaining update logic
    }

    // Defaults for 4D params. Eye/target driven by chase_ship; hyper_rot is now full Mat4d4
    static bool camera_set = false;
    if (!camera_set) {
        camera_.w_slice  = 0.0f;
        camera_.hyper_rot = Mat4d4::identity();
        camera_set = true;
    }

    retro_state_.w_morph = 0.15f;  // mild perspective for depth feel, no heavy 4D
    retro_state_.w_slice = 0.0f;

    // ------------------------------------------------------------------
    // Locked ship-relative controls (tied to ship forward, no free off-axis flip)
    // WASD: strafe left/right + up/down in ship's local plane (using persistent forward)
    // QE: forward/back thrust along ship's forward
    // RF: minor w-layer (locked small for arcade feel)
    // Camera is rigidly chased to ship (see chase_ship)
    // ------------------------------------------------------------------
    if (win && player_entity_.valid()) {
        if (auto* pc = world_->registry().get<PhysicsComponent>(player_entity_)) {
            if (pc->kinematic) {
                const float ps = 3.8f;  // player speed (units/sec)
                float wvx = 0, wvy = 0, wvz = 0, wvw = 0;

                // Build local basis from ship's persistent forward (locked, no flip)
                float fwd[3] = {ship_forward_[0], ship_forward_[1], ship_forward_[2]};
                float up[3] = {0, 1, 0};
                float rgt[3] = {
                    fwd[1]*up[2] - fwd[2]*up[1],
                    fwd[2]*up[0] - fwd[0]*up[2],
                    fwd[0]*up[1] - fwd[1]*up[0]
                };
                float rlen = sqrtf(rgt[0]*rgt[0] + rgt[1]*rgt[1] + rgt[2]*rgt[2]);
                if (rlen > 0.001f) {
                    rgt[0] /= rlen; rgt[1] /= rlen; rgt[2] /= rlen;
                } else {
                    rgt[0] = 1; rgt[1] = 0; rgt[2] = 0;
                }
                float shp_up[3] = {
                    rgt[1]*fwd[2] - rgt[2]*fwd[1],
                    rgt[2]*fwd[0] - rgt[0]*fwd[2],
                    rgt[0]*fwd[1] - rgt[1]*fwd[0]
                };

                bool left  = (glfwGetKey(win, GLFW_KEY_LEFT ) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS);
                bool right = (glfwGetKey(win, GLFW_KEY_RIGHT) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS);
                bool upk   = (glfwGetKey(win, GLFW_KEY_UP   ) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS);
                bool down  = (glfwGetKey(win, GLFW_KEY_DOWN ) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS);

                // Horizontal strafe along local right (tied to ship)
                if (left)  { wvx -= rgt[0] * ps; wvy -= rgt[1] * ps; wvz -= rgt[2] * ps; }
                if (right) { wvx += rgt[0] * ps; wvy += rgt[1] * ps; wvz += rgt[2] * ps; }

                // Vertical strafe along ship's local up (locked to ship, no world up flip)
                if (upk)   { wvx += shp_up[0] * ps * 0.7f; wvy += shp_up[1] * ps * 0.7f; wvz += shp_up[2] * ps * 0.7f; }
                if (down)  { wvx -= shp_up[0] * ps * 0.7f; wvy -= shp_up[1] * ps * 0.7f; wvz -= shp_up[2] * ps * 0.7f; }

                // Forward thrust along ship's forward (tied to ship)
                bool fwd_thrust = (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS);
                bool back_thrust = (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS);
                if (fwd_thrust)  { wvx += fwd[0] * ps; wvy += fwd[1] * ps; wvz += fwd[2] * ps; }
                if (back_thrust) { wvx -= fwd[0] * ps; wvy -= fwd[1] * ps; wvz -= fwd[2] * ps; }

                // Minor w (4th dim) - locked small, not erratic
                bool w_neg = (glfwGetKey(win, GLFW_KEY_R) == GLFW_PRESS);
                bool w_pos = (glfwGetKey(win, GLFW_KEY_F) == GLFW_PRESS);
                if (w_neg) wvw -= ps * 0.3f;
                if (w_pos) wvw += ps * 0.3f;

                pc->velocity[0] = wvx;
                pc->velocity[1] = wvy;
                pc->velocity[2] = wvz;
                pc->velocity[3] = wvw;
            }
        }
    }

    // === BULLET SPAWNING (Phase 1) ===
    fire_cooldown_ -= dt;
    if (win && glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS && fire_cooldown_ <= 0.0f && player_entity_.valid()) {
        if (auto* ptc = world_->registry().get<TransformComponent>(player_entity_)) {
            if (auto* ppc = world_->registry().get<PhysicsComponent>(player_entity_)) {
                auto proj = world_->registry().create();
                auto& pt = world_->registry().emplace<TransformComponent>(proj);

                // Spawn slightly "ahead" of the ship nose (visual forward is negative screen y in classic layout)
                float nose_offset = 0.18f;
                pt.position[0] = ptc->position[0];
                pt.position[1] = ptc->position[1] - nose_offset;  // forward in screen
                pt.position[2] = ptc->position[2];
                pt.position[3] = ptc->position[3];

                auto& pc = world_->registry().emplace<PhysicsComponent>(proj);
                pc.mass = 0.06f;
                pc.kinematic = true;

                ke_phys_body pb{};
                pb.position = ke_vec4_make(pt.position[0], pt.position[1], pt.position[2], pt.position[3]);

                float bs = 13.5f;
                // Inherit some player velocity + strong forward component
                pb.velocity = ke_vec4_make(
                    ppc->velocity[0] * 0.25f,
                    (ppc->velocity[1] > 1.0f ? ppc->velocity[1] * 0.6f : -bs),
                    ppc->velocity[2] * 0.3f + 2.2f,
                    ppc->velocity[3] * 0.15f
                );
                pb.mass = pc.mass;
                pc.body_index = world_->physics().create_body(pb);

                active_bullets_.push_back({proj, 1.15f});
                fire_cooldown_ = 0.085f;

                // Populate GameEntity prototype for bullets
                GameEntity bg = {};
                bg.pos = ke_vec4_make(pt.position[0], pt.position[1], pt.position[2], pt.position[3]);
                bg.vel = pb.velocity;
                bg.type = 2;
                bg.active = true;
                bg.lifetime = 1.15f;
                bg.scale = 0.5f;
                bg.color = 0xFFAAFFEEu;
                frame_renderer_->add_bullet(bg);
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

    // 4D visual field locked for strict arcade shooter (no erratic free manipulation)
    // Subtle fixed 4D for retro feel; background stars still use flow for dynamism
    camera_.w_slice = 0.0f;
    camera_.hyper_rot = Mat4d4::identity();
    // (Previously free keys Z/X/7/0 etc. removed to lock to ship and prevent flipping/erratic 4D)

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
            ship_z_ = tc->position[2];   // use actual physics position for 3D model
            ship_w_ = tc->position[3];
        }
        if (auto* pc = world_->registry().get<PhysicsComponent>(player_entity_)) {
            ship_vel_x_ = pc->velocity[0];
            ship_vel_y_ = pc->velocity[1];
            ship_vel_z_ = pc->velocity[2];
        }

        // Feed ship velocity into the starfield so background reacts (Phase 1 feel)
        frame_renderer_->update_background_flow(ship_vel_x_, ship_vel_y_, ship_vel_z_, dt);
    }

    // Locked 3rd person perspective: Star Fox style 3D chase cam.
    // Camera manipulates relative to the ship in 3D env (no world rotation via cam).
    // 4D effects (slice/hyper) are for visual flair on background/special elements.
    if (player_entity_.valid()) {
        if (auto* tc = world_->registry().get<TransformComponent>(player_entity_)) {
            float ship_pos[3] = {tc->position[0], tc->position[1], tc->position[2]};
            // Use persistent forward for locked orientation (updated from previous vel)
            camera_.chase_ship(ship_pos, ship_forward_, dt);
        }
    }

    // Update persistent ship forward from current velocity for locked ship-tied controls next frame
    // This prevents flipping off axis by smoothing and using consistent direction
    if (player_entity_.valid()) {
        if (auto* ppc = world_->registry().get<PhysicsComponent>(player_entity_)) {
            float vx = ppc->velocity[0];
            float vy = ppc->velocity[1];
            float vz = ppc->velocity[2];
            float len = sqrtf(vx*vx + vy*vy + vz*vz);
            if (len > 0.1f) {
                float nf[3] = {vx/len, vy/len, vz/len};
                float t = 0.25f;  // smoothing factor
                for(int i = 0; i < 3; i++) {
                    ship_forward_[i] = ship_forward_[i] * (1.0f - t) + nf[i] * t;
                }
            }
            // normalize
            float fl = sqrtf(ship_forward_[0]*ship_forward_[0] + ship_forward_[1]*ship_forward_[1] + ship_forward_[2]*ship_forward_[2]);
            if (fl > 0.001f) {
                for(int i = 0; i < 3; i++) ship_forward_[i] /= fl;
            }
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
        const float limx = 8.0f, limy = 6.0f, limz = 5.0f;  // wider for scaled space / running track feel
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

    // === Bullet lifetime + despawn + collect for renderer ===
    std::vector<FrameRenderer::BulletVisual> live_bullet_vis;
    for (auto it = active_bullets_.begin(); it != active_bullets_.end(); ) {
        it->lifetime -= dt;

        if (auto* btc = world_->registry().get<TransformComponent>(it->entity)) {
            if (auto* bpc = world_->registry().get<PhysicsComponent>(it->entity)) {
                if (auto* bb = world_->physics().get_body(bpc->body_index)) {
                    // sync latest
                    btc->position[0] = bb->position.x;
                    btc->position[1] = bb->position.y;
                    btc->position[2] = bb->position.z;
                    btc->position[3] = bb->position.w;

                    // collect visual (small fast neon)
                    live_bullet_vis.push_back({
                        btc->position[0], btc->position[1], btc->position[2], btc->position[3],
                        bb->velocity.x, bb->velocity.y, bb->velocity.z,
                        0.055f,
                        0xFFAAFFEEu
                    });

                    // Kill bullets that fly too far (screen/enviro)
                    if (std::abs(bb->position.x) > 3.8f ||
                        std::abs(bb->position.y) > 3.5f ||
                        std::abs(bb->position.z) > 3.0f) {
                        it->lifetime = 0.0f;
                    }
                }
            }
        }

        if (it->lifetime <= 0.0f) {
            // simple despawn (entity will be GC'd on next registry cleanup if we add one; for now just drop)
            it = active_bullets_.erase(it);
        } else {
            ++it;
        }
    }

    frame_renderer_->set_bullets(live_bullet_vis);

    // === Collect enemies for rendering (Galaga formations prototype) ===
    std::vector<FrameRenderer::EnemyVisual> enemy_vis;
    for (auto& de : drones_) {
        if (auto* dtc = world_->registry().get<TransformComponent>(de)) {
            enemy_vis.push_back({
                dtc->position[0], dtc->position[1], dtc->position[2], dtc->position[3],
                0.0f, // yaw for future
                1.0f,
                0xFF88FF88u // green enemies
            });
        }
    }
    frame_renderer_->set_enemies(enemy_vis);

    // (enemies_game managed in renderer prototype for independent movement demo; ECS drones for other systems)

    // === Very basic collision (bullets vs drones) using spatial hash broadphase ===
    for (auto& ab : active_bullets_) {
        if (ab.lifetime <= 0.1f) continue; // dying

        auto* btc = world_->registry().get<TransformComponent>(ab.entity);
        if (!btc) continue;

        int32_t cands[8];
        int nc = world_->physics().query_radius(btc->position[0], btc->position[1], btc->position[2], 0.35f, cands, 8);

        for (int ci = 0; ci < nc; ++ci) {
            int bi = cands[ci];
            // Skip player and the bullet itself
            if (bi == player_body_index_) continue;

            // Check if this body belongs to one of our drones
            bool is_drone = false;
            for (auto& de : drones_) {
                if (auto* dpc = world_->registry().get<PhysicsComponent>(de)) {
                    if (dpc->body_index == bi) {
                        is_drone = true;
                        break;
                    }
                }
            }
            if (is_drone) {
                // Hit! Simple: kill the drone entity (remove from list) and the bullet
                // For visual pop we just let lifetime kill the bullet
                ab.lifetime = 0.01f;

                // Remove the drone
                for (auto dit = drones_.begin(); dit != drones_.end(); ++dit) {
                    if (auto* dpc = world_->registry().get<PhysicsComponent>(*dit)) {
                        if (dpc->body_index == bi) {
                            // Destroy the drone entity (components will be cleaned on registry side if we add destroy)
                            world_->registry().destroy(*dit);
                            dit = drones_.erase(dit);
                            break;
                        }
                    }
                }
                break;
            }
        }
    }

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