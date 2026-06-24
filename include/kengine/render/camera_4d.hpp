#pragma once

#include "kengine/render/retro_pipeline.hpp"
#include "kengine/render/retro_types.hpp"
#include <cstdint>

namespace kengine {

// Basic 4D-aware camera. The 3D view/projection (mvp) is computed for the
// post-projection 3D space that the 4D->3D morph produces inside shaders.
// The 4D-specific parameters (w_slice, hyper_rot) are applied in-shader before projection.
struct Camera4D {
    // 3D eye in the projected space (after 4D->3D)
    float eye[3]   = {0.0f, 0.0f, 6.0f};
    float target[3]= {0.0f, 0.0f, 0.0f};
    float up[3]    = {0.0f, 1.0f, 0.0f};

    float fov    = 0.78539816339f; // ~45 deg
    float near_z = 0.1f;
    float far_z  = 200.0f;

    // 4D slice + hyperplane rotations (radians). Fed directly to shaders.
    float w_slice     = 0.0f;
    float hyper_rot[4] = {0.15f, 0.25f, 0.05f, 0.10f}; // xw, yw, xy, zw

    // Simple velocity for controls (integrated outside)
    float move_speed   = 4.0f;
    float rot_speed    = 1.2f;
    float slice_speed  = 0.8f;

    // Build column-major mvp into out_mvp[16] using the 3D projected eye.
    void compute_mvp(float aspect, float* out_mvp) const;

    // Fill the full RetroPushConstants using this camera + visual state + time/extent.
    // Existing shader uniforms/PC layout is unchanged.
    void fill_push_constants(RetroPushConstants& pc,
                             const RetroVisualState& state,
                             float time,
                             std::uint32_t width,
                             std::uint32_t height) const;

    // Basic control helpers (call from input loop with dt)
    void move(float dx, float dy, float dz);      // local strafe (x right, y up, z forward)
    void adjust_slice(float delta);
    void adjust_hyper(int index, float delta);    // index 0..3
    void reset();

    // 3D Star Fox style chase camera: positions eye behind/above ship, target ahead.
    // ship_pos/forward are in the projected 3D space. Call every frame.
    void chase_ship(const float ship_pos[3], const float ship_forward[3], float dt);
};

} // namespace kengine
