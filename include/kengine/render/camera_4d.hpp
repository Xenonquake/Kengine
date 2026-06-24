#pragma once

#include "kengine/render/retro_pipeline.hpp"
#include "kengine/render/retro_types.hpp"
#include "kengine/math/vec4d.hpp"
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

    // 4D slice + full hyper-rotation matrix (SO(4))
    float w_slice = 0.0f;
    Mat4d4 hyper_rot = Mat4d4::identity();  // composable 4D rotations

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
    // Adjust a specific plane rotation (for legacy, composes on current)
    void adjust_hyper(int plane, float delta);    // plane 0..5 for xy,xz,xw,yz,yw,zw
    void reset();

    // 3D Star Fox style chase camera: positions eye behind/above ship, target ahead.
    // ship_pos/forward are in the projected 3D space. Call every frame.
    void chase_ship(const float ship_pos[3], const float ship_forward[3], float dt);

    // Frustum for GPU culling (in the final projected 3D space).
    // planes[6][4]: each [nx,ny,nz, d] satisfying dot(n, p) + d == 0 for points on plane.
    // Order: left, right, bottom, top, near, far. Planes point inward.
    struct FrustumPlanes {
        float planes[6][4];
    };

    // Extract 6 frustum planes from current view+proj (using the 3D mvp).
    void extract_frustum_planes(float aspect, FrustumPlanes& out) const;
};

} // namespace kengine
