#include "kengine/render/camera_4d.hpp"
#include "mat4.h"
#include <cstring>
#include <cmath>

namespace kengine {

void Camera4D::compute_mvp(float aspect, float* out_mvp) const {
    ke_mat4 proj = ke_mat4_perspective(fov, aspect, near_z, far_z);

    ke_vec3 e = ke_vec3_make(eye[0], eye[1], eye[2]);
    ke_vec3 t = ke_vec3_make(target[0], target[1], target[2]);
    ke_vec3 u = ke_vec3_make(up[0], up[1], up[2]);
    ke_mat4 view = ke_mat4_look_at(e, t, u);

    ke_mat4 mvp = ke_mat4_mul(proj, view);
    std::memcpy(out_mvp, mvp.m, sizeof(float) * 16);
}

void Camera4D::fill_push_constants(RetroPushConstants& pc,
                                   const RetroVisualState& state,
                                   float time,
                                   std::uint32_t width,
                                   std::uint32_t height) const {
    float aspect = (height > 0) ? (static_cast<float>(width) / static_cast<float>(height)) : 1.0f;
    compute_mvp(aspect, pc.mvp);

    pc.w_slice            = w_slice;
    pc.w_morph            = state.w_morph;
    pc.glow_intensity     = state.glow_intensity;
    pc.time               = time;
    pc.hyper_rot[0]       = hyper_rot[0];
    pc.hyper_rot[1]       = hyper_rot[1];
    pc.hyper_rot[2]       = hyper_rot[2];
    pc.hyper_rot[3]       = hyper_rot[3];
    pc.viewport[0]        = static_cast<float>(width);
    pc.viewport[1]        = static_cast<float>(height);
    pc.scanline_strength  = state.scanline_strength;
    pc.pixel_snap         = state.pixel_snap;
    pc.palette_index      = static_cast<float>(state.style);
}

void Camera4D::move(float dx, float dy, float dz) {
    // Build a simple local basis from current look dir (target-eye approx)
    float lx = target[0] - eye[0];
    float ly = target[1] - eye[1];
    float lz = target[2] - eye[2];
    float llen = std::sqrt(lx*lx + ly*ly + lz*lz);
    if (llen < 0.0001f) llen = 1.0f;
    lx /= llen; ly /= llen; lz /= llen;

    // right = cross(look, up)
    float rx = ly * up[2] - lz * up[1];
    float ry = lz * up[0] - lx * up[2];
    float rz = lx * up[1] - ly * up[0];
    float rlen = std::sqrt(rx*rx + ry*ry + rz*rz);
    if (rlen > 0.0001f) { rx /= rlen; ry /= rlen; rz /= rlen; }

    // up' = cross(right, look)
    float ux = ry * lz - rz * ly;
    float uy = rz * lx - rx * lz;
    float uz = rx * ly - ry * lx;
    float ulen = std::sqrt(ux*ux + uy*uy + uz*uz);
    if (ulen > 0.0001f) { ux /= ulen; uy /= ulen; uz /= ulen; }

    // Apply deltas: dx along right, dy along up', dz along look (forward positive = -look? convention: +dz moves toward -look? here +dz moves forward toward target)
    eye[0] += rx * dx + ux * dy + lx * dz;
    eye[1] += ry * dx + uy * dy + ly * dz;
    eye[2] += rz * dx + uz * dy + lz * dz;

    // keep target relative? for simple free cam we also shift target by same
    target[0] += rx * dx + ux * dy + lx * dz;
    target[1] += ry * dx + uy * dy + ly * dz;
    target[2] += rz * dx + uz * dy + lz * dz;
}

void Camera4D::adjust_slice(float delta) {
    w_slice += delta;
}

void Camera4D::adjust_hyper(int index, float delta) {
    if (index >= 0 && index < 4) {
        hyper_rot[index] += delta;
    }
}

void Camera4D::reset() {
    eye[0] = 0.0f; eye[1] = 0.0f; eye[2] = 6.0f;
    target[0] = 0.0f; target[1] = 0.0f; target[2] = 0.0f;
    up[0] = 0; up[1] = 1; up[2] = 0;
    w_slice = 0.0f;
    hyper_rot[0] = 0.15f;
    hyper_rot[1] = 0.25f;
    hyper_rot[2] = 0.05f;
    hyper_rot[3] = 0.10f;
}

void Camera4D::chase_ship(const float ship_pos[3], const float ship_forward[3], float dt) {
    // Compute 3D chase: eye slightly behind and above ship in its local space.
    // This makes the camera manipulate relative to the ship in 3D environment,
    // rather than rotating the entire world view.
    float fwd[3] = {ship_forward[0], ship_forward[1], ship_forward[2]};
    float flen = sqrtf(fwd[0]*fwd[0] + fwd[1]*fwd[1] + fwd[2]*fwd[2]);
    if (flen > 0.001f) {
        fwd[0] /= flen; fwd[1] /= flen; fwd[2] /= flen;
    } else {
        fwd[0] = 0; fwd[1] = -1; fwd[2] = 0;  // default forward
    }

    // Right and up basis (no roll for simplicity, use world up cross)
    float up_world[3] = {0, 1, 0};
    float right[3] = {
        fwd[1]*up_world[2] - fwd[2]*up_world[1],
        fwd[2]*up_world[0] - fwd[0]*up_world[2],
        fwd[0]*up_world[1] - fwd[1]*up_world[0]
    };
    float rlen = sqrtf(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    if (rlen > 0.001f) {
        right[0]/=rlen; right[1]/=rlen; right[2]/=rlen;
    } else {
        right[0] = 1; right[1] = 0; right[2] = 0;
    }

    float ship_up[3] = {
        right[1]*fwd[2] - right[2]*fwd[1],
        right[2]*fwd[0] - right[0]*fwd[2],
        right[0]*fwd[1] - right[1]*fwd[0]
    };

    // Desired offset: behind along -fwd, above along ship_up
    const float dist = 2.2f;
    const float height = 0.9f;
    float desired_eye[3] = {
        ship_pos[0] - fwd[0] * dist + ship_up[0] * height,
        ship_pos[1] - fwd[1] * dist + ship_up[1] * height,
        ship_pos[2] - fwd[2] * dist + ship_up[2] * height
    };

    // Target: ahead of ship
    const float lead = 0.8f;
    float desired_target[3] = {
        ship_pos[0] + fwd[0] * lead,
        ship_pos[1] + fwd[1] * lead,
        ship_pos[2] + fwd[2] * lead
    };

    // Smooth lerp for cinematic feel (use dt for frame rate independence)
    const float smooth = 8.0f * dt;  // higher = snappier
    for (int i=0; i<3; i++) {
        eye[i] = eye[i] + (desired_eye[i] - eye[i]) * smooth;
        target[i] = target[i] + (desired_target[i] - target[i]) * smooth;
    }

    // Keep up as world up for no roll
    up[0] = 0; up[1] = 1; up[2] = 0;

    // 4D params (w_slice, hyper_rot) left for background/4D effects only; main gameplay is 3D ship-relative.
}

} // namespace kengine
