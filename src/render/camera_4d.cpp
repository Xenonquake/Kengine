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

} // namespace kengine
