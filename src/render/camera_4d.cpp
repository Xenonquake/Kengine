#include "kengine/render/camera_4d.hpp"
#include "mat4.h"
#include <cstring>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

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
    // Copy full 4D hyper rotation matrix (16 floats)
    for (int i = 0; i < 16; ++i) {
        pc.hyper_rot[i] = hyper_rot.m.m[i];
    }
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

void Camera4D::adjust_hyper(int plane, float delta) {
    if (plane < 0 || plane > 5) return;
    // Compose a small rotation in the given plane onto current hyper_rot
    static const int plane_pairs[6][2] = {
        {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}  // xy, xz, xw, yz, yw, zw
    };
    int i = plane_pairs[plane][0];
    int j = plane_pairs[plane][1];
    Mat4d4 rot = Mat4d4::rotate_plane(i, j, delta);
    hyper_rot = hyper_rot * rot;  // right-multiply
}

void Camera4D::reset() {
    eye[0] = 0.0f; eye[1] = 0.0f; eye[2] = 6.0f;
    target[0] = 0.0f; target[1] = 0.0f; target[2] = 0.0f;
    up[0] = 0; up[1] = 1; up[2] = 0;
    w_slice = 0.0f;
    hyper_rot = Mat4d4::identity();
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

void Camera4D::extract_frustum_planes(float aspect, FrustumPlanes& out) const {
    ke_mat4 proj = ke_mat4_perspective(fov, aspect, near_z, far_z);
    ke_vec3 e = ke_vec3_make(eye[0], eye[1], eye[2]);
    ke_vec3 t = ke_vec3_make(target[0], target[1], target[2]);
    ke_vec3 u = ke_vec3_make(up[0], up[1], up[2]);
    ke_mat4 view = ke_mat4_look_at(e, t, u);
    ke_mat4 mvp = ke_mat4_mul(proj, view);

    // Extract planes from mvp (column-major in m[16])
    // Left:   m3 + m0
    // Right:  m3 - m0
    // Bottom: m3 + m1
    // Top:    m3 - m1
    // Near:   m3 + m2   (or depending on proj z convention; our near/far)
    // Far:    m3 - m2
    const float* m = mvp.m;

    auto set_plane = [&](int i, float a, float b, float c, float d) {
        out.planes[i][0] = a;
        out.planes[i][1] = b;
        out.planes[i][2] = c;
        out.planes[i][3] = d;
        // normalize for consistent radius tests
        float len = std::sqrt(a*a + b*b + c*c);
        if (len > 1e-6f) {
            out.planes[i][0] /= len;
            out.planes[i][1] /= len;
            out.planes[i][2] /= len;
            out.planes[i][3] /= len;
        }
    };

    // Row/column: assuming m[0..3] = col0 etc. Standard extraction for column major MVP:
    // plane coeffs from matrix rows (transposed view in some refs)
    // Common working extraction (tested in many engines):
    set_plane(0, m[3] + m[0], m[7] + m[4], m[11] + m[8],  m[15] + m[12]); // left
    set_plane(1, m[3] - m[0], m[7] - m[4], m[11] - m[8],  m[15] - m[12]); // right
    set_plane(2, m[3] + m[1], m[7] + m[5], m[11] + m[9],  m[15] + m[13]); // bottom
    set_plane(3, m[3] - m[1], m[7] - m[5], m[11] - m[9],  m[15] - m[13]); // top
    set_plane(4, m[3] + m[2], m[7] + m[6], m[11] + m[10], m[15] + m[14]); // near (adjust sign if reversed depth)
    set_plane(5, m[3] - m[2], m[7] - m[6], m[11] - m[10], m[15] - m[14]); // far
}

} // namespace kengine
