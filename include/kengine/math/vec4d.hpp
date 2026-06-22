#pragma once

#include "vec4.h"
#include "mat4.h"
#include <array>

namespace kengine {

struct Vec4d {
    float x = 0, y = 0, z = 0, w = 0;

    Vec4d() = default;
    Vec4d(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    explicit Vec4d(ke_vec4 v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

    ke_vec4 to_c() const { return ke_vec4_make(x, y, z, w); }

    Vec4d operator+(const Vec4d& o) const { return Vec4d(ke_vec4_add(to_c(), o.to_c())); }
    Vec4d operator-(const Vec4d& o) const { return Vec4d(ke_vec4_sub(to_c(), o.to_c())); }
    Vec4d operator*(float s) const { return Vec4d(ke_vec4_scale(to_c(), s)); }

    float dot(const Vec4d& o) const { return ke_vec4_dot(to_c(), o.to_c()); }
    float length() const { return ke_vec4_length(to_c()); }
    Vec4d normalized() const { return Vec4d(ke_vec4_normalize(to_c())); }

    /* Project 4D -> 3D via perspective divide on w */
    std::array<float, 3> project3d() const {
        float inv_w = (w != 0.0f) ? (1.0f / w) : 1.0f;
        return {x * inv_w, y * inv_w, z * inv_w};
    }
};

struct Mat4d {
    ke_mat4 m = ke_mat4_identity();

    static Mat4d identity() { return Mat4d{}; }
    static Mat4d perspective(float fov_y, float aspect, float near_z, float far_z) {
        Mat4d r;
        r.m = ke_mat4_perspective(fov_y, aspect, near_z, far_z);
        return r;
    }

    Vec4d transform(const Vec4d& v) const { return Vec4d(ke_mat4_mul_vec4(m, v.to_c())); }
};

} // namespace kengine