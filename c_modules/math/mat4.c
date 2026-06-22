#include "mat4.h"
#include <math.h>
#include <string.h>

ke_mat4 ke_mat4_identity(void) {
    ke_mat4 r = {0};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

ke_mat4 ke_mat4_mul(ke_mat4 a, ke_mat4 b) {
    ke_mat4 r = {0};
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a.m[k * 4 + row] * b.m[col * 4 + k];
            }
            r.m[col * 4 + row] = sum;
        }
    }
    return r;
}

ke_vec4 ke_mat4_mul_vec4(ke_mat4 m, ke_vec4 v) {
    return (ke_vec4){
        m.m[0] * v.x + m.m[4] * v.y + m.m[8]  * v.z + m.m[12] * v.w,
        m.m[1] * v.x + m.m[5] * v.y + m.m[9]  * v.z + m.m[13] * v.w,
        m.m[2] * v.x + m.m[6] * v.y + m.m[10] * v.z + m.m[14] * v.w,
        m.m[3] * v.x + m.m[7] * v.y + m.m[11] * v.z + m.m[15] * v.w
    };
}

ke_vec3 ke_mat4_mul_vec3_point(ke_mat4 m, ke_vec3 v) {
    ke_vec4 r = ke_mat4_mul_vec4(m, (ke_vec4){v.x, v.y, v.z, 1.0f});
    float inv_w = (fabsf(r.w) > 1e-8f) ? (1.0f / r.w) : 1.0f;
    return (ke_vec3){r.x * inv_w, r.y * inv_w, r.z * inv_w};
}

ke_vec3 ke_mat4_mul_vec3_dir(ke_mat4 m, ke_vec3 v) {
    ke_vec4 r = ke_mat4_mul_vec4(m, (ke_vec4){v.x, v.y, v.z, 0.0f});
    return (ke_vec3){r.x, r.y, r.z};
}

ke_mat4 ke_mat4_perspective(float fov_y, float aspect, float near_z, float far_z) {
    float tan_half = tanf(fov_y * 0.5f);
    ke_mat4 r = {0};
    r.m[0]  = 1.0f / (aspect * tan_half);
    r.m[5]  = 1.0f / tan_half;
    r.m[10] = -(far_z + near_z) / (far_z - near_z);
    r.m[11] = -1.0f;
    r.m[14] = -(2.0f * far_z * near_z) / (far_z - near_z);
    return r;
}

ke_mat4 ke_mat4_look_at(ke_vec3 eye, ke_vec3 target, ke_vec3 up) {
    ke_vec3 f = ke_vec3_normalize(ke_vec3_sub(target, eye));
    ke_vec3 s = ke_vec3_normalize(ke_vec3_cross(f, up));
    ke_vec3 u = ke_vec3_cross(s, f);

    ke_mat4 r = ke_mat4_identity();
    r.m[0] = s.x;  r.m[4] = s.y;  r.m[8]  = s.z;
    r.m[1] = u.x;  r.m[5] = u.y;  r.m[9]  = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -ke_vec3_dot(s, eye);
    r.m[13] = -ke_vec3_dot(u, eye);
    r.m[14] =  ke_vec3_dot(f, eye);
    return r;
}