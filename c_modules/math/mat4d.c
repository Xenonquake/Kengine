#include "mat4d.h"
#include <math.h>
#include <string.h>

ke_mat4d ke_mat4d_identity(void) {
    ke_mat4d r;
    memset(r.m, 0, sizeof(r.m));
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

ke_mat4d ke_mat4d_mul(ke_mat4d a, ke_mat4d b) {
    ke_mat4d r = {0};
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

ke_vec4 ke_mat4d_mul_vec4(ke_mat4d m, ke_vec4 v) {
    return (ke_vec4){
        m.m[0]*v.x + m.m[4]*v.y + m.m[8]*v.z  + m.m[12]*v.w,
        m.m[1]*v.x + m.m[5]*v.y + m.m[9]*v.z  + m.m[13]*v.w,
        m.m[2]*v.x + m.m[6]*v.y + m.m[10]*v.z + m.m[14]*v.w,
        m.m[3]*v.x + m.m[7]*v.y + m.m[11]*v.z + m.m[15]*v.w
    };
}

ke_mat4d ke_mat4d_rotate_plane(int i, int j, float angle) {
    ke_mat4d r = ke_mat4d_identity();
    float c = cosf(angle), s = sinf(angle);
    r.m[i * 4 + i] = c;
    r.m[i * 4 + j] = -s;
    r.m[j * 4 + i] = s;
    r.m[j * 4 + j] = c;
    return r;
}

ke_vec3 ke_mat4d_project_to_3d(ke_vec4 p, float w_focus, float perspective) {
    float denom = fmaxf(0.001f, w_focus - p.w);
    float scale = perspective / denom;
    return (ke_vec3){p.x * scale, p.y * scale, p.z * scale};
}

ke_vec3 ke_mat4d_project_morph(ke_vec4 p, float w_focus, float perspective, float morph) {
    ke_vec3 flat = {p.x, p.y, 0.0f};
    ke_vec3 proj = ke_mat4d_project_to_3d(p, w_focus, perspective);
    return (ke_vec3){
        flat.x + (proj.x - flat.x) * morph,
        flat.y + (proj.y - flat.y) * morph,
        flat.z + (proj.z - flat.z) * morph
    };
}

ke_mat4d ke_mat4d_model(ke_vec4 translation, const float rotations[4], float scale) {
    ke_mat4d r = ke_mat4d_identity();
    ke_mat4d rot = ke_mat4d_identity();
    rot = ke_mat4d_mul(ke_mat4d_rotate_plane(0, 3, rotations[0]), rot); /* xw */
    rot = ke_mat4d_mul(ke_mat4d_rotate_plane(1, 3, rotations[1]), rot); /* yw */
    rot = ke_mat4d_mul(ke_mat4d_rotate_plane(0, 1, rotations[2]), rot); /* xy */
    rot = ke_mat4d_mul(ke_mat4d_rotate_plane(2, 3, rotations[3]), rot); /* zw */
    r = rot;
    r.m[0]  *= scale; r.m[1]  *= scale; r.m[2]  *= scale;
    r.m[4]  *= scale; r.m[5]  *= scale; r.m[6]  *= scale;
    r.m[8]  *= scale; r.m[9]  *= scale; r.m[10] *= scale;
    r.m[12] = translation.x;
    r.m[13] = translation.y;
    r.m[14] = translation.z;
    r.m[15] = translation.w;
    return r;
}