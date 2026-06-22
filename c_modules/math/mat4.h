#pragma once

#include "vec4.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ke_mat4 {
    float m[16]; /* column-major */
} ke_mat4;

ke_mat4 ke_mat4_identity(void);
ke_mat4 ke_mat4_mul(ke_mat4 a, ke_mat4 b);
ke_vec4 ke_mat4_mul_vec4(ke_mat4 m, ke_vec4 v);
ke_vec3 ke_mat4_mul_vec3_point(ke_mat4 m, ke_vec3 v);
ke_vec3 ke_mat4_mul_vec3_dir(ke_mat4 m, ke_vec3 v);
ke_mat4 ke_mat4_perspective(float fov_y, float aspect, float near_z, float far_z);
ke_mat4 ke_mat4_look_at(ke_vec3 eye, ke_vec3 target, ke_vec3 up);

#ifdef __cplusplus
}
#endif