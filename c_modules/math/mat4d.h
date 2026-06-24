#pragma once

#include "vec4.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ke_mat4d {
    float m[16]; /* column-major 4x4 */
} ke_mat4d;

ke_mat4d ke_mat4d_identity(void);
ke_mat4d ke_mat4d_mul(ke_mat4d a, ke_mat4d b);
ke_vec4  ke_mat4d_mul_vec4(ke_mat4d m, ke_vec4 v);

/* Rotate in the (i,j) plane of (x,y,z,w); indices 0..3. */
ke_mat4d ke_mat4d_rotate_plane(int i, int j, float angle);

/* Project 4D point to 3D: stereographic with w_offset, scaled by perspective. */
ke_vec3 ke_mat4d_project_to_3d(ke_vec4 p, float w_focus, float perspective);

/* Lerp between flat 2D (z=0,w=0) and full 4D projection. morph in [0,1]. */
ke_vec3 ke_mat4d_project_morph(ke_vec4 p, float w_focus, float perspective, float morph);

/* Build model matrix: 4D translation + per-plane rotations + uniform scale.
 * rotations[6] for full SO(4): [0]=xy, [1]=xz, [2]=xw, [3]=yz, [4]=yw, [5]=zw
 */
ke_mat4d ke_mat4d_model(ke_vec4 translation, const float rotations[6], float scale);

/* Lerp between two 4D matrices (element-wise, for simple interpolation) */
ke_mat4d ke_mat4d_lerp(ke_mat4d a, ke_mat4d b, float t);

#ifdef __cplusplus
}
#endif