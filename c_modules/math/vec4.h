#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ke_vec4 {
    float x, y, z, w;
} ke_vec4;

typedef struct ke_vec3 {
    float x, y, z;
} ke_vec3;

ke_vec4 ke_vec4_make(float x, float y, float z, float w);
ke_vec4 ke_vec4_add(ke_vec4 a, ke_vec4 b);
ke_vec4 ke_vec4_sub(ke_vec4 a, ke_vec4 b);
ke_vec4 ke_vec4_scale(ke_vec4 v, float s);
float   ke_vec4_dot(ke_vec4 a, ke_vec4 b);
float   ke_vec4_length(ke_vec4 v);
ke_vec4 ke_vec4_normalize(ke_vec4 v);
ke_vec4 ke_vec4_lerp(ke_vec4 a, ke_vec4 b, float t);

ke_vec3 ke_vec3_make(float x, float y, float z);
ke_vec3 ke_vec3_add(ke_vec3 a, ke_vec3 b);
ke_vec3 ke_vec3_sub(ke_vec3 a, ke_vec3 b);
ke_vec3 ke_vec3_scale(ke_vec3 v, float s);
float   ke_vec3_dot(ke_vec3 a, ke_vec3 b);
float   ke_vec3_length(ke_vec3 v);
ke_vec3 ke_vec3_normalize(ke_vec3 v);
ke_vec3 ke_vec3_cross(ke_vec3 a, ke_vec3 b);

#ifdef __cplusplus
}
#endif