#include "vec4.h"
#include <math.h>

ke_vec4 ke_vec4_make(float x, float y, float z, float w) {
    return (ke_vec4){x, y, z, w};
}

ke_vec4 ke_vec4_add(ke_vec4 a, ke_vec4 b) {
    return (ke_vec4){a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

ke_vec4 ke_vec4_sub(ke_vec4 a, ke_vec4 b) {
    return (ke_vec4){a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

ke_vec4 ke_vec4_scale(ke_vec4 v, float s) {
    return (ke_vec4){v.x * s, v.y * s, v.z * s, v.w * s};
}

float ke_vec4_dot(ke_vec4 a, ke_vec4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

float ke_vec4_length(ke_vec4 v) {
    return sqrtf(ke_vec4_dot(v, v));
}

ke_vec4 ke_vec4_normalize(ke_vec4 v) {
    float len = ke_vec4_length(v);
    if (len < 1e-8f) return (ke_vec4){0, 0, 0, 0};
    return ke_vec4_scale(v, 1.0f / len);
}

ke_vec4 ke_vec4_lerp(ke_vec4 a, ke_vec4 b, float t) {
    return ke_vec4_add(a, ke_vec4_scale(ke_vec4_sub(b, a), t));
}

ke_vec3 ke_vec3_make(float x, float y, float z) {
    return (ke_vec3){x, y, z};
}

ke_vec3 ke_vec3_add(ke_vec3 a, ke_vec3 b) {
    return (ke_vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

ke_vec3 ke_vec3_sub(ke_vec3 a, ke_vec3 b) {
    return (ke_vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

ke_vec3 ke_vec3_scale(ke_vec3 v, float s) {
    return (ke_vec3){v.x * s, v.y * s, v.z * s};
}

float ke_vec3_dot(ke_vec3 a, ke_vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float ke_vec3_length(ke_vec3 v) {
    return sqrtf(ke_vec3_dot(v, v));
}

ke_vec3 ke_vec3_normalize(ke_vec3 v) {
    float len = ke_vec3_length(v);
    if (len < 1e-8f) return (ke_vec3){0, 0, 0};
    return ke_vec3_scale(v, 1.0f / len);
}

ke_vec3 ke_vec3_cross(ke_vec3 a, ke_vec3 b) {
    return (ke_vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}