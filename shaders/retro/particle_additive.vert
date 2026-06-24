#version 450

layout(location = 0) in vec4 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in uint inColor;
layout(location = 4) in vec3 inVel;  // velocity for speed effects

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec4 vColor;
layout(location = 2) out float vSpeed;

layout(push_constant) uniform PC {
    mat4 mvp;
    float w_slice;
    float w_morph;
    float glow_intensity;
    float time;
    vec4 hyper_rot;
    vec2 viewport;
    float scanline_strength;
    float pixel_snap;
    float palette_index;
    float _pad;
} pc;

vec4 rotate_plane(vec4 p, int i, int j, float angle) {
    float c = cos(angle), s = sin(angle);
    float pi = p[i], pj = p[j];
    p[i] = pi * c - pj * s;
    p[j] = pi * s + pj * c;
    return p;
}

void main() {
    vec4 p = inPos;
    p.w -= pc.w_slice;
    p = rotate_plane(p, 0, 3, pc.hyper_rot.x + pc.time * 0.3);
    p = rotate_plane(p, 1, 3, pc.hyper_rot.y + pc.time * 0.2);
    p = rotate_plane(p, 0, 1, pc.hyper_rot.z);
    p = rotate_plane(p, 2, 3, pc.hyper_rot.w);

    vec3 pos2d = vec3(p.x, p.y, 0.0);
    float denom = max(0.001, 2.0 - p.w);
    vec3 proj = vec3(p.x, p.y, p.z) * (2.0 / denom);
    vec3 pos3 = mix(pos2d, proj, pc.w_morph);

    float speed = length(inVel);
    float pulse = 1.0 + 0.3 * sin(pc.time * 8.0 + p.w * 10.0) + speed * 0.1;
    gl_Position = pc.mvp * vec4(pos3 * pulse, 1.0);
    vUV = inUV;
    vColor = unpackUnorm4x8(inColor);
    vSpeed = speed;
}