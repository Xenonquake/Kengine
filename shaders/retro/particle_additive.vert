#version 450

layout(location = 0) in vec4 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in uint inColor;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec4 vColor;

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

void main() {
    vec4 p = inPos;
    p.w -= pc.w_slice;
    vec3 pos2d = vec3(p.x, p.y, 0.0);
    float denom = max(0.001, 2.0 - p.w);
    vec3 proj = vec3(p.x, p.y, p.z) * (2.0 / denom);
    vec3 pos3 = mix(pos2d, proj, pc.w_morph);
    float pulse = 1.0 + 0.3 * sin(pc.time * 8.0 + p.w * 10.0);
    gl_Position = pc.mvp * vec4(pos3 * pulse, 1.0);
    vUV = inUV;
    vColor = unpackUnorm4x8(inColor);
}