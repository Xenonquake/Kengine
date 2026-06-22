#version 450

layout(location = 0) in vec4 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in uint inColor;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec4 vColor;
layout(location = 2) out float vIntensity;

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

vec3 project_morph(vec4 p, float morph) {
    vec3 pos2d = vec3(p.x, p.y, 0.0);
    float denom = max(0.001, 2.0 - p.w);
    vec3 proj = vec3(p.x, p.y, p.z) * (2.0 / denom);
    return mix(pos2d, proj, morph);
}

void main() {
    vec4 p = inPos;
    p.w -= pc.w_slice;
    vec3 pos3 = project_morph(p, pc.w_morph);
    gl_Position = pc.mvp * vec4(pos3, 1.0);
    vUV = inUV;
    vColor = unpackUnorm4x8(inColor);
    vIntensity = pc.glow_intensity;
}