#version 450

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in uint inColor;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec4 vColor;
layout(location = 2) out float vPaletteIdx;

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
    vec4 pos4 = vec4(inPos, 0.0, 1.0);
    gl_Position = pc.mvp * pos4;
    vUV = inUV;
    vColor = unpackUnorm4x8(inColor);
    vPaletteIdx = float(inColor & 0xFFu) / 255.0;
}