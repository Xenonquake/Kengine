#version 450

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec4 vColor;
layout(location = 2) in float vIntensity;

layout(location = 0) out vec4 outColor;

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
    float line = 1.0 - abs(vUV.y - 0.5) * 2.0;
    line = pow(max(line, 0.0), 1.5);
    vec3 col = vColor.rgb * vIntensity * line * 3.0;
    outColor = vec4(col, line);
}