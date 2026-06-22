#version 450

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec4 vColor;
layout(location = 2) in float vGlow;
layout(location = 3) in float vPaletteIdx;

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

vec3 neon(float idx) {
    return mix(vec3(0.0,1.0,0.9), vec3(1.0,0.0,0.8), idx) * vGlow;
}

void main() {
    vec2 uv = vUV - 0.5;
    float dist = length(uv);
    float core = smoothstep(0.5, 0.0, dist);
    float ring = smoothstep(0.45, 0.35, dist) * smoothstep(0.2, 0.35, dist);
    vec3 col = neon(vPaletteIdx) * (core + ring * 2.0);
    float scan = 1.0 - pc.scanline_strength * abs(sin(gl_FragCoord.y * 3.14159));
    col *= scan;
    float alpha = core + ring * 0.6;
    outColor = vec4(col, alpha);
}