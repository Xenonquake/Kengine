#version 450

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec4 vColor;
layout(location = 2) in float vPaletteIdx;

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

vec3 retro_palette(float idx, float palette_index) {
    int p = int(palette_index + 0.5);
    if (p == 0) return mix(vec3(0.0,1.0,1.0), vec3(1.0,0.2,0.8), idx);
    if (p == 2) return mix(vec3(0.4,0.5,0.2), vec3(1.0,0.5,0.1), idx);
    return mix(vec3(0.0,1.0,0.8), vec3(1.0,0.1,0.6), idx);
}

void main() {
    vec2 uv = vUV;
    float edge = smoothstep(0.0, 0.08, min(min(uv.x, 1.0-uv.x), min(uv.y, 1.0-uv.y)));
    vec3 base = vColor.rgb * retro_palette(vPaletteIdx, pc.palette_index);
    float scan = 1.0 - pc.scanline_strength * abs(sin(gl_FragCoord.y * 3.14159));
    vec3 col = base * edge * scan;
    outColor = vec4(col, vColor.a * edge);
}