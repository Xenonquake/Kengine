#version 450

layout(location = 0) in vec2 vUV;
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

vec3 arcade_bg(vec2 uv, float morph) {
    float grid = abs(sin(uv.x * 40.0 * (1.0 + morph))) * abs(sin(uv.y * 40.0 * (1.0 + morph)));
    vec3 col = vec3(0.02, 0.02, 0.08) + vec3(0.0, 0.05, 0.12) * grid;
    float vignette = 1.0 - length(uv - 0.5) * 0.8;
    return col * vignette;
}

void main() {
    vec2 uv = vUV;
    vec3 bg = arcade_bg(uv, pc.w_morph);

    // Procedural "4D cross-section" rings
    float rings = 0.0;
    for (int i = 0; i < 4; ++i) {
        float fi = float(i);
        vec2 center = vec2(0.5) + 0.15 * vec2(
            sin(pc.time * 0.5 + fi + pc.hyper_rot.x),
            cos(pc.time * 0.7 + fi + pc.hyper_rot.y)
        );
        float r = 0.08 + 0.02 * fi + 0.05 * pc.w_morph;
        float d = abs(length(uv - center) - r);
        rings += exp(-d * 80.0) * (0.6 + 0.4 * sin(pc.time * 3.0 + fi));
    }

    vec3 neon = vec3(0.0, 1.0, 0.85) * rings * pc.glow_intensity;
    float scan = 1.0 - pc.scanline_strength * abs(sin(uv.y * pc.viewport.y * 3.14159));
    vec3 col = (bg + neon) * scan;

    // CRT chromatic fringe at 4D transition boundary
    float fringe = smoothstep(0.3, 0.7, pc.w_morph) * 0.15;
    col.r += fringe * sin(uv.x * 200.0 + pc.time * 5.0);
    col.b -= fringe * sin(uv.x * 200.0 + pc.time * 5.0);

    outColor = vec4(col, 1.0);
}