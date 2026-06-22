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

// Solid black background + minimal starfield for a clean Space Arcade look.
// The previous pink/blue gradient + checkerboard arcade effect has been removed
// so sprites (stars/rocks/ship) stand out clearly.
vec3 space_bg(vec2 uv) {
    vec3 col = vec3(0.0);

    // Very faint, sparse procedural stars (pixel-friendly, low density)
    vec2 star_uv = uv * 95.0;
    vec2 star_id = floor(star_uv);
    vec2 star_fr = fract(star_uv);
    float star_rnd = fract(sin(dot(star_id, vec2(127.1, 311.7))) * 43758.5453123);
    float star = step(0.9965, star_rnd);           // very sparse
    star *= (0.7 + 0.3 * sin(pc.time * 3.0 + star_rnd * 20.0)); // gentle twinkle
    // soft point shape
    star *= max(0.0, 1.0 - length(star_fr - 0.5) * 3.5);
    col += star * 0.9;

    // occasional slightly brighter / larger ones
    float big_rnd = fract(sin(dot(star_id + 17.0, vec2(54.3, 19.7))) * 43758.5453);
    float big = step(0.9992, big_rnd);
    big *= max(0.0, 1.0 - length(star_fr - 0.5) * 2.0);
    col += big * 1.6;

    return col;
}

void main() {
    vec2 uv = vUV;
    vec3 bg = space_bg(uv);

    // Subtle procedural "4D cross-section" rings (space/neon feel, now on pure black)
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