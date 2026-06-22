#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D sceneColor;
layout(set = 0, binding = 1) uniform sampler2D sceneDepth;
layout(set = 0, binding = 2) uniform sampler2D bloomTex;  // for bloom composite

layout(push_constant) uniform PC {
    float exposure;
    float scanline_strength;
    float w_morph;
    float time;
} pc;

vec3 aces_tonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 scene = texture(sceneColor, vUV).rgb;
    vec3 bloom = texture(bloomTex, vUV).rgb * 0.8;  // bloom strength (additive composite)
    vec3 color = (scene + bloom) * pc.exposure;
    vec3 ldr = aces_tonemap(color);

    float scan = 1.0 - pc.scanline_strength * abs(sin(vUV.y * 1080.0 * 3.14159));
    ldr *= scan;

    float fringe = smoothstep(0.3, 0.7, pc.w_morph) * 0.08;
    ldr.r += fringe * sin(vUV.x * 300.0 + pc.time * 4.0);
    ldr.b -= fringe * sin(vUV.x * 300.0 + pc.time * 4.0);

    outColor = vec4(ldr, 1.0);
}
