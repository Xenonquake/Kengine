#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D colorTex;
layout(set = 0, binding = 1) uniform sampler2D depthTex;

layout(push_constant) uniform Push {
    float focusDistance;
    float aperture;
    float maxBlur;
} pc;

void main() {
    float depth = texture(depthTex, vUV).r;
    float coc = clamp(abs(depth - pc.focusDistance) * pc.aperture, 0.0, pc.maxBlur);

    vec2 texel = 1.0 / vec2(textureSize(colorTex, 0));
    vec3 color = vec3(0.0);
    float total = 0.0;

    for (int x = -2; x <= 2; ++x) {
        for (int y = -2; y <= 2; ++y) {
            vec2 offset = vec2(x, y) * texel * coc;
            float w = 1.0 / (1.0 + length(vec2(x, y)));
            color += texture(colorTex, vUV + offset).rgb * w;
            total += w;
        }
    }
    outColor = vec4(color / total, 1.0);
}