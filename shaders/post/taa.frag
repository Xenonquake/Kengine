#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D currentFrame;
layout(set = 0, binding = 1) uniform sampler2D historyFrame;

layout(push_constant) uniform Push {
    float blendFactor;
} pc;

void main() {
    vec3 current  = texture(currentFrame, vUV).rgb;
    vec3 history  = texture(historyFrame, vUV).rgb;

    vec2 texel = 1.0 / vec2(textureSize(currentFrame, 0));
    vec3 nearest = vec3(0.0);
    float best_depth = 1.0;

    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec3 s = texture(currentFrame, vUV + vec2(x, y) * texel).rgb;
            float d = dot(s, vec3(0.299, 0.587, 0.114));
            if (d < best_depth) { best_depth = d; nearest = s; }
        }
    }

    vec3 result = mix(history, nearest, pc.blendFactor);
    outColor = vec4(result, 1.0);
}