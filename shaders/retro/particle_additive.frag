#version 450

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec4 vColor;
layout(location = 2) in float vSpeed;

layout(location = 0) out vec4 outColor;

void main() {
    float d = length(vUV - 0.5);
    float glow = exp(-d * 8.0) * (1.0 + vSpeed * 0.15);
    outColor = vec4(vColor.rgb * glow * 4.0, glow);
}