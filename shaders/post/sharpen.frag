#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D inputTex;

layout(push_constant) uniform Push {
    float amount;
} pc;

void main() {
    vec2 texel = 1.0 / vec2(textureSize(inputTex, 0));

    vec3 center = texture(inputTex, vUV).rgb;
    vec3 blur = (
        texture(inputTex, vUV + vec2(-texel.x, 0)).rgb +
        texture(inputTex, vUV + vec2( texel.x, 0)).rgb +
        texture(inputTex, vUV + vec2(0, -texel.y)).rgb +
        texture(inputTex, vUV + vec2(0,  texel.y)).rgb
    ) * 0.25;

    vec3 sharpened = center + (center - blur) * pc.amount;
    outColor = vec4(clamp(sharpened, 0.0, 1.0), 1.0);
}