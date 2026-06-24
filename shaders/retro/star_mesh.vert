#version 450

layout(location = 0) in vec3 inPos;

layout(location = 0) out vec3 vColor;

layout(push_constant) uniform PC {
    mat4 mvp;
    float w_slice;
    float w_morph;
    float glow_intensity;
    float time;
    mat4 hyper_rot;
    vec2 viewport;
    float scanline_strength;
    float pixel_snap;
    float palette_index;
    float _pad;
} pc;

layout(std430, set = 1, binding = 0) readonly buffer StarInstances {
    vec4 data[];  // .xyz = pos, .w = scale
} stars;

void main() {
    int idx = gl_InstanceIndex;
    vec4 s = stars.data[idx];
    vec3 worldPos = inPos * s.w + s.xyz;

    vec4 p = vec4(worldPos, 0.0);
    p.w -= pc.w_slice;
    p = pc.hyper_rot * p;

    vec3 pos2d = vec3(p.x, p.y, 0.0);
    float denom = max(0.001, 2.0 - p.w);
    vec3 proj = vec3(p.x, p.y, p.z) * (2.0 / denom);
    vec3 pos3 = mix(pos2d, proj, pc.w_morph);

    gl_Position = pc.mvp * vec4(pos3, 1.0);
    vColor = vec3(1.0, 1.0, 1.0);
}