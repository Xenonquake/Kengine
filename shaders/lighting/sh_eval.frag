#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform SHLighting {
    float coeffs_r[9];
    float coeffs_g[9];
    float coeffs_b[9];
} sh;

vec3 eval_sh(vec3 n, float c[9]) {
    float x = n.x, y = n.y, z = n.z;
    float result = 0.0;
    result += c[0] * 0.28209479;
    result += c[1] * 0.48860251 * z;
    result += c[2] * 0.48860251 * y;
    result += c[3] * 0.48860251 * x;
    result += c[4] * 1.09254843 * x * z;
    result += c[5] * 1.09254843 * z * y;
    result += c[6] * 0.31539157 * (3.0 * y * y - 1.0);
    result += c[7] * 1.09254843 * x * y;
    result += c[8] * 0.54627422 * (x * x - z * z);
    return vec3(max(result, 0.0));
}

void main() {
    vec3 normal = normalize(vec3(vUV * 2.0 - 1.0, 1.0));
    vec3 irradiance = eval_sh(normal, sh.coeffs_r)
                    + eval_sh(normal, sh.coeffs_g)
                    + eval_sh(normal, sh.coeffs_b);
    outColor = vec4(irradiance * 0.333, 1.0);
}