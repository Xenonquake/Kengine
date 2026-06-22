#version 450

const int SH_NUM_COEF = 9;

vec3 sh_eval(vec3 n, float coeffs[SH_NUM_COEF]) {
    float x = n.x, y = n.y, z = n.z;

    float basis[SH_NUM_COEF];
    basis[0] = 0.28209479177387814;
    basis[1] = 0.48860251190291992 * z;
    basis[2] = 0.48860251190291992 * y;
    basis[3] = 0.48860251190291992 * x;
    basis[4] = 1.0925484305920792 * x * z;
    basis[5] = 1.0925484305920792 * z * y;
    basis[6] = 0.31539156525252005 * (3.0 * y * y - 1.0);
    basis[7] = 1.0925484305920792 * x * y;
    basis[8] = 0.5462742152960396 * (x * x - z * z);

    float result = 0.0;
    for (int i = 0; i < SH_NUM_COEF; ++i) {
        result += coeffs[i] * basis[i];
    }
    return vec3(max(result, 0.0));
}