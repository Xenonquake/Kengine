#include "sh_eval.h"
#include <math.h>

float ke_sh_eval_irradiance(const float normal[3], const float coeffs[KE_SH_NUM_COEF]) {
    float basis[KE_SH_NUM_COEF];
    ke_sh_basis_eval(normal, basis);

    float result = 0.0f;
    for (int i = 0; i < KE_SH_NUM_COEF; ++i) {
        result += coeffs[i] * basis[i];
    }
    return fmaxf(0.0f, result);
}

void ke_sh_eval_irradiance_rgb(
    const float normal[3],
    const float coeffs_r[KE_SH_NUM_COEF],
    const float coeffs_g[KE_SH_NUM_COEF],
    const float coeffs_b[KE_SH_NUM_COEF],
    float out_rgb[3]
) {
    out_rgb[0] = ke_sh_eval_irradiance(normal, coeffs_r);
    out_rgb[1] = ke_sh_eval_irradiance(normal, coeffs_g);
    out_rgb[2] = ke_sh_eval_irradiance(normal, coeffs_b);
}