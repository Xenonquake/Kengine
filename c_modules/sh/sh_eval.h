#pragma once

#include "sh_basis.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Evaluate irradiance/radiance from SH coefficients at surface normal. */
float ke_sh_eval_irradiance(const float normal[3], const float coeffs[KE_SH_NUM_COEF]);

/* RGB irradiance (coeffs interleaved or per-channel arrays). */
void ke_sh_eval_irradiance_rgb(
    const float normal[3],
    const float coeffs_r[KE_SH_NUM_COEF],
    const float coeffs_g[KE_SH_NUM_COEF],
    const float coeffs_b[KE_SH_NUM_COEF],
    float out_rgb[3]
);

#ifdef __cplusplus
}
#endif