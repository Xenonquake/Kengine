#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Real SH basis, l_max = 2 -> 9 coefficients (RGB tripled at runtime). */
#define KE_SH_LMAX     2
#define KE_SH_NUM_COEF 9

/* Evaluate all basis functions Y_l^m(n) at unit normal n. */
void ke_sh_basis_eval(const float normal[3], float basis[KE_SH_NUM_COEF]);

/* Associated Legendre P_l^m(mu) via stable recurrence. */
float ke_sh_legendre(int l, int m, float mu);

#ifdef __cplusplus
}
#endif