#include "sh_basis.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float ke_sh_legendre(int l, int m, float mu) {
    /* Condon-Shortley phase included in basis normalization below. */
    float p_mm = 1.0f;
    if (m > 0) {
        float somx2 = sqrtf(fmaxf(0.0f, 1.0f - mu * mu));
        float fact = 1.0f;
        for (int i = 1; i <= m; ++i) {
            p_mm *= -fact * somx2;
            fact += 2.0f;
        }
    }
    if (l == m) return p_mm;

    float p_mmp1 = mu * (2.0f * (float)m + 1.0f) * p_mm;
    if (l == m + 1) return p_mmp1;

    float p_lm2 = p_mm;
    float p_lm1 = p_mmp1;
    float p_l   = 0.0f;
    for (int ll = m + 2; ll <= l; ++ll) {
        p_l = ((2.0f * (float)ll - 1.0f) * mu * p_lm1 - ((float)ll + (float)m - 1.0f) * p_lm2) / ((float)ll - (float)m);
        p_lm2 = p_lm1;
        p_lm1 = p_l;
    }
    return p_l;
}

static float sh_k(int l, int m) {
    return sqrtf((2.0f * (float)l + 1.0f) / (4.0f * (float)M_PI)
               * tgammaf((float)l - (float)m + 1.0f) / tgammaf((float)l + (float)m + 1.0f));
}

void ke_sh_basis_eval(const float normal[3], float basis[KE_SH_NUM_COEF]) {
    float x = normal[0], y = normal[1], z = normal[2];
    float phi = atan2f(z, x);
    float mu  = y;

    /* l=0 */
    basis[0] = 0.28209479177387814f; /* Y_0^0 */

    /* l=1 */
    float c1 = 0.48860251190291992f;
    basis[1] = c1 * z;  /* Y_1^{-1} */
    basis[2] = c1 * mu; /* Y_1^{0}  */
    basis[3] = c1 * x;  /* Y_1^{1}  */

    /* l=2 */
    float c2_0 = 0.31539156525252005f;
    float c2_1 = 1.0925484305920792f;
    float c2_2 = 0.54627421529603960f;
    basis[4] = c2_1 * x * z;
    basis[5] = c2_1 * z * mu;
    basis[6] = c2_0 * (3.0f * mu * mu - 1.0f);
    basis[7] = c2_1 * x * mu;
    basis[8] = c2_2 * (x * x - z * z);

    (void)phi;
    (void)sh_k;
}