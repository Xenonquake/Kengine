#include "gauss_legendre.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Golub-Welsch style nodes via Newton on Legendre polynomial. */
static float legendre_p(int n, float x) {
    if (n == 0) return 1.0f;
    if (n == 1) return x;
    float p0 = 1.0f, p1 = x, p2 = 0.0f;
    for (int k = 2; k <= n; ++k) {
        p2 = ((2.0f * k - 1.0f) * x * p1 - (k - 1.0f) * p0) / (float)k;
        p0 = p1;
        p1 = p2;
    }
    return p1;
}

static float legendre_dp(int n, float x) {
    return (float)n * (legendre_p(n - 1, x) - x * legendre_p(n, x)) / (1.0f - x * x);
}

int ke_gl_nodes_generate(ke_gl_node* nodes, size_t n) {
    if (!nodes || n == 0) return -1;

    for (size_t i = 0; i < n; ++i) {
        float x = cosf((float)M_PI * ((float)i + 0.75f) / ((float)n + 0.5f));
        for (int iter = 0; iter < 20; ++iter) {
            float px  = legendre_p((int)n, x);
            float dpx = legendre_dp((int)n, x);
            float dx  = -px / dpx;
            x += dx;
            if (fabsf(dx) < 1e-12f) break;
        }
        nodes[i].mu     = x;
        nodes[i].weight = 2.0f / ((1.0f - x * x) * legendre_dp((int)n, x) * legendre_dp((int)n, x));
    }
    return 0;
}

void ke_gl_direction(float mu, float phi, float out_xyz[3]) {
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - mu * mu));
    out_xyz[0] = sin_theta * cosf(phi);
    out_xyz[1] = mu;
    out_xyz[2] = sin_theta * sinf(phi);
}