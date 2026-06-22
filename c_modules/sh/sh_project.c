#include "sh_project.h"
#include "../math/gauss_legendre.h"
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int ke_sh_samples_generate(ke_sh_sample* samples, size_t n_mu, size_t n_phi) {
    if (!samples || n_mu == 0 || n_phi == 0) return -1;

    ke_gl_node* nodes = (ke_gl_node*)malloc(n_mu * sizeof(ke_gl_node));
    if (!nodes) return -1;

    if (ke_gl_nodes_generate(nodes, n_mu) != 0) {
        free(nodes);
        return -1;
    }

    size_t idx = 0;
    for (size_t i = 0; i < n_mu; ++i) {
        float dphi = 2.0f * (float)M_PI / (float)n_phi;
        for (size_t j = 0; j < n_phi; ++j) {
            float phi = dphi * ((float)j + 0.5f);
            ke_sh_sample* s = &samples[idx++];
            ke_gl_direction(nodes[i].mu, phi, s->direction);
            /* Solid angle weight: GL weight * dphi */
            s->weight = nodes[i].weight * dphi;
        }
    }

    free(nodes);
    return 0;
}

void ke_sh_project(
    const ke_sh_sample* samples,
    size_t sample_count,
    ke_sh_radiance_fn radiance,
    void* userdata,
    float coeffs[KE_SH_NUM_COEF]
) {
    memset(coeffs, 0, KE_SH_NUM_COEF * sizeof(float));

    float basis[KE_SH_NUM_COEF];
    for (size_t i = 0; i < sample_count; ++i) {
        float L = radiance(samples[i].direction, userdata);
        ke_sh_basis_eval(samples[i].direction, basis);
        float w = samples[i].weight;
        for (int c = 0; c < KE_SH_NUM_COEF; ++c) {
            coeffs[c] += L * basis[c] * w;
        }
    }
}