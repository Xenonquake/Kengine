#pragma once

#include "sh_basis.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float (*ke_sh_radiance_fn)(const float dir[3], void* userdata);

typedef struct ke_sh_sample {
    float direction[3];
    float weight;
} ke_sh_sample;

/* Generate Gauss-Legendre (mu) + uniform (phi) sample set. */
int ke_sh_samples_generate(ke_sh_sample* samples, size_t n_mu, size_t n_phi);

/* Project radiance function onto SH coefficients (single channel). */
void ke_sh_project(
    const ke_sh_sample* samples,
    size_t sample_count,
    ke_sh_radiance_fn radiance,
    void* userdata,
    float coeffs[KE_SH_NUM_COEF]
);

#ifdef __cplusplus
}
#endif