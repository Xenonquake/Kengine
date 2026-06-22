#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ke_gl_node {
    float mu;    /* cos(theta), in [-1, 1] */
    float weight;
} ke_gl_node;

/* Generate n Gauss-Legendre nodes/weights on [-1, 1] for mu = cos(theta). */
int ke_gl_nodes_generate(ke_gl_node* nodes, size_t n);

/* Map (mu, phi) to unit sphere direction (x, y, z). */
void ke_gl_direction(float mu, float phi, float out_xyz[3]);

#ifdef __cplusplus
}
#endif