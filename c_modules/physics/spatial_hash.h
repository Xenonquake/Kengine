#pragma once

#include "verlet_solver.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ke_spatial_hash {
    int32_t*  cell_heads;
    int32_t*  next_indices;
    float     cell_size;
    int32_t   grid_x, grid_y, grid_z;
    size_t    body_count;
} ke_spatial_hash;

ke_spatial_hash* ke_spatial_hash_create(float cell_size, int32_t gx, int32_t gy, int32_t gz, size_t max_bodies);
void             ke_spatial_hash_destroy(ke_spatial_hash* hash);
void             ke_spatial_hash_rebuild(ke_spatial_hash* hash, const ke_phys_world* world);

#ifdef __cplusplus
}
#endif