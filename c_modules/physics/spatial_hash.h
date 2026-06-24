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

// Query helpers (operate on xyz only; w ignored for spatial locality).
// Bodies are returned by their index in the phys world at last rebuild.
// Use ke_phys_world bodies[index] or PhysicsWorld::get_body to access data.

// Callback style (preferred, zero heap in hot path)
typedef void (*ke_spatial_hash_visit_fn)(int32_t body_index, void* user_data);

void ke_spatial_hash_visit_radius(const ke_spatial_hash* hash,
                                  float cx, float cy, float cz, float radius,
                                  ke_spatial_hash_visit_fn fn, void* user_data);

void ke_spatial_hash_visit_aabb(const ke_spatial_hash* hash,
                                float min_x, float min_y, float min_z,
                                float max_x, float max_y, float max_z,
                                ke_spatial_hash_visit_fn fn, void* user_data);

// Buffer style (convenience). Returns number of indices written (<= max_out).
int ke_spatial_hash_query_radius(const ke_spatial_hash* hash,
                                 float cx, float cy, float cz, float radius,
                                 int32_t* out_indices, int max_out);

int ke_spatial_hash_query_aabb(const ke_spatial_hash* hash,
                               float min_x, float min_y, float min_z,
                               float max_x, float max_y, float max_z,
                               int32_t* out_indices, int max_out);

#ifdef __cplusplus
}
#endif