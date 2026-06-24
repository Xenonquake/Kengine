#include "spatial_hash.h"
#include <stdlib.h>
#include <string.h>

static int32_t cell_index(const ke_spatial_hash* h, int32_t x, int32_t y, int32_t z) {
    return x + y * h->grid_x + z * h->grid_x * h->grid_y;
}

static void clamp_cell(const ke_spatial_hash* h, int32_t* cx, int32_t* cy, int32_t* cz) {
    if (*cx < 0) *cx = 0; else if (*cx >= h->grid_x) *cx = h->grid_x - 1;
    if (*cy < 0) *cy = 0; else if (*cy >= h->grid_y) *cy = h->grid_y - 1;
    if (*cz < 0) *cz = 0; else if (*cz >= h->grid_z) *cz = h->grid_z - 1;
}

// Compute inclusive cell range for a radius query (conservative, may over-report)
static void radius_cell_range(const ke_spatial_hash* h,
                              float cx, float cy, float cz, float radius,
                              int32_t* min_cx, int32_t* min_cy, int32_t* min_cz,
                              int32_t* max_cx, int32_t* max_cy, int32_t* max_cz) {
    float s = h->cell_size;
    *min_cx = (int32_t)((cx - radius) / s);
    *min_cy = (int32_t)((cy - radius) / s);
    *min_cz = (int32_t)((cz - radius) / s);
    *max_cx = (int32_t)((cx + radius) / s);
    *max_cy = (int32_t)((cy + radius) / s);
    *max_cz = (int32_t)((cz + radius) / s);
    clamp_cell(h, min_cx, min_cy, min_cz);
    clamp_cell(h, max_cx, max_cy, max_cz);
}

// Compute inclusive cell range for AABB (min inclusive, max inclusive)
static void aabb_cell_range(const ke_spatial_hash* h,
                            float min_x, float min_y, float min_z,
                            float max_x, float max_y, float max_z,
                            int32_t* min_cx, int32_t* min_cy, int32_t* min_cz,
                            int32_t* max_cx, int32_t* max_cy, int32_t* max_cz) {
    float s = h->cell_size;
    *min_cx = (int32_t)(min_x / s);
    *min_cy = (int32_t)(min_y / s);
    *min_cz = (int32_t)(min_z / s);
    *max_cx = (int32_t)(max_x / s);
    *max_cy = (int32_t)(max_y / s);
    *max_cz = (int32_t)(max_z / s);
    clamp_cell(h, min_cx, min_cy, min_cz);
    clamp_cell(h, max_cx, max_cy, max_cz);
}

ke_spatial_hash* ke_spatial_hash_create(float cell_size, int32_t gx, int32_t gy, int32_t gz, size_t max_bodies) {
    ke_spatial_hash* h = (ke_spatial_hash*)calloc(1, sizeof(ke_spatial_hash));
    if (!h) return NULL;
    h->cell_size = cell_size;
    h->grid_x = gx; h->grid_y = gy; h->grid_z = gz;
    size_t cell_count = (size_t)gx * (size_t)gy * (size_t)gz;
    h->cell_heads  = (int32_t*)malloc(cell_count * sizeof(int32_t));
    h->next_indices = (int32_t*)malloc(max_bodies * sizeof(int32_t));
    if (!h->cell_heads || !h->next_indices) {
        free(h->cell_heads); free(h->next_indices); free(h);
        return NULL;
    }
    h->body_count = max_bodies;
    return h;
}

void ke_spatial_hash_destroy(ke_spatial_hash* hash) {
    if (!hash) return;
    free(hash->cell_heads);
    free(hash->next_indices);
    free(hash);
}

void ke_spatial_hash_rebuild(ke_spatial_hash* hash, const ke_phys_world* world) {
    if (!hash || !world) return;
    size_t cell_count = (size_t)hash->grid_x * (size_t)hash->grid_y * (size_t)hash->grid_z;
    for (size_t i = 0; i < cell_count; ++i) hash->cell_heads[i] = -1;

    for (size_t i = 0; i < world->count; ++i) {
        int32_t cx = (int32_t)(world->bodies[i].position.x / hash->cell_size);
        int32_t cy = (int32_t)(world->bodies[i].position.y / hash->cell_size);
        int32_t cz = (int32_t)(world->bodies[i].position.z / hash->cell_size);
        cx = cx < 0 ? 0 : (cx >= hash->grid_x ? hash->grid_x - 1 : cx);
        cy = cy < 0 ? 0 : (cy >= hash->grid_y ? hash->grid_y - 1 : cy);
        cz = cz < 0 ? 0 : (cz >= hash->grid_z ? hash->grid_z - 1 : cz);

        int32_t ci = cell_index(hash, cx, cy, cz);
        hash->next_indices[i] = hash->cell_heads[ci];
        hash->cell_heads[ci]  = (int32_t)i;
    }
}

static void visit_cell_range(const ke_spatial_hash* hash,
                             int32_t min_cx, int32_t min_cy, int32_t min_cz,
                             int32_t max_cx, int32_t max_cy, int32_t max_cz,
                             ke_spatial_hash_visit_fn fn, void* user) {
    if (!hash || !fn) return;
    for (int32_t cz = min_cz; cz <= max_cz; ++cz) {
        for (int32_t cy = min_cy; cy <= max_cy; ++cy) {
            for (int32_t cx = min_cx; cx <= max_cx; ++cx) {
                int32_t ci = cell_index(hash, cx, cy, cz);
                int32_t idx = hash->cell_heads[ci];
                while (idx >= 0) {
                    fn(idx, user);
                    idx = hash->next_indices[idx];
                }
            }
        }
    }
}

void ke_spatial_hash_visit_radius(const ke_spatial_hash* hash,
                                  float cx, float cy, float cz, float radius,
                                  ke_spatial_hash_visit_fn fn, void* user) {
    if (!hash || radius < 0.0f) return;
    int32_t min_cx, min_cy, min_cz, max_cx, max_cy, max_cz;
    radius_cell_range(hash, cx, cy, cz, radius, &min_cx, &min_cy, &min_cz, &max_cx, &max_cy, &max_cz);
    visit_cell_range(hash, min_cx, min_cy, min_cz, max_cx, max_cy, max_cz, fn, user);
}

void ke_spatial_hash_visit_aabb(const ke_spatial_hash* hash,
                                float min_x, float min_y, float min_z,
                                float max_x, float max_y, float max_z,
                                ke_spatial_hash_visit_fn fn, void* user) {
    if (!hash) return;
    int32_t min_cx, min_cy, min_cz, max_cx, max_cy, max_cz;
    aabb_cell_range(hash, min_x, min_y, min_z, max_x, max_y, max_z, &min_cx, &min_cy, &min_cz, &max_cx, &max_cy, &max_cz);
    visit_cell_range(hash, min_cx, min_cy, min_cz, max_cx, max_cy, max_cz, fn, user);
}

int ke_spatial_hash_query_radius(const ke_spatial_hash* hash,
                                 float cx, float cy, float cz, float radius,
                                 int32_t* out_indices, int max_out) {
    if (!hash || !out_indices || max_out <= 0) return 0;

    int32_t min_cx, min_cy, min_cz, max_cx, max_cy, max_cz;
    radius_cell_range(hash, cx, cy, cz, radius, &min_cx, &min_cy, &min_cz, &max_cx, &max_cy, &max_cz);

    int count = 0;
    for (int32_t cz_ = min_cz; cz_ <= max_cz && count < max_out; ++cz_) {
        for (int32_t cy_ = min_cy; cy_ <= max_cy && count < max_out; ++cy_) {
            for (int32_t cx_ = min_cx; cx_ <= max_cx && count < max_out; ++cx_) {
                int32_t ci = cell_index(hash, cx_, cy_, cz_);
                int32_t idx = hash->cell_heads[ci];
                while (idx >= 0 && count < max_out) {
                    out_indices[count++] = idx;
                    idx = hash->next_indices[idx];
                }
            }
        }
    }
    return count;
}

int ke_spatial_hash_query_aabb(const ke_spatial_hash* hash,
                               float min_x, float min_y, float min_z,
                               float max_x, float max_y, float max_z,
                               int32_t* out_indices, int max_out) {
    if (!hash || !out_indices || max_out <= 0) return 0;

    int32_t min_cx, min_cy, min_cz, max_cx, max_cy, max_cz;
    aabb_cell_range(hash, min_x, min_y, min_z, max_x, max_y, max_z, &min_cx, &min_cy, &min_cz, &max_cx, &max_cy, &max_cz);

    int count = 0;
    for (int32_t cz_ = min_cz; cz_ <= max_cz && count < max_out; ++cz_) {
        for (int32_t cy_ = min_cy; cy_ <= max_cy && count < max_out; ++cy_) {
            for (int32_t cx_ = min_cx; cx_ <= max_cx && count < max_out; ++cx_) {
                int32_t ci = cell_index(hash, cx_, cy_, cz_);
                int32_t idx = hash->cell_heads[ci];
                while (idx >= 0 && count < max_out) {
                    out_indices[count++] = idx;
                    idx = hash->next_indices[idx];
                }
            }
        }
    }
    return count;
}