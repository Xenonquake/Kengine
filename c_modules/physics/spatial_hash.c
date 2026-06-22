#include "spatial_hash.h"
#include <stdlib.h>
#include <string.h>

static int32_t cell_index(const ke_spatial_hash* h, int32_t x, int32_t y, int32_t z) {
    return x + y * h->grid_x + z * h->grid_x * h->grid_y;
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