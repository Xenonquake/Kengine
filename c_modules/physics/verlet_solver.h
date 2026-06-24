#pragma once

#include "../math/vec4.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ke_phys_body {
    ke_vec4 position;    /* xyz = 3D position, w = 4th dimension coordinate */
    ke_vec4 velocity;
    ke_vec4 acceleration;
    float   mass;
    float   inverse_mass;
    uint32_t flags;
} ke_phys_body;

#define KE_PHYS_STATIC    (1u << 0)
#define KE_PHYS_KINEMATIC (1u << 1)

typedef struct ke_phys_world {
    ke_phys_body* bodies;
    size_t        count;
    size_t        capacity;
    float         gravity[4]; /* 4D gravity vector */
    float         damping;
} ke_phys_world;

ke_phys_world* ke_phys_world_create(size_t initial_capacity);
void           ke_phys_world_destroy(ke_phys_world* world);
int            ke_phys_world_add_body(ke_phys_world* world, ke_phys_body body);
void           ke_phys_world_step(ke_phys_world* world, float dt);

#ifdef __cplusplus
}
#endif