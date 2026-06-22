#include "verlet_solver.h"
#include <stdlib.h>
#include <string.h>

ke_phys_world* ke_phys_world_create(size_t initial_capacity) {
    ke_phys_world* w = (ke_phys_world*)calloc(1, sizeof(ke_phys_world));
    if (!w) return NULL;
    w->capacity = initial_capacity > 0 ? initial_capacity : 256;
    w->bodies   = (ke_phys_body*)calloc(w->capacity, sizeof(ke_phys_body));
    if (!w->bodies) { free(w); return NULL; }
    w->gravity[1] = -9.81f;
    w->damping    = 0.999f;
    return w;
}

void ke_phys_world_destroy(ke_phys_world* world) {
    if (!world) return;
    free(world->bodies);
    free(world);
}

int ke_phys_world_add_body(ke_phys_world* world, ke_phys_body body) {
    if (!world) return -1;
    if (world->count >= world->capacity) {
        size_t new_cap = world->capacity * 2;
        ke_phys_body* new_bodies = (ke_phys_body*)realloc(world->bodies, new_cap * sizeof(ke_phys_body));
        if (!new_bodies) return -1;
        world->bodies   = new_bodies;
        world->capacity = new_cap;
    }
    if (body.mass > 0.0f) body.inverse_mass = 1.0f / body.mass;
    world->bodies[world->count++] = body;
    return 0;
}

void ke_phys_world_step(ke_phys_world* world, float dt) {
    if (!world || dt <= 0.0f) return;

    const float dt2 = dt * dt;
    ke_vec4 grav = ke_vec4_make(world->gravity[0], world->gravity[1],
                                world->gravity[2], world->gravity[3]);

    for (size_t i = 0; i < world->count; ++i) {
        ke_phys_body* b = &world->bodies[i];
        if (b->flags & KE_PHYS_STATIC) continue;

        ke_vec4 total_accel = ke_vec4_add(grav, b->acceleration);

        /* Velocity Verlet: low latency, good stability */
        ke_vec4 new_pos = ke_vec4_add(b->position,
            ke_vec4_add(
                ke_vec4_scale(b->velocity, dt),
                ke_vec4_scale(total_accel, 0.5f * dt2)
            )
        );

        ke_vec4 new_vel = ke_vec4_scale(
            ke_vec4_add(b->velocity, ke_vec4_scale(total_accel, dt)),
            world->damping
        );

        b->position     = new_pos;
        b->velocity     = new_vel;
        b->acceleration = (ke_vec4){0, 0, 0, 0};
    }
}