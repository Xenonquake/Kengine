#pragma once

#include "verlet_solver.h"
#include "spatial_hash.h"
#include <memory>

namespace kengine {

class PhysicsWorld {
public:
    PhysicsWorld();
    ~PhysicsWorld();

    ke_phys_world* raw() { return world_; }
    const ke_phys_world* raw() const { return world_; }

    void step(float dt);

    // Body management: returns body index (>=0) or -1 on failure.
    int  create_body(const ke_phys_body& body);
    ke_phys_body* get_body(int index);
    const ke_phys_body* get_body(int index) const;

    // Convenience
    size_t body_count() const;

    void set_velocity(int index, float vx, float vy, float vz, float vw = 0.0f);
    void add_acceleration(int index, float ax, float ay, float az, float aw = 0.0f);
    // Accumulate force as acceleration (a += F * inv_mass)
    void accumulate_force(int index, float fx, float fy, float fz, float fw = 0.0f);

    // Deprecated compat
    void add_body(const ke_phys_body& body);

    void set_gravity(float x, float y, float z, float w = 0.0f);

    // --- Spatial hash access (3D on xyz; rebuilt automatically after each step) ---
    ke_spatial_hash* spatial_hash() { return spatial_; }
    const ke_spatial_hash* spatial_hash() const { return spatial_; }

    // Query helpers (return indices into bodies; 0..body_count()-1 at time of last rebuild)
    int query_radius(float cx, float cy, float cz, float radius, int32_t* out, int max_out) const;
    int query_aabb(float minx, float miny, float minz,
                   float maxx, float maxy, float maxz,
                   int32_t* out, int max_out) const;

    // Callback visit (C-style, zero-alloc)
    using SpatialVisitFn = void (*)(int32_t body_index, void* user);
    void visit_radius(float cx, float cy, float cz, float radius, SpatialVisitFn fn, void* user) const;
    void visit_aabb(float minx, float miny, float minz,
                    float maxx, float maxy, float maxz,
                    SpatialVisitFn fn, void* user) const;

private:
    ke_phys_world* world_ = nullptr;
    ke_spatial_hash* spatial_ = nullptr;
};

} // namespace kengine