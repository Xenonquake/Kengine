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

    void step(float dt);
    void add_body(const ke_phys_body& body);
    void set_gravity(float x, float y, float z, float w = 0.0f);

private:
    ke_phys_world* world_ = nullptr;
    ke_spatial_hash* spatial_ = nullptr;
};

} // namespace kengine