#pragma once

#include "kengine/ecs/registry.hpp"
#include "kengine/physics/physics_world.hpp"

namespace kengine {

// World / Scene manager owns the ECS registry and the physics world (ke_phys_world via wrapper).
// This centralizes the simulation state alongside entities.
// Physics hot path stays in C (verlet + spatial hash); C++ systems query/update via this.
class World {
public:
    World();
    ~World() = default;

    Registry& registry() { return registry_; }
    const Registry& registry() const { return registry_; }

    PhysicsWorld& physics() { return physics_; }
    const PhysicsWorld& physics() const { return physics_; }

    // Raw access to ke_phys_world for any C interop if needed.
    ke_phys_world* raw_physics() { return physics_.raw(); }
    const ke_phys_world* raw_physics() const { return physics_.raw(); }

    // Optional: step the physics from here (called from systems/loop)
    void step_physics(float dt);

    // Future: could hold systems list, spatial queries, etc.

private:
    Registry registry_;
    PhysicsWorld physics_;
};

} // namespace kengine
