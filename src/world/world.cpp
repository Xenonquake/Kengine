#include "kengine/world/world.hpp"

namespace kengine {

World::World() {
    // PhysicsWorld internally owns ke_phys_world + spatial hash.
    // Hot path (step, queries) stays in C modules.
}

void World::step_physics(float dt) {
    physics_.step(dt);
}

} // namespace kengine
