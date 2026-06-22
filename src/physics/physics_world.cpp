#include "kengine/physics/physics_world.hpp"

namespace kengine {

PhysicsWorld::PhysicsWorld() {
    world_ = ke_phys_world_create(1024);
    spatial_ = ke_spatial_hash_create(2.0f, 64, 64, 64, 1024);
}

PhysicsWorld::~PhysicsWorld() {
    ke_spatial_hash_destroy(spatial_);
    ke_phys_world_destroy(world_);
}

void PhysicsWorld::step(float dt) {
    ke_phys_world_step(world_, dt);
    ke_spatial_hash_rebuild(spatial_, world_);
}

void PhysicsWorld::add_body(const ke_phys_body& body) {
    ke_phys_world_add_body(world_, body);
}

void PhysicsWorld::set_gravity(float x, float y, float z, float w) {
    world_->gravity[0] = x;
    world_->gravity[1] = y;
    world_->gravity[2] = z;
    world_->gravity[3] = w;
}

} // namespace kengine