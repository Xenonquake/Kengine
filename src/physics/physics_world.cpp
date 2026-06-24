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

int PhysicsWorld::create_body(const ke_phys_body& body) {
    if (!world_) return -1;
    size_t idx = world_->count;
    if (ke_phys_world_add_body(world_, body) != 0) return -1;
    return static_cast<int>(idx);
}

ke_phys_body* PhysicsWorld::get_body(int index) {
    if (!world_ || index < 0 || static_cast<size_t>(index) >= world_->count) return nullptr;
    return &world_->bodies[index];
}

const ke_phys_body* PhysicsWorld::get_body(int index) const {
    if (!world_ || index < 0 || static_cast<size_t>(index) >= world_->count) return nullptr;
    return &world_->bodies[index];
}

size_t PhysicsWorld::body_count() const {
    return world_ ? world_->count : 0;
}

void PhysicsWorld::set_velocity(int index, float vx, float vy, float vz, float vw) {
    if (auto* b = get_body(index)) {
        b->velocity = ke_vec4_make(vx, vy, vz, vw);
    }
}

void PhysicsWorld::add_acceleration(int index, float ax, float ay, float az, float aw) {
    if (auto* b = get_body(index)) {
        b->acceleration = ke_vec4_add(b->acceleration, ke_vec4_make(ax, ay, az, aw));
    }
}

void PhysicsWorld::accumulate_force(int index, float fx, float fy, float fz, float fw) {
    if (auto* b = get_body(index)) {
        if (b->inverse_mass > 0.0f) {
            b->acceleration = ke_vec4_add(b->acceleration, ke_vec4_make(
                fx * b->inverse_mass, fy * b->inverse_mass,
                fz * b->inverse_mass, fw * b->inverse_mass));
        }
    }
}

void PhysicsWorld::add_body(const ke_phys_body& body) {
    create_body(body);
}

void PhysicsWorld::set_gravity(float x, float y, float z, float w) {
    world_->gravity[0] = x;
    world_->gravity[1] = y;
    world_->gravity[2] = z;
    world_->gravity[3] = w;
}

// --- Spatial queries ---

int PhysicsWorld::query_radius(float cx, float cy, float cz, float radius,
                               int32_t* out, int max_out) const {
    if (!spatial_) return 0;
    return ke_spatial_hash_query_radius(spatial_, cx, cy, cz, radius, out, max_out);
}

int PhysicsWorld::query_aabb(float minx, float miny, float minz,
                             float maxx, float maxy, float maxz,
                             int32_t* out, int max_out) const {
    if (!spatial_) return 0;
    return ke_spatial_hash_query_aabb(spatial_, minx, miny, minz, maxx, maxy, maxz, out, max_out);
}

void PhysicsWorld::visit_radius(float cx, float cy, float cz, float radius,
                                SpatialVisitFn fn, void* user) const {
    if (!spatial_ || !fn) return;
    ke_spatial_hash_visit_radius(spatial_, cx, cy, cz, radius, fn, user);
}

void PhysicsWorld::visit_aabb(float minx, float miny, float minz,
                              float maxx, float maxy, float maxz,
                              SpatialVisitFn fn, void* user) const {
    if (!spatial_ || !fn) return;
    ke_spatial_hash_visit_aabb(spatial_, minx, miny, minz, maxx, maxy, maxz, fn, user);
}

} // namespace kengine