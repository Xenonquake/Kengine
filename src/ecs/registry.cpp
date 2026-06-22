#include "kengine/ecs/registry.hpp"

namespace kengine {

Entity Registry::create() {
    EntityId id;
    if (!free_list_.empty()) {
        id = free_list_.back();
        free_list_.pop_back();
    } else {
        id = next_id_++;
    }
    return Entity{id};
}

void Registry::destroy(Entity entity) {
    if (!entity.valid()) return;
    for (auto& [type_id, pool] : pools_) {
        pool.erase(entity.id);
    }
    free_list_.push_back(entity.id);
}

} // namespace kengine