#pragma once

#include "kengine/ecs/component.hpp"
#include "kengine/render/retro_types.hpp"
#include "kengine/ecs/entity.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

namespace kengine {

class Registry {
public:
    Entity create();
    void destroy(Entity entity);

    template<typename T, typename... Args>
    T& emplace(Entity entity, Args&&... args) {
        auto& pool = get_pool<T>();
        pool[entity.id] = std::make_unique<T>(std::forward<Args>(args)...);
        return *static_cast<T*>(pool[entity.id].get());
    }

    template<typename T>
    T* get(Entity entity) {
        auto& pool = get_pool<T>();
        auto it = pool.find(entity.id);
        return it != pool.end() ? static_cast<T*>(it->second.get()) : nullptr;
    }

    template<typename T>
    bool has(Entity entity) const {
        auto it = pools_.find(ComponentType<T>::id());
        if (it == pools_.end()) return false;
        const auto& pool = it->second;
        return pool.find(entity.id) != pool.end();
    }

    template<typename T, typename Fn>
    void view(Fn&& fn) {
        auto& pool = get_pool<T>();
        for (auto& [id, comp] : pool) {
            fn(Entity{id}, *static_cast<T*>(comp.get()));
        }
    }

private:
    using ComponentPool = std::unordered_map<EntityId, std::unique_ptr<IComponent>>;

    template<typename T>
    ComponentPool& get_pool() {
        return pools_[ComponentType<T>::id()];
    }

    template<typename T>
    const ComponentPool& get_pool() const {
        static ComponentPool empty;
        auto it = pools_.find(ComponentType<T>::id());
        return it != pools_.end() ? it->second : empty;
    }

    EntityId next_id_ = 0;
    std::vector<EntityId> free_list_;
    std::unordered_map<ComponentTypeId, ComponentPool> pools_;
};

// ---------------------------------------------------------------------------
// Built-in components
// ---------------------------------------------------------------------------

struct TransformComponent : IComponent {
    float position[4] = {0, 0, 0, 0}; /* 4D position */
    float rotation[4] = {0, 0, 0, 0}; /* hyperquaternion placeholder */
    float scale[3]    = {1, 1, 1};
};

struct MeshComponent : IComponent {
    std::uint32_t mesh_id = 0;
    std::uint32_t material_id = 0;
};

struct CameraComponent : IComponent {
    float fov       = 60.0f;
    float near_plane = 0.1f;
    float far_plane  = 1000.0f;
    bool  primary   = false;
};

struct PhysicsComponent : IComponent {
    float mass = 1.0f;
    bool  static_body = false;
};

struct RetroVisualComponent : IComponent {
    RetroVisualState visual;
};

} // namespace kengine