#pragma once

#include "kengine/core/types.hpp"
#include <typeindex>

namespace kengine {

struct IComponent {
    virtual ~IComponent() = default;
};

template<typename T>
struct ComponentType {
    static ComponentTypeId id() {
        static ComponentTypeId type_id = next_id_++;
        return type_id;
    }

private:
    static inline ComponentTypeId next_id_ = 0;
};

} // namespace kengine