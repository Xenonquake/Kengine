#pragma once

#include "kengine/core/types.hpp"
#include <typeindex>

namespace kengine {

struct IComponent {
    virtual ~IComponent() = default;
};

namespace detail {
inline ComponentTypeId& component_type_counter() {
    static ComponentTypeId counter = 0;
    return counter;
}
}

template<typename T>
struct ComponentType {
    static ComponentTypeId id() {
        static const ComponentTypeId type_id = detail::component_type_counter()++;
        return type_id;
    }
};

} // namespace kengine