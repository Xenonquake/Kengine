#pragma once

#include "kengine/core/types.hpp"

namespace kengine {

struct Entity {
    EntityId id = kInvalidEntity;
    bool valid() const { return id != kInvalidEntity; }
};

} // namespace kengine