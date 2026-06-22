#pragma once

#include <cstdint>
#include <cstddef>

namespace kengine {

using EntityId = std::uint32_t;
constexpr EntityId kInvalidEntity = UINT32_MAX;

using ComponentTypeId = std::uint32_t;

} // namespace kengine