#pragma once

#include "vec4.h"  // from c_modules/math via include paths

namespace kengine {

struct GameEntity {
    ke_vec4 pos = {0, 0, 0, 0};           // x,y,z,w — full 4D position
    ke_vec4 vel = {0, 0, 0, 0};           // velocity in 4D
    float yaw = 0.0f;                     // heading (for ship orientation)
    float health = 100.0f;
    int type = 0;                         // 0=player, 1=enemy_grunt, 2=bullet, 3=powerup, 4=debris, 5=structure
    bool active = true;
    float lifetime = -1.0f;               // for bullets/particles/debris (-1 = infinite)
    float scale = 1.0f;
    uint32_t color = 0xFFFFFFFFu;
};

} // namespace kengine
