#pragma once

#include <cstdint>

namespace kengine {

enum class RetroStyle {
    Galaga,
    GeometryCore,
    DeathTank,
};

struct RetroVisualState {
    RetroStyle style            = RetroStyle::GeometryCore;
    float w_morph               = 0.0f;
    float w_slice               = 0.0f;
    float glow_intensity        = 1.5f;
    float scanline_strength     = 0.35f;
    float pixel_snap            = 64.0f;
};

// Per-instance data for GPU culling + indirect draw (for enemies, bullets, dense particles).
// Positions are in the projected 3D world space used by the camera.
struct GpuSpriteInstance {
    float    pos[3];
    float    scale;
    uint32_t color;
    float    vel[3];
    uint32_t type;
};

} // namespace kengine