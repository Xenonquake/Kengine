// Shared 4D -> screen projection for retro arcade -> 4D morph

struct RetroPC {
    mat4 mvp;
    float w_slice;
    float w_morph;
    float glow_intensity;
    float time;
    vec4 hyper_rot;       // xw, yw, xy, zw
    vec2 viewport;
    float scanline_strength;
    float pixel_snap;
    float palette_index;
    float _pad;
};

vec4 rotate_plane(vec4 p, int i, int j, float angle) {
    float c = cos(angle), s = sin(angle);
    float pi = p[i], pj = p[j];
    p[i] = pi * c - pj * s;
    p[j] = pi * s + pj * c;
    return p;
}

vec4 apply_hyper_rotation(vec4 p, vec4 angles) {
    p = rotate_plane(p, 0, 3, angles.x); // xw
    p = rotate_plane(p, 1, 3, angles.y); // yw
    p = rotate_plane(p, 0, 1, angles.z); // xy
    p = rotate_plane(p, 2, 3, angles.w); // zw
    return p;
}

vec3 project_4d_to_3d(vec4 p, float w_focus) {
    float denom = max(0.001, w_focus - p.w);
    float scale = 2.0 / denom;
    return vec3(p.x, p.y, p.z) * scale;
}

vec3 project_morph(vec4 p, float w_focus, float morph) {
    vec3 pos2d = vec3(p.x, p.y, 0.0);
    vec3 proj = project_4d_to_3d(p, w_focus);
    return mix(pos2d, proj, morph);
}

vec2 snap_pixel(vec2 pos, vec2 viewport, float strength) {
    vec2 pixel = pos / viewport;
    vec2 snapped = (floor(pixel * strength) + 0.5) / strength;
    return snapped * viewport;
}

vec3 retro_palette(float idx, float palette_index) {
    int p = int(palette_index + 0.5);
    if (p == 0) {
        // Galaga: cyan/magenta/yellow
        vec3 cols[4] = vec3[](
            vec3(0.0, 1.0, 1.0),
            vec3(1.0, 0.2, 0.8),
            vec3(1.0, 1.0, 0.0),
            vec3(0.2, 0.6, 1.0)
        );
        return cols[int(mod(idx, 4.0))];
    }
    if (p == 2) {
        // Death Tank: olive + orange
        vec3 cols[4] = vec3[](
            vec3(0.4, 0.5, 0.2),
            vec3(1.0, 0.5, 0.1),
            vec3(0.8, 0.8, 0.3),
            vec3(0.3, 0.3, 0.3)
        );
        return cols[int(mod(idx, 4.0))];
    }
    // GeometryCore: neon
    vec3 cols[4] = vec3[](
        vec3(0.0, 1.0, 0.8),
        vec3(1.0, 0.1, 0.6),
        vec3(0.2, 0.5, 1.0),
        vec3(1.0, 0.9, 0.2)
    );
    return cols[int(mod(idx, 4.0))];
}