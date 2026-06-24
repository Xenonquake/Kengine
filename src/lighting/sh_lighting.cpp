#include "kengine/lighting/spherical_harmonics.hpp"
#include <cmath>
#include <cstring>

namespace kengine {

SHLightingSystem::SHLightingSystem(std::size_t probe_count) {
    probes_.resize(probe_count);
    coeff_buffer_.resize(probe_count);
    probe_positions_.resize(probe_count * 3, 0.0f);
    for (size_t i = 0; i < probe_count; ++i) {
        probes_[i].position[0] = 0.0f;
        probes_[i].position[1] = 0.0f;
        probes_[i].position[2] = 0.0f;
        probe_positions_[i*3+0] = 0.0f;
        probe_positions_[i*3+1] = 0.0f;
        probe_positions_[i*3+2] = 0.0f;
    }
}

void SHLightingSystem::setup_probe_grid(std::size_t nx, std::size_t ny, std::size_t nz,
                                        const float origin[3], const float spacing[3]) {
    std::size_t total = nx * ny * nz;
    if (total == 0) total = 1;
    probes_.resize(total);
    coeff_buffer_.resize(total);
    probe_positions_.resize(total * 3);

    grid_nx_ = nx; grid_ny_ = ny; grid_nz_ = nz;
    grid_origin_[0] = origin ? origin[0] : 0.0f;
    grid_origin_[1] = origin ? origin[1] : 0.0f;
    grid_origin_[2] = origin ? origin[2] : 0.0f;
    grid_spacing_[0] = spacing ? spacing[0] : 1.0f;
    grid_spacing_[1] = spacing ? spacing[1] : 1.0f;
    grid_spacing_[2] = spacing ? spacing[2] : 1.0f;

    std::size_t idx = 0;
    for (std::size_t z = 0; z < nz; ++z) {
        for (std::size_t y = 0; y < ny; ++y) {
            for (std::size_t x = 0; x < nx; ++x) {
                float px = grid_origin_[0] + x * grid_spacing_[0];
                float py = grid_origin_[1] + y * grid_spacing_[1];
                float pz = grid_origin_[2] + z * grid_spacing_[2];
                probes_[idx].position[0] = px;
                probes_[idx].position[1] = py;
                probes_[idx].position[2] = pz;
                probe_positions_[idx*3+0] = px;
                probe_positions_[idx*3+1] = py;
                probe_positions_[idx*3+2] = pz;
                ++idx;
            }
        }
    }
}

void SHLightingSystem::set_probe_position(std::size_t index, const float pos[3]) {
    if (index >= probes_.size()) return;
    probes_[index].position[0] = pos[0];
    probes_[index].position[1] = pos[1];
    probes_[index].position[2] = pos[2];
    if (index * 3 + 2 < probe_positions_.size()) {
        probe_positions_[index*3+0] = pos[0];
        probe_positions_[index*3+1] = pos[1];
        probe_positions_[index*3+2] = pos[2];
    }
}

void SHLightingSystem::bake_probe(std::size_t probe_index, ke_sh_radiance_fn radiance,
                                  void* userdata, std::size_t n_mu, std::size_t n_phi) {
    if (probe_index >= probes_.size()) return;

    std::size_t sample_count = n_mu * n_phi;
    std::vector<ke_sh_sample> samples(sample_count);

    if (ke_sh_samples_generate(samples.data(), n_mu, n_phi) != 0) return;

    float r[KE_SH_NUM_COEF], g[KE_SH_NUM_COEF], b[KE_SH_NUM_COEF];
    ke_sh_project(samples.data(), sample_count, radiance, userdata, r);

    /* For demo: replicate single-channel projection to RGB */
    std::memcpy(g, r, sizeof(r));
    std::memcpy(b, r, sizeof(r));

    auto& probe = probes_[probe_index];
    std::copy(std::begin(r), std::end(r), probe.sh.r.begin());
    std::copy(std::begin(g), std::end(g), probe.sh.g.begin());
    std::copy(std::begin(b), std::end(b), probe.sh.b.begin());
    coeff_buffer_[probe_index] = probe.sh;
}

void SHLightingSystem::bake_all(ke_sh_radiance_fn radiance, void* userdata) {
    for (std::size_t i = 0; i < probes_.size(); ++i) {
        bake_probe(i, radiance, userdata);
    }
}

void SHLightingSystem::prepare_samples(std::size_t n_mu, std::size_t n_phi) {
    std::size_t count = n_mu * n_phi;
    samples_.resize(count);
    ke_sh_samples_generate(samples_.data(), n_mu, n_phi);
}

/* Simple distance-weighted interpolation (good for arbitrary probe placement; for true grid can add trilinear) */
std::array<float, 3> SHLightingSystem::evaluate_at(
    const float position[3], const float normal[3]) const {
    float rgb[3] = {0.0f, 0.0f, 0.0f};
    if (probes_.empty()) return {0,0,0};

    if (probes_.size() == 1) {
        ke_sh_eval_irradiance_rgb(normal,
            probes_[0].sh.r.data(), probes_[0].sh.g.data(), probes_[0].sh.b.data(), rgb);
        return {rgb[0], rgb[1], rgb[2]};
    }

    // Find weighted sum of nearby probes
    float total_w = 0.0f;
    float acc_r[KE_SH_NUM_COEF] = {}, acc_g[KE_SH_NUM_COEF] = {}, acc_b[KE_SH_NUM_COEF] = {};

    for (size_t i = 0; i < probes_.size(); ++i) {
        const auto& p = probes_[i];
        float dx = p.position[0] - position[0];
        float dy = p.position[1] - position[1];
        float dz = p.position[2] - position[2];
        float d2 = dx*dx + dy*dy + dz*dz + 0.0001f;
        float w = 1.0f / d2;  // inverse square weight
        total_w += w;

        for (int c = 0; c < KE_SH_NUM_COEF; ++c) {
            acc_r[c] += w * p.sh.r[c];
            acc_g[c] += w * p.sh.g[c];
            acc_b[c] += w * p.sh.b[c];
        }
    }

    if (total_w > 0.0f) {
        float inv = 1.0f / total_w;
        float nr[KE_SH_NUM_COEF], ng[KE_SH_NUM_COEF], nb[KE_SH_NUM_COEF];
        for (int c = 0; c < KE_SH_NUM_COEF; ++c) {
            nr[c] = acc_r[c] * inv;
            ng[c] = acc_g[c] * inv;
            nb[c] = acc_b[c] * inv;
        }
        ke_sh_eval_irradiance_rgb(normal, nr, ng, nb, rgb);
    }
    return {rgb[0], rgb[1], rgb[2]};
}

void SHLightingSystem::set_coefficients_from_gpu(const float* flat27_per_probe, std::size_t n) {
    if (!flat27_per_probe) return;
    size_t pcnt = std::min(n, probes_.size());
    for (size_t p = 0; p < pcnt; ++p) {
        const float* src = flat27_per_probe + p * 27;
        SHCoefficients sh{};
        for (int k=0; k < KE_SH_NUM_COEF; ++k) {
            sh.r[k] = src[k];
            sh.g[k] = src[9 + k];
            sh.b[k] = src[18 + k];
        }
        coeff_buffer_[p] = sh;
        probes_[p].sh = sh;
    }
}

} // namespace kengine