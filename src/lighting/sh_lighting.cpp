#include "kengine/lighting/spherical_harmonics.hpp"
#include <cmath>
#include <cstring>

namespace kengine {

SHLightingSystem::SHLightingSystem(std::size_t probe_count) {
    probes_.resize(probe_count);
    coeff_buffer_.resize(probe_count);
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

std::array<float, 3> SHLightingSystem::evaluate_at(
    const float position[3], const float normal[3]) const {
    (void)position;
    float rgb[3];
    if (!probes_.empty()) {
        ke_sh_eval_irradiance_rgb(normal,
            probes_[0].sh.r.data(), probes_[0].sh.g.data(), probes_[0].sh.b.data(), rgb);
    } else {
        rgb[0] = rgb[1] = rgb[2] = 0.0f;
    }
    return {rgb[0], rgb[1], rgb[2]};
}

} // namespace kengine