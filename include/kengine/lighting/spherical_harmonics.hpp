#pragma once

#include "sh_basis.h"
#include "sh_project.h"
#include "sh_eval.h"
#include <array>
#include <cstddef>
#include <vector>

namespace kengine {

struct SHCoefficients {
    std::array<float, KE_SH_NUM_COEF> r{};
    std::array<float, KE_SH_NUM_COEF> g{};
    std::array<float, KE_SH_NUM_COEF> b{};
};

struct LightProbe {
    float position[3] = {0, 0, 0};
    SHCoefficients sh;
};

class SHLightingSystem {
public:
    explicit SHLightingSystem(std::size_t probe_count = 64);

    /* Bake pass: CPU projection using Gauss-Legendre sampling */
    void bake_probe(std::size_t probe_index, ke_sh_radiance_fn radiance, void* userdata,
                    std::size_t n_mu = 8, std::size_t n_phi = 16);

    /* Bake all probes from environment radiance function */
    void bake_all(ke_sh_radiance_fn radiance, void* userdata);

    const LightProbe& probe(std::size_t index) const { return probes_[index]; }
    std::size_t probe_count() const { return probes_.size(); }

    /* Runtime: evaluate SH at normal, with trilinear probe interpolation stub */
    std::array<float, 3> evaluate_at(const float position[3], const float normal[3]) const;

    const std::vector<SHCoefficients>& coefficient_buffer() const { return coeff_buffer_; }

private:
    std::vector<LightProbe> probes_;
    std::vector<SHCoefficients> coeff_buffer_;
};

} // namespace kengine