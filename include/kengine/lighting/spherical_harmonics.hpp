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

    /* Configure probes as regular 3D grid (for interpolation) */
    void setup_probe_grid(std::size_t nx, std::size_t ny, std::size_t nz,
                          const float origin[3], const float spacing[3]);

    /* Set explicit position for a probe (after grid or manual) */
    void set_probe_position(std::size_t index, const float pos[3]);

    /* Bake pass: CPU projection using Gauss-Legendre sampling */
    void bake_probe(std::size_t probe_index, ke_sh_radiance_fn radiance, void* userdata,
                    std::size_t n_mu = 8, std::size_t n_phi = 16);

    /* Bake all probes from environment radiance function */
    void bake_all(ke_sh_radiance_fn radiance, void* userdata);

    const LightProbe& probe(std::size_t index) const { return probes_[index]; }
    std::size_t probe_count() const { return probes_.size(); }

    /* Expose for GPU upload / SSBO */
    const std::vector<float>& probe_positions() const { return probe_positions_; } // flat x,y,z
    const std::vector<SHCoefficients>& coefficient_buffer() const { return coeff_buffer_; }

    /* For GPU bake result sync (updates both coeff_buffer_ and probes_) */
    void set_coefficients_from_gpu(const float* flat27_per_probe, std::size_t probe_count);

    /* Runtime: evaluate SH at normal, with trilinear / distance-weighted probe interpolation */
    std::array<float, 3> evaluate_at(const float position[3], const float normal[3]) const;

    /* GPU bake hook (samples generated on CPU, coeffs computed on GPU via sh_bake.comp) */
    void prepare_samples(std::size_t n_mu = 8, std::size_t n_phi = 16);
    const std::vector<ke_sh_sample>& samples() const { return samples_; }

private:
    std::vector<LightProbe> probes_;
    std::vector<SHCoefficients> coeff_buffer_;
    std::vector<float> probe_positions_; // 3*count
    std::vector<ke_sh_sample> samples_;
    std::size_t grid_nx_ = 0, grid_ny_ = 0, grid_nz_ = 0;
    float grid_origin_[3] = {0,0,0};
    float grid_spacing_[3] = {1,1,1};
};

} // namespace kengine