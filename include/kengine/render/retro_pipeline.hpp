#pragma once

#include "kengine/render/retro_types.hpp"
#include "kengine/vulkan/pipeline_builder.hpp"
#include "kengine/vulkan/pipeline_cache.hpp"
#include "kengine/vulkan/shader_module.hpp"
#include <vulkan/vulkan_raii.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <unordered_map>

namespace kengine {

enum class RetroPipelineKind {
    Sprite2D,
    Sprite4D,
    VectorGlow,
    ParticleAdditive,
    RetroComposite,
    Fullscreen
};

using RetroPipelineState = RetroVisualState;

/* Must match GLSL push_constant layout (128 bytes). */
struct RetroPushConstants {
    float mvp[16];
    float w_slice;
    float w_morph;
    float glow_intensity;
    float time;
    float hyper_rot[4];   /* xw, yw, xy, zw */
    float viewport[2];
    float scanline_strength;
    float pixel_snap;
    float palette_index;  /* 0=Galaga, 1=GeometryCore, 2=DeathTank */
    float _pad;
};

struct RetroVertex2D {
    float pos[2];
    float uv[2];
    std::uint32_t color;
    std::uint32_t texIndex = 0;  // bindless texture index
};

struct RetroVertex4D {
    float pos[4];
    float uv[2];
    std::uint32_t color;
    std::uint32_t texIndex = 0;  // bindless texture index
};

class RetroPipelineSet {
public:
    RetroPipelineSet(const vk::raii::Device& device, PipelineCache& cache,
                     const std::filesystem::path& shader_dir,
                     const DynamicRenderingFormats& formats);

    vk::raii::Pipeline& get(RetroPipelineKind kind);
    vk::raii::PipelineLayout& layout(RetroPipelineKind kind);

    const DynamicRenderingFormats& formats() const { return formats_; }

    // Bindless texture support
    vk::DescriptorSetLayout bindlessLayout() const { return bindlessLayout_ ? *bindlessLayout_ : vk::DescriptorSetLayout{}; }
    const vk::raii::DescriptorSet* bindlessSet() const { return bindlessSet_ ? &*bindlessSet_ : nullptr; }
    void updateTexture(uint32_t index, vk::ImageView view, vk::Sampler sampler);

private:
    vk::raii::Pipeline create_pipeline(RetroPipelineKind kind);

    const vk::raii::Device& device_;
    PipelineCache& cache_;
    std::filesystem::path shader_dir_;
    DynamicRenderingFormats formats_;

    ShaderModule shader_sprite2d_vert_;
    ShaderModule shader_sprite2d_frag_;
    ShaderModule shader_sprite4d_vert_;
    ShaderModule shader_sprite4d_frag_;
    ShaderModule shader_vector_vert_;
    ShaderModule shader_vector_frag_;
    ShaderModule shader_particle_vert_;
    ShaderModule shader_particle_frag_;
    ShaderModule shader_composite_frag_;
    ShaderModule shader_fullscreen_vert_;

    std::unordered_map<RetroPipelineKind, vk::raii::Pipeline> pipelines_;
    std::unordered_map<RetroPipelineKind, vk::raii::PipelineLayout> layouts_;

    // Bindless for texture atlases
    std::optional<vk::raii::DescriptorSetLayout> bindlessLayout_;
    std::optional<vk::raii::DescriptorPool> bindlessPool_;
    std::optional<vk::raii::DescriptorSet> bindlessSet_;
    std::vector<vk::raii::Sampler> textureSamplers_; // keep alive

    void createBindlessResources();
};

} // namespace kengine