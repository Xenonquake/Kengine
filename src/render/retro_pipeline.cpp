#include "kengine/render/retro_pipeline.hpp"
#include <cstring>
#include <stdexcept>

namespace kengine {

static std::filesystem::path spv_path(const std::filesystem::path& dir, const char* rel) {
    return dir / (std::string(rel) + ".spv");
}

static GraphicsPipelineDesc base_desc(const DynamicRenderingFormats& formats) {
    GraphicsPipelineDesc d;
    d.rendering = formats;
    d.push_constant_size = sizeof(RetroPushConstants);
    return d;
}

static GraphicsPipelineDesc desc_sprite2d(const DynamicRenderingFormats& formats,
    const ShaderModule& vert, const ShaderModule& frag) {
    auto d = base_desc(formats);
    d.vertex_shader   = &vert;
    d.fragment_shader = &frag;
    d.bindings  = {{0, sizeof(RetroVertex2D), vk::VertexInputRate::eVertex}};
    d.attributes = {
        {0, 0, vk::Format::eR32G32Sfloat, offsetof(RetroVertex2D, pos)},
        {1, 0, vk::Format::eR32G32Sfloat, offsetof(RetroVertex2D, uv)},
        {2, 0, vk::Format::eR32Uint,      offsetof(RetroVertex2D, color)}
    };
    d.blend = BlendMode::Alpha;
    d.depth_test = false;
    d.depth_write = false;
    return d;
}

static GraphicsPipelineDesc desc_sprite4d(const DynamicRenderingFormats& formats,
    const ShaderModule& vert, const ShaderModule& frag, BlendMode blend) {
    auto d = base_desc(formats);
    d.vertex_shader   = &vert;
    d.fragment_shader = &frag;
    d.bindings  = {{0, sizeof(RetroVertex4D), vk::VertexInputRate::eVertex}};
    d.attributes = {
        {0, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(RetroVertex4D, pos)},
        {1, 0, vk::Format::eR32G32Sfloat,      offsetof(RetroVertex4D, uv)},
        {2, 0, vk::Format::eR32Uint,           offsetof(RetroVertex4D, color)}
    };
    d.blend = blend;
    d.depth_test = true;
    d.depth_write = true;
    return d;
}

static GraphicsPipelineDesc desc_fullscreen(const DynamicRenderingFormats& formats,
    const ShaderModule& vert, const ShaderModule& frag, BlendMode blend) {
    auto d = base_desc(formats);
    d.vertex_shader   = &vert;
    d.fragment_shader = &frag;
    d.depth_test  = false;
    d.depth_write = false;
    d.blend       = blend;
    return d;
}

RetroPipelineSet::RetroPipelineSet(const vk::raii::Device& device, PipelineCache& cache,
                                   const std::filesystem::path& shader_dir,
                                   const DynamicRenderingFormats& formats)
    : device_(device), cache_(cache), shader_dir_(shader_dir), formats_(formats),
      shader_sprite2d_vert_(device, spv_path(shader_dir, "retro/sprite_2d.vert")),
      shader_sprite2d_frag_(device, spv_path(shader_dir, "retro/sprite_2d.frag")),
      shader_sprite4d_vert_(device, spv_path(shader_dir, "retro/sprite_4d.vert")),
      shader_sprite4d_frag_(device, spv_path(shader_dir, "retro/sprite_4d.frag")),
      shader_vector_vert_(device, spv_path(shader_dir, "retro/vector_glow.vert")),
      shader_vector_frag_(device, spv_path(shader_dir, "retro/vector_glow.frag")),
      shader_particle_vert_(device, spv_path(shader_dir, "retro/particle_additive.vert")),
      shader_particle_frag_(device, spv_path(shader_dir, "retro/particle_additive.frag")),
      shader_composite_frag_(device, spv_path(shader_dir, "retro/retro_composite.frag")),
      shader_fullscreen_vert_(device, spv_path(shader_dir, "common/fullscreen.vert")) {

    const RetroPipelineKind kinds[] = {
        RetroPipelineKind::Sprite2D,
        RetroPipelineKind::Sprite4D,
        RetroPipelineKind::VectorGlow,
        RetroPipelineKind::ParticleAdditive,
        RetroPipelineKind::RetroComposite,
        RetroPipelineKind::Fullscreen
    };

    for (auto kind : kinds) {
        layouts_.emplace(kind, GraphicsPipelineBuilder::create_layout(
            device_, {}, sizeof(RetroPushConstants),
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment));
        pipelines_.emplace(kind, create_pipeline(kind));
    }
}

vk::raii::Pipeline& RetroPipelineSet::get(RetroPipelineKind kind) {
    return pipelines_.at(kind);
}

vk::raii::PipelineLayout& RetroPipelineSet::layout(RetroPipelineKind kind) {
    return layouts_.at(kind);
}

vk::raii::Pipeline RetroPipelineSet::create_pipeline(RetroPipelineKind kind) {
    GraphicsPipelineDesc desc;

    switch (kind) {
    case RetroPipelineKind::Sprite2D:
        desc = desc_sprite2d(formats_, shader_sprite2d_vert_, shader_sprite2d_frag_);
        break;
    case RetroPipelineKind::Sprite4D:
        desc = desc_sprite4d(formats_, shader_sprite4d_vert_, shader_sprite4d_frag_, BlendMode::Additive);
        break;
    case RetroPipelineKind::VectorGlow:
        desc = desc_sprite4d(formats_, shader_vector_vert_, shader_vector_frag_, BlendMode::Additive);
        desc.topology = vk::PrimitiveTopology::eLineStrip;
        desc.depth_test = false;
        desc.depth_write = false;
        break;
    case RetroPipelineKind::ParticleAdditive:
        desc = desc_sprite4d(formats_, shader_particle_vert_, shader_particle_frag_, BlendMode::Additive);
        desc.depth_test = false;
        desc.depth_write = false;
        break;
    case RetroPipelineKind::RetroComposite:
        desc = desc_fullscreen(formats_, shader_fullscreen_vert_, shader_composite_frag_, BlendMode::Opaque);
        break;
    case RetroPipelineKind::Fullscreen:
        // Use a compatible frag that only consumes loc 0 (vUV) from fullscreen.vert
        desc = desc_fullscreen(formats_, shader_fullscreen_vert_, shader_composite_frag_, BlendMode::Opaque);
        break;
    }

    return GraphicsPipelineBuilder::create(device_, layouts_.at(kind), cache_.handle(), desc);
}

} // namespace kengine