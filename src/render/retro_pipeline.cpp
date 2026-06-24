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
        {2, 0, vk::Format::eR32Uint,      offsetof(RetroVertex2D, color)},
        {3, 0, vk::Format::eR32Uint,      offsetof(RetroVertex2D, texIndex)}
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
        {4, 0, vk::Format::eR32G32B32Sfloat,    offsetof(RetroVertex4D, vel)},  // loc4 for vel
        {1, 0, vk::Format::eR32G32Sfloat,      offsetof(RetroVertex4D, uv)},
        {2, 0, vk::Format::eR32Uint,           offsetof(RetroVertex4D, color)},
        {3, 0, vk::Format::eR32Uint,           offsetof(RetroVertex4D, texIndex)}
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
      shader_fullscreen_vert_(device, spv_path(shader_dir, "common/fullscreen.vert")),
      shader_star_mesh_vert_(device, spv_path(shader_dir, "retro/star_mesh.vert")),
      shader_star_mesh_frag_(device, spv_path(shader_dir, "retro/star_mesh.frag")) {

    createBindlessResources();
    createStarBufferLayout();

    const RetroPipelineKind kinds[] = {
        RetroPipelineKind::Sprite2D,
        RetroPipelineKind::Sprite4D,
        RetroPipelineKind::VectorGlow,
        RetroPipelineKind::ParticleAdditive,
        RetroPipelineKind::RetroComposite,
        RetroPipelineKind::StarMesh,
        RetroPipelineKind::Fullscreen
    };

    for (auto kind : kinds) {
        std::vector<vk::DescriptorSetLayout> setLayouts;
        if (bindlessLayout_) {
            setLayouts.push_back(*bindlessLayout_);
        }
        if (kind == RetroPipelineKind::StarMesh && starBufferLayout_) {
            // add set 1 for star instances (set 0 is bindless if present)
            if (setLayouts.size() == 0) {
                // if no bindless, we still need to provide empty for set 0? or adjust
                // For simplicity, assume bindless or handle in draw
            }
            setLayouts.push_back(*starBufferLayout_);  // will be set 1
        }
        layouts_.emplace(kind, GraphicsPipelineBuilder::create_layout(
            device_, setLayouts, sizeof(RetroPushConstants),
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
        desc.topology = vk::PrimitiveTopology::eLineList;  // for wireframe 3D ship model (pairs of lines)
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
    case RetroPipelineKind::StarMesh:
        // Simple 3D mesh for stars (positions only, instanced)
        desc = base_desc(formats_);
        desc.vertex_shader   = &shader_star_mesh_vert_;
        desc.fragment_shader = &shader_star_mesh_frag_;
        desc.bindings  = {{0, sizeof(float) * 3, vk::VertexInputRate::eVertex}};
        desc.attributes = {
            {0, 0, vk::Format::eR32G32B32Sfloat, 0}
        };
        desc.blend = BlendMode::Additive;
        desc.depth_test = true;
        desc.depth_write = true;
        desc.topology = vk::PrimitiveTopology::eTriangleList;
        // We will use set 1 for star instance SSBO
        break;
    }

    return GraphicsPipelineBuilder::create(device_, layouts_.at(kind), cache_.handle(), desc);
}

void RetroPipelineSet::createBindlessResources() {
    // Create bindless layout for textures (set 0, binding 0)
    vk::DescriptorSetLayoutBinding bind{};
    bind.binding = 0;
    bind.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    bind.descriptorCount = 1024;
    bind.stageFlags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex;
    bind.pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
    vk::DescriptorBindingFlags bindFlags = vk::DescriptorBindingFlagBits::ePartiallyBound
                                         | vk::DescriptorBindingFlagBits::eUpdateAfterBind
                                         | vk::DescriptorBindingFlagBits::eVariableDescriptorCount;
    flagsInfo.bindingCount = 1;
    flagsInfo.pBindingFlags = &bindFlags;

    vk::DescriptorSetLayoutCreateInfo layoutCI{};
    layoutCI.bindingCount = 1;
    layoutCI.pBindings = &bind;
    layoutCI.pNext = &flagsInfo;
    layoutCI.flags = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;

    bindlessLayout_ = device_.createDescriptorSetLayout(layoutCI);

    // Pool supporting update after bind + large
    vk::DescriptorPoolSize poolSize{};
    poolSize.type = vk::DescriptorType::eCombinedImageSampler;
    poolSize.descriptorCount = 1024;

    vk::DescriptorPoolCreateInfo poolCI{};
    poolCI.flags = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind |
                   vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    poolCI.maxSets = 1;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes = &poolSize;

    bindlessPool_ = device_.createDescriptorPool(poolCI);

    // Allocate the set
    vk::DescriptorSetVariableDescriptorCountAllocateInfo varCount{};
    uint32_t maxCount = 1024;
    varCount.descriptorSetCount = 1;
    varCount.pDescriptorCounts = &maxCount;

    vk::DescriptorSetAllocateInfo allocCI{};
    allocCI.descriptorPool = *bindlessPool_;
    allocCI.descriptorSetCount = 1;
    vk::DescriptorSetLayout rawLayout = *bindlessLayout_;
    allocCI.pSetLayouts = &rawLayout;
    allocCI.pNext = &varCount;

    auto sets = device_.allocateDescriptorSets(allocCI);
    if (!sets.empty()) {
        bindlessSet_ = std::move(sets[0]);
    }
}

void RetroPipelineSet::createStarBufferLayout() {
    // SSBO for star instance data (set 1, binding 0)
    vk::DescriptorSetLayoutBinding bind{};
    bind.binding = 0;
    bind.descriptorType = vk::DescriptorType::eStorageBuffer;
    bind.descriptorCount = 1;
    bind.stageFlags = vk::ShaderStageFlagBits::eVertex;

    vk::DescriptorSetLayoutCreateInfo layoutCI{};
    layoutCI.bindingCount = 1;
    layoutCI.pBindings = &bind;

    starBufferLayout_ = device_.createDescriptorSetLayout(layoutCI);
}

void RetroPipelineSet::updateTexture(uint32_t index, vk::ImageView view, vk::Sampler sampler) {
    if (!bindlessSet_ || index >= 1024) return;

    vk::DescriptorImageInfo imgInfo{};
    imgInfo.sampler = sampler;
    imgInfo.imageView = view;
    imgInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::WriteDescriptorSet write{};
    write.dstSet = *bindlessSet_;
    write.dstBinding = 0;
    write.dstArrayElement = index;
    write.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    write.descriptorCount = 1;
    write.pImageInfo = &imgInfo;

    device_.updateDescriptorSets({write}, {});
}

} // namespace kengine
