#include "kengine/render/pipeline_manager.hpp"
#include "mat4.h"
#include <cstring>

namespace kengine {

PipelineManager::PipelineManager(VulkanContext& ctx, Swapchain& swapchain,
                                 const std::filesystem::path& shader_dir,
                                 const std::filesystem::path& cache_path)
    : ctx_(ctx), swapchain_(swapchain), shader_dir_(shader_dir), cache_path_(cache_path) {}

DynamicRenderingFormats PipelineManager::scene_formats() const {
    DynamicRenderingFormats f;
    f.color_format = SceneTargetSet::kSceneColorFormat;
    f.depth_format = SceneTargetSet::kSceneDepthFormat;
    f.has_depth    = true;
    return f;
}

void PipelineManager::warmup(vk::Extent2D extent) {
    pipeline_cache_ = std::make_unique<PipelineCache>(
        ctx_.device(), ctx_.physical_device(), cache_path_);

    scene_targets_ = std::make_unique<SceneTargetSet>(
        ctx_.device(), ctx_.allocator().handle(), ctx_.frames_in_flight());
    scene_targets_->recreate(extent);

    scene_pipelines_ = std::make_unique<RetroPipelineSet>(
        ctx_.device(), *pipeline_cache_, shader_dir_, scene_formats());
}

void PipelineManager::recreate(vk::Extent2D extent) {
    if (scene_targets_) scene_targets_->recreate(extent);
    scene_pipelines_ = std::make_unique<RetroPipelineSet>(
        ctx_.device(), *pipeline_cache_, shader_dir_, scene_formats());
}

void PipelineManager::save_cache() {
    if (pipeline_cache_) pipeline_cache_->save();
}

RetroPushConstants PipelineManager::make_push_constants(
    const RetroPipelineState& state, float time, vk::Extent2D extent) const {

    RetroPushConstants pc{};
    float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    ke_mat4 proj = ke_mat4_perspective(0.785398f, aspect, 0.1f, 100.0f);
    ke_mat4 view = ke_mat4_look_at(
        ke_vec3_make(0, 0, 3),
        ke_vec3_make(0, 0, 0),
        ke_vec3_make(0, 1, 0));
    ke_mat4 mvp = ke_mat4_mul(proj, view);
    std::memcpy(pc.mvp, mvp.m, sizeof(pc.mvp));

    pc.w_slice            = state.w_slice;
    pc.w_morph            = state.w_morph;
    pc.glow_intensity     = state.glow_intensity;
    pc.time               = time;
    pc.hyper_rot[0]       = 0.4f;
    pc.hyper_rot[1]       = 0.3f;
    pc.hyper_rot[2]       = 0.2f;
    pc.hyper_rot[3]       = 0.5f;
    pc.viewport[0]        = static_cast<float>(extent.width);
    pc.viewport[1]        = static_cast<float>(extent.height);
    pc.scanline_strength  = state.scanline_strength;
    pc.pixel_snap         = state.pixel_snap;
    pc.palette_index      = static_cast<float>(state.style);
    return pc;
}

void PipelineManager::bind_scene(const vk::raii::CommandBuffer& cmd, RetroPipelineKind kind) const {
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, scene_pipelines_->get(kind));
}

void PipelineManager::push_scene_state(const vk::raii::CommandBuffer& cmd, RetroPipelineKind kind,
                                       const RetroPushConstants& pc) const {
    cmd.pushConstants<RetroPushConstants>(
        scene_pipelines_->layout(kind),
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        0, pc);
}

} // namespace kengine