#include "kengine/render/pipeline_manager.hpp"
#include "mat4.h"
#include <cstring>
#include <cmath>

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
    const RetroPipelineState& state, const Camera4D& cam,
    float time, vk::Extent2D extent) const {

    RetroPushConstants pc{};
    cam.fill_push_constants(pc, state, time, extent.width, extent.height);
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