#pragma once

#include "kengine/render/retro_pipeline.hpp"
#include "kengine/render/scene_targets.hpp"
#include "kengine/vulkan/context.hpp"
#include "kengine/vulkan/pipeline_cache.hpp"
#include "kengine/vulkan/swapchain.hpp"
#include <filesystem>
#include <memory>

namespace kengine {

class PipelineManager {
public:
    PipelineManager(VulkanContext& ctx, Swapchain& swapchain,
                    const std::filesystem::path& shader_dir,
                    const std::filesystem::path& cache_path);

    void warmup(vk::Extent2D extent);
    void recreate(vk::Extent2D extent);

    PipelineCache& cache() { return *pipeline_cache_; }
    RetroPipelineSet& scene() { return *scene_pipelines_; }
    SceneTargetSet& scene_targets() { return *scene_targets_; }

    RetroPushConstants make_push_constants(const RetroPipelineState& state,
                                           float time, vk::Extent2D extent) const;

    void bind_scene(const vk::raii::CommandBuffer& cmd, RetroPipelineKind kind) const;
    void push_scene_state(const vk::raii::CommandBuffer& cmd, RetroPipelineKind kind,
                          const RetroPushConstants& pc) const;

    void save_cache();

private:
    DynamicRenderingFormats scene_formats() const;

    VulkanContext& ctx_;
    Swapchain& swapchain_;
    std::filesystem::path shader_dir_;
    std::filesystem::path cache_path_;

    std::unique_ptr<PipelineCache> pipeline_cache_;
    std::unique_ptr<RetroPipelineSet> scene_pipelines_;
    std::unique_ptr<SceneTargetSet> scene_targets_;
};

} // namespace kengine