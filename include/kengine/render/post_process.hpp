#pragma once

#include "kengine/render/frame_graph.hpp"
#include "kengine/render/retro_types.hpp"
#include "kengine/vulkan/context.hpp"
#include "kengine/vulkan/pipeline_cache.hpp"
#include "kengine/vulkan/shader_module.hpp"
#include <vulkan/vulkan_raii.hpp>
#include <optional>
#include <string>
#include <vector>

namespace kengine {

struct TonemapPushConstants {
    float exposure           = 1.2f;
    float scanline_strength  = 0.35f;
    float w_morph            = 0.0f;
    float time               = 0.0f;
};

class PostProcessPipeline {
public:
    PostProcessPipeline(VulkanContext& ctx, PipelineCache& cache);

    void init(const std::filesystem::path& shader_dir, vk::Format swapchain_format,
              std::uint32_t frames_in_flight);
    void recreate(vk::Format swapchain_format);

    void update_descriptors(std::uint32_t frame_index,
                            vk::ImageView scene_color, vk::Format scene_color_format,
                            vk::ImageView scene_depth);

    void record_present(const vk::raii::CommandBuffer& cmd, std::uint32_t frame_index,
                        vk::ImageView swapchain_view, vk::Extent2D extent,
                        const TonemapPushConstants& pc);

    void configure(const FrameGraphConfig& config);
    void register_passes(FrameGraph& frame_graph);

    // Utility for demo/test of post effects
    TonemapPushConstants make_test_tonemap_constants(float time, float w_morph, float scan) const;

private:
    void create_pipeline(vk::Format swapchain_format);
    void create_descriptor_resources(std::uint32_t frames_in_flight);

    VulkanContext& ctx_;
    PipelineCache& cache_;
    FrameGraphConfig config_;

    std::optional<ShaderModule> shader_fullscreen_vert_;
    std::optional<ShaderModule> shader_tonemap_frag_;

    std::optional<vk::raii::DescriptorSetLayout> descriptor_layout_;
    std::optional<vk::raii::PipelineLayout> pipeline_layout_;
    std::optional<vk::raii::Pipeline> pipeline_;
    std::optional<vk::raii::DescriptorPool> descriptor_pool_;
    std::optional<vk::raii::Sampler> sampler_;
    std::vector<vk::raii::DescriptorSet> descriptor_sets_;

    std::filesystem::path shader_dir_;
    bool initialized_ = false;
};

} // namespace kengine