#pragma once

#include "kengine/render/frame_graph.hpp"
#include "kengine/render/retro_types.hpp"
#include "kengine/vulkan/context.hpp"
#include "kengine/vulkan/gpu_image.hpp"
#include "kengine/vulkan/pipeline_cache.hpp"
#include "kengine/vulkan/shader_module.hpp"
#include "kengine/vulkan/vma_allocator.hpp"
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
                            vk::ImageView scene_depth,
                            vk::ImageView bloom = {});  // optional bloom for composite

    void record_present(const vk::raii::CommandBuffer& cmd, std::uint32_t frame_index,
                        vk::ImageView swapchain_view, vk::Extent2D extent,
                        const TonemapPushConstants& pc);

    void configure(const FrameGraphConfig& config);
    void register_passes(FrameGraph& frame_graph);

    void on_resize(vk::Extent2D extent);  // allocate bloom etc at correct size

    // Utility for demo/test of post effects
    TonemapPushConstants make_test_tonemap_constants(float time, float w_morph, float scan) const;

    // Post pass recording (called via frame graph lambdas)
    void record_bloom_threshold(const vk::raii::CommandBuffer& cmd, std::uint32_t frame_index, vk::Extent2D extent);
    void record_bloom_blur(const vk::raii::CommandBuffer& cmd, std::uint32_t frame_index, vk::Extent2D extent);
    // TODO: dof, taa, sharpen record methods

    // Expose for wiring / debug
    GpuImage& bloom_input() { return bloom_input_; }
    GpuImage& bloom_output() { return bloom_output_; }

private:
    void create_pipeline(vk::Format swapchain_format);
    void create_descriptor_resources(std::uint32_t frames_in_flight);
    void create_bloom_compute_pipelines();
    void create_post_resources(vk::Extent2D extent);
    void destroy_post_resources();

    VulkanContext& ctx_;
    PipelineCache& cache_;
    FrameGraphConfig config_;

    std::optional<ShaderModule> shader_fullscreen_vert_;
    std::optional<ShaderModule> shader_tonemap_frag_;

    // Bloom compute shaders
    std::optional<ShaderModule> shader_bloom_threshold_;
    std::optional<ShaderModule> shader_bloom_blur_;

    std::optional<vk::raii::DescriptorSetLayout> descriptor_layout_;
    std::optional<vk::raii::PipelineLayout> pipeline_layout_;
    std::optional<vk::raii::Pipeline> pipeline_;
    std::optional<vk::raii::DescriptorPool> descriptor_pool_;
    std::optional<vk::raii::Sampler> sampler_;
    std::vector<vk::raii::DescriptorSet> descriptor_sets_;

    // Bloom storage image descriptors (compute passes use storage images)
    std::optional<vk::raii::DescriptorPool> bloom_storage_pool_;
    std::optional<vk::raii::DescriptorSet> bloom_storage_set_;

    // Compute pipeline layouts / pipelines for bloom
    std::optional<vk::raii::DescriptorSetLayout> bloom_compute_desc_layout_;
    std::optional<vk::raii::PipelineLayout> bloom_compute_layout_;
    std::optional<vk::raii::Pipeline> bloom_threshold_pipeline_;
    std::optional<vk::raii::Pipeline> bloom_blur_pipeline_;

    // Intermediate post targets (currently bloom ping-pong). In real impl we would have per-frame or managed better.
    GpuImage bloom_input_;
    GpuImage bloom_output_;

    std::filesystem::path shader_dir_;
    bool initialized_ = false;
    vk::Extent2D current_extent_{};
};

} // namespace kengine