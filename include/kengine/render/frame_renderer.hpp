#pragma once

#include "kengine/render/camera_4d.hpp"
#include "kengine/render/frame_graph.hpp"
#include "kengine/render/pipeline_manager.hpp"
#include "kengine/render/post_process.hpp"
#include "kengine/render/retro_pipeline.hpp"
#include "kengine/vulkan/context.hpp"
#include "kengine/vulkan/swapchain.hpp"
#include <cstdint>
#include <vulkan/vulkan_raii.hpp>
#include <vector>

namespace kengine {

class FrameRenderer {
public:
    FrameRenderer(VulkanContext& ctx, Swapchain& swapchain,
                  PipelineManager& pipelines, PostProcessPipeline& post,
                  FrameGraph& frame_graph);
    ~FrameRenderer();

    void bind_frame_graph_passes();
    void render_frame(float time, const RetroPipelineState& state, const Camera4D& cam);

    bool needs_resize(vk::Extent2D extent) const;
    void on_resize(vk::Extent2D extent);
    void recreate_swapchain_sync();  // recreate per-image semaphores/fences when swapchain image count changes

private:
    void create_sync_objects();
    void create_command_buffers();
    void create_geometry_buffers();
    void update_ship_geometry(float time);
    void record_scene_pass(std::uint32_t sync_index, float time,
                           const RetroPipelineState& state, const Camera4D& cam);
    void record_present_pass(std::uint32_t sync_index, std::uint32_t image_index,
                             float time, const RetroPipelineState& state);

    VulkanContext& ctx_;
    Swapchain& swapchain_;
    PipelineManager& pipelines_;
    PostProcessPipeline& post_;
    FrameGraph& frame_graph_;

    std::uint32_t sync_index_ = 0;
    std::uint32_t active_sync_index_ = 0;
    std::uint32_t active_image_index_ = 0;
    float pending_time_ = 0.0f;
    RetroPipelineState pending_state_{};
    Camera4D pending_cam_{};
    vk::Extent2D last_extent_{};

    std::uint32_t frames_in_flight_;
    std::uint32_t swapchain_image_count_ = 0;

    // Acquire semaphores are indexed by frame-in-flight (used for acquire + submit wait)
    std::vector<vk::raii::Semaphore> acquire_semaphores_;
    // Render finished / present semaphores indexed by swapchain image (per-image to avoid reuse issues)
    std::vector<vk::raii::Semaphore> render_finished_semaphores_;
    // Present fences (via VK_KHR_swapchain_maintenance1) also per swapchain image
    std::vector<vk::raii::Fence> present_fences_;
    // CPU fences for command buffer reuse, indexed by frame-in-flight
    std::vector<vk::raii::Fence> in_flight_;
    std::vector<vk::raii::CommandBuffer> command_buffers_;

    vk::raii::Buffer sprite_vertex_buffer_{nullptr};
    vk::raii::DeviceMemory sprite_vertex_memory_{nullptr};
    std::uint32_t sprite_vertex_count_ = 0;

    vk::raii::Buffer ship_vertex_buffer_{nullptr};
    vk::raii::DeviceMemory ship_vertex_memory_{nullptr};
    std::uint32_t ship_vertex_count_ = 0;

    // CPU-side ship shape base (rebuilt each update)
    std::vector<RetroVertex4D> ship_base_shape_;
};

} // namespace kengine