#include "kengine/render/scene_targets.hpp"

namespace kengine {

SceneTargetSet::SceneTargetSet(const vk::raii::Device& device, VmaAllocator allocator,
                               std::uint32_t frames_in_flight)
    : device_(device), allocator_(allocator), frames_in_flight_(frames_in_flight) {
    frames_.resize(frames_in_flight);
}

void SceneTargetSet::recreate(vk::Extent2D extent) {
    extent_ = extent;

    GpuImageDesc color_desc;
    color_desc.extent = extent;
    color_desc.format = kSceneColorFormat;
    color_desc.usage  = vk::ImageUsageFlagBits::eColorAttachment
                      | vk::ImageUsageFlagBits::eSampled
                      | vk::ImageUsageFlagBits::eTransferSrc;
    color_desc.aspect = vk::ImageAspectFlagBits::eColor;

    GpuImageDesc depth_desc;
    depth_desc.extent = extent;
    depth_desc.format = kSceneDepthFormat;
    depth_desc.usage  = vk::ImageUsageFlagBits::eDepthStencilAttachment
                      | vk::ImageUsageFlagBits::eSampled;
    depth_desc.aspect = vk::ImageAspectFlagBits::eDepth;

    for (auto& frame : frames_) {
        if (frame.color.valid()) {
            frame.color.recreate(color_desc);
            frame.depth.recreate(depth_desc);
        } else {
            frame.color = GpuImage(device_, allocator_, color_desc);
            frame.depth = GpuImage(device_, allocator_, depth_desc);
        }
    }
}

} // namespace kengine