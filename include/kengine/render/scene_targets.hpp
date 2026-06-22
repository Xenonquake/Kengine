#pragma once

#include "kengine/vulkan/gpu_image.hpp"
#include "kengine/vulkan/vma_allocator.hpp"
#include <vulkan/vulkan_raii.hpp>
#include <vector>

namespace kengine {

struct SceneFrameTargets {
    GpuImage color;
    GpuImage depth;
};

class SceneTargetSet {
public:
    SceneTargetSet(const vk::raii::Device& device, VmaAllocator allocator,
                   std::uint32_t frames_in_flight);

    void recreate(vk::Extent2D extent);
    const SceneFrameTargets& frame(std::uint32_t index) const { return frames_[index]; }
    vk::Extent2D extent() const { return extent_; }
    std::uint32_t count() const { return static_cast<std::uint32_t>(frames_.size()); }

    static constexpr vk::Format kSceneColorFormat = vk::Format::eR8G8B8A8Unorm;
    static constexpr vk::Format kSceneDepthFormat = vk::Format::eD32Sfloat;

private:
    const vk::raii::Device& device_;
    VmaAllocator allocator_;
    std::uint32_t frames_in_flight_;
    vk::Extent2D extent_{};
    std::vector<SceneFrameTargets> frames_;
};

} // namespace kengine