#pragma once

#include "kengine/vulkan/gpu_image.hpp"
#include "kengine/vulkan/vma_allocator.hpp"
#include <vulkan/vulkan_raii.hpp>
#include <cstdint>
#include <vector>

namespace kengine {

// Simple texture wrapper for sampled images (supports bindless).
// For now, created from raw RGBA8 data. Loading from files can use stb_image.
class Texture {
public:
    Texture() = default;
    // Create + upload from raw data (RGBA8 assumed for simplicity).
    // Requires queue + cmd pool for one-shot transfer (common for init time textures).
    Texture(const vk::raii::Device& device, VmaAllocator allocator,
            const vk::raii::Queue& graphicsQueue, const vk::raii::CommandPool& cmdPool,
            const void* data, uint32_t width, uint32_t height,
            vk::Format format = vk::Format::eR8G8B8A8Unorm);

    ~Texture() = default;

    Texture(Texture&&) noexcept = default;
    Texture& operator=(Texture&&) noexcept = default;

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    vk::Image     image() const { return image_.image(); }
    vk::ImageView view() const { return image_.view(); }
    vk::Sampler   sampler() const { return *sampler_; }
    uint32_t      width() const { return width_; }
    uint32_t      height() const { return height_; }
    vk::Format    format() const { return image_.format(); }

    bool valid() const { return image_.valid() && sampler_; }

private:
    void createSampler(const vk::raii::Device& device);

    GpuImage image_;
    std::optional<vk::raii::Sampler> sampler_;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
};

} // namespace kengine
