#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <filesystem>
#include <vector>

namespace kengine {

class PipelineCache {
public:
    PipelineCache(const vk::raii::Device& device,
                  const vk::raii::PhysicalDevice& physical_device,
                  const std::filesystem::path& cache_path);

    const vk::raii::PipelineCache& handle() const { return *cache_; }
    bool loaded_from_disk() const { return loaded_from_disk_; }

    void save();

private:
    std::vector<std::uint8_t> load_cache_data(
        const vk::raii::PhysicalDevice& physical_device);

    std::filesystem::path cache_path_;
    bool loaded_from_disk_ = false;
    vk::PhysicalDeviceProperties device_props_{};
    const vk::raii::Device* device_ = nullptr;
    std::optional<vk::raii::PipelineCache> cache_;
};

} // namespace kengine