#include "kengine/vulkan/pipeline_cache.hpp"
#include <vulkan/vulkan.h>
#include <cstring>
#include <fstream>
#include <iostream>

namespace kengine {

namespace {

constexpr std::uint32_t kCacheMagic = 0x0043504B; /* 'KPC\0' */

struct PipelineCacheHeader {
    std::uint32_t magic;
    std::uint32_t vendor_id;
    std::uint32_t device_id;
    std::uint32_t driver_version;
    std::uint32_t data_size;
};

} // namespace

PipelineCache::PipelineCache(const vk::raii::Device& device,
                             const vk::raii::PhysicalDevice& physical_device,
                             const std::filesystem::path& cache_path)
    : cache_path_(cache_path), device_(&device) {

    device_props_ = physical_device.getProperties();
    auto initial = load_cache_data(physical_device);

    vk::PipelineCacheCreateInfo info;
    if (!initial.empty()) {
        info.initialDataSize = initial.size();
        info.pInitialData    = initial.data();
    }

    cache_ = device.createPipelineCache(info);

    if (loaded_from_disk_) {
        std::cout << "[PipelineCache] Loaded from " << cache_path_ << '\n';
    } else {
        std::cout << "[PipelineCache] Created fresh cache\n";
    }
}

std::vector<std::uint8_t> PipelineCache::load_cache_data(
    const vk::raii::PhysicalDevice& physical_device) {

    if (cache_path_.empty() || !std::filesystem::exists(cache_path_)) return {};

    std::ifstream file(cache_path_, std::ios::binary);
    if (!file) return {};

    PipelineCacheHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file || header.magic != kCacheMagic || header.data_size == 0) return {};

    auto props = physical_device.getProperties();
    if (header.vendor_id != props.vendorID
        || header.device_id != props.deviceID
        || header.driver_version != props.driverVersion) {
        std::cout << "[PipelineCache] Stale cache (device/driver mismatch), rebuilding\n";
        return {};
    }

    std::vector<std::uint8_t> data(header.data_size);
    file.read(reinterpret_cast<char*>(data.data()), header.data_size);
    if (!file) return {};

    loaded_from_disk_ = true;
    return data;
}

void PipelineCache::save() {
    if (!cache_ || cache_path_.empty()) return;

    VkDevice vk_device = static_cast<VkDevice>(static_cast<vk::Device>(*device_));
    VkPipelineCache vk_cache = static_cast<VkPipelineCache>(static_cast<vk::PipelineCache>(*cache_));

    size_t data_size = 0;
    vkGetPipelineCacheData(vk_device, vk_cache, &data_size, nullptr);
    if (data_size == 0) return;

    std::vector<std::uint8_t> data(data_size);
    vkGetPipelineCacheData(vk_device, vk_cache, &data_size, data.data());

    std::error_code ec;
    std::filesystem::create_directories(cache_path_.parent_path(), ec);

    std::ofstream file(cache_path_, std::ios::binary | std::ios::trunc);
    if (!file) {
        std::cerr << "[PipelineCache] Failed to write " << cache_path_ << '\n';
        return;
    }

    PipelineCacheHeader header{};
    header.magic          = kCacheMagic;
    header.vendor_id      = device_props_.vendorID;
    header.device_id      = device_props_.deviceID;
    header.driver_version = device_props_.driverVersion;
    header.data_size      = static_cast<std::uint32_t>(data.size());

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));

    if (file) {
        std::cout << "[PipelineCache] Saved " << data.size() << " bytes to " << cache_path_ << '\n';
    }
}

} // namespace kengine