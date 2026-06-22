#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <filesystem>
#include <vector>

namespace kengine {

class ShaderModule {
public:
    ShaderModule() = default;
    ShaderModule(const vk::raii::Device& device, const std::vector<std::uint32_t>& spirv);
    ShaderModule(const vk::raii::Device& device, const std::filesystem::path& spv_path);

    const vk::raii::ShaderModule& handle() const { return *module_; }
    bool valid() const { return module_.has_value(); }

private:
    std::optional<vk::raii::ShaderModule> module_;
};

std::vector<std::uint32_t> load_spirv_file(const std::filesystem::path& path);

} // namespace kengine