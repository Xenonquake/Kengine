#include "kengine/vulkan/shader_module.hpp"
#include <fstream>
#include <stdexcept>

namespace kengine {

std::vector<std::uint32_t> load_spirv_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open shader: " + path.string());

    auto size = file.tellg();
    if (size % 4 != 0) throw std::runtime_error("Invalid SPIR-V size: " + path.string());

    file.seekg(0);
    std::vector<std::uint32_t> spirv(static_cast<std::size_t>(size) / 4);
    file.read(reinterpret_cast<char*>(spirv.data()), size);
    return spirv;
}

ShaderModule::ShaderModule(const vk::raii::Device& device, const std::vector<std::uint32_t>& spirv) {
    vk::ShaderModuleCreateInfo info;
    info.codeSize = spirv.size() * sizeof(std::uint32_t);
    info.pCode    = spirv.data();
    module_ = device.createShaderModule(info);
}

ShaderModule::ShaderModule(const vk::raii::Device& device, const std::filesystem::path& spv_path)
    : ShaderModule(device, load_spirv_file(spv_path)) {}

} // namespace kengine