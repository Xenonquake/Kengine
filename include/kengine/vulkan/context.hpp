#pragma once

#include "kengine/vulkan/vma_allocator.hpp"
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vector>

namespace kengine {

struct VulkanConfig {
    bool enable_validation = true;
    bool vsync             = true;
    std::uint32_t frames_in_flight = 2;
};

class VulkanContext {
public:
    explicit VulkanContext(const VulkanConfig& config = {});
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    void init_window(int width, int height, const char* title);
    void init_vulkan();
    void shutdown();

    GLFWwindow* window() const { return window_; }
    const vk::raii::Instance& instance() const { return *instance_; }
    const vk::raii::PhysicalDevice& physical_device() const { return *physical_device_; }
    const vk::raii::Device& device() const { return *device_; }
    const vk::raii::Queue& graphics_queue() const { return *graphics_queue_; }
    const vk::raii::Queue& present_queue() const { return *present_queue_; }
    vk::raii::Queue& graphics_queue() { return *graphics_queue_; }

    std::uint32_t graphics_queue_family() const { return graphics_family_; }
    std::uint32_t present_queue_family() const { return present_family_; }
    std::uint32_t frames_in_flight() const { return config_.frames_in_flight; }

    GpuAllocator& allocator() { return allocator_; }
    const vk::raii::CommandPool& command_pool() const { return *command_pool_; }

    vk::Extent2D framebuffer_size() const;

private:
    bool check_validation_support();
    void create_instance();
    void pick_physical_device();
    void create_device();
    void create_command_pool();
    bool validation_layers_enabled_ = false;

    VulkanConfig config_;
    GLFWwindow* window_ = nullptr;

    vk::raii::Context vulkan_context_;
    std::optional<vk::raii::Instance> instance_;
    std::optional<vk::raii::DebugUtilsMessengerEXT> debug_messenger_;
    std::optional<vk::raii::PhysicalDevice> physical_device_;
    std::optional<vk::raii::Device> device_;
    std::optional<vk::raii::Queue> graphics_queue_;
    std::optional<vk::raii::Queue> present_queue_;
    std::optional<vk::raii::CommandPool> command_pool_;

    std::uint32_t graphics_family_ = 0;
    std::uint32_t present_family_ = 0;

    GpuAllocator allocator_;
};

} // namespace kengine