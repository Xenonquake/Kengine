#include <vulkan/vulkan.h>
#include "kengine/vulkan/context.hpp"
#include <iostream>
#include <set>
#include <stdexcept>

namespace kengine {

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* /*user_data*/) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[Vulkan] " << callback_data->pMessage << '\n';
    }
    (void)type;
    return VK_FALSE;
}

static vk::Bool32 vk_hpp_debug_callback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT type,
    const vk::DebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data) {
    return debug_callback(
        static_cast<VkDebugUtilsMessageSeverityFlagBitsEXT>(severity),
        static_cast<VkDebugUtilsMessageTypeFlagsEXT>(type),
        reinterpret_cast<const VkDebugUtilsMessengerCallbackDataEXT*>(callback_data),
        user_data);
}

VulkanContext::VulkanContext(const VulkanConfig& config) : config_(config) {}

VulkanContext::~VulkanContext() {
    shutdown();
}

void VulkanContext::init_window(int width, int height, const char* title) {
    if (!glfwInit()) throw std::runtime_error("Failed to init GLFW");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window_) throw std::runtime_error("Failed to create window");
}

void VulkanContext::init_vulkan() {
    validation_layers_enabled_ = config_.enable_validation && check_validation_support();
    create_instance();
    pick_physical_device();
    create_device();
    create_command_pool();
    allocator_.init(*instance_, *physical_device_, *device_);
}

void VulkanContext::shutdown() {
    // Ensure idle before tearing down
    if (device_) {
        device_->waitIdle();
    }

    // Reset things that depend on device first
    command_pool_.reset();
    // allocator will be destroyed in its dtor (after this, before device in full context dtor)

    device_.reset();
    debug_messenger_.reset();
    physical_device_.reset();
    instance_.reset();
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
}

vk::Extent2D VulkanContext::framebuffer_size() const {
    int w, h;
    glfwGetFramebufferSize(window_, &w, &h);
    return vk::Extent2D{static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h)};
}

bool VulkanContext::check_validation_support() {
    auto layers = vk::enumerateInstanceLayerProperties();
    const char* desired = "VK_LAYER_KHRONOS_validation";
    for (const auto& layer : layers) {
        if (strcmp(layer.layerName, desired) == 0) return true;
    }
    return false;
}

void VulkanContext::create_instance() {
    vk::ApplicationInfo app_info;
    app_info.pApplicationName   = "Kengine";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName        = "Kengine";
    app_info.engineVersion      = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion         = VK_API_VERSION_1_3;

    std::uint32_t glfw_ext_count = 0;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
    if (!glfw_extensions) throw std::runtime_error("GLFW Vulkan extensions unavailable");

    std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_ext_count);
    if (validation_layers_enabled_) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    vk::InstanceCreateInfo create_info;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount   = static_cast<std::uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();

    const char* validation_layer = "VK_LAYER_KHRONOS_validation";
    if (validation_layers_enabled_) {
        create_info.enabledLayerCount   = 1;
        create_info.ppEnabledLayerNames = &validation_layer;
    }

    instance_ = vk::raii::Instance{vulkan_context_, create_info};

    if (validation_layers_enabled_) {
        vk::DebugUtilsMessengerCreateInfoEXT debug_info;
        debug_info.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
                                   | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
        debug_info.messageType     = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
                                   | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;
        debug_info.pfnUserCallback = vk_hpp_debug_callback;
        debug_messenger_ = instance_->createDebugUtilsMessengerEXT(debug_info);
    }
}

void VulkanContext::pick_physical_device() {
    auto devices = instance_->enumeratePhysicalDevices();
    if (devices.empty()) throw std::runtime_error("No Vulkan physical devices");

    for (const auto& dev : devices) {
        auto queue_families = dev.getQueueFamilyProperties();
        bool graphics_found = false;
        bool present_found  = false;

        for (std::uint32_t i = 0; i < queue_families.size(); ++i) {
            if (queue_families[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                graphics_family_ = i;
                graphics_found   = true;
            }
        }

        VkSurfaceKHR raw_surface = VK_NULL_HANDLE;
        VkInstance vk_instance = static_cast<VkInstance>(static_cast<vk::Instance>(*instance_));
        if (glfwCreateWindowSurface(vk_instance, window_, nullptr, &raw_surface) != VK_SUCCESS) {
            continue;
        }
        vk::raii::SurfaceKHR test_surface{*instance_, raw_surface};

        for (std::uint32_t i = 0; i < queue_families.size(); ++i) {
            if (dev.getSurfaceSupportKHR(i, *test_surface)) {
                present_family_ = i;
                present_found   = true;
                break;
            }
        }

        if (graphics_found && present_found) {
            physical_device_ = dev;
            return;
        }
    }
    throw std::runtime_error("No suitable GPU found");
}

void VulkanContext::create_device() {
    std::set<std::uint32_t> families = {graphics_family_, present_family_};
    std::vector<vk::DeviceQueueCreateInfo> queue_infos;
    float priority = 1.0f;
    for (auto family : families) {
        vk::DeviceQueueCreateInfo qci;
        qci.queueFamilyIndex = family;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &priority;
        queue_infos.push_back(qci);
    }

    vk::PhysicalDeviceFeatures2 features2;
    vk::PhysicalDeviceVulkan13Features features13;
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;
    features2.pNext             = &features13;

    // For GPU-driven indirect draw + count (culling)
    vk::PhysicalDeviceVulkan12Features features12{};
    features12.drawIndirectCount = VK_TRUE;
    features12.pNext = features2.pNext;
    features2.pNext = &features12;

    // Descriptor indexing (bindless) features - chained if extension is available
    VkPhysicalDeviceDescriptorIndexingFeatures descriptorIndexingFeatures{};
    descriptorIndexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    descriptorIndexingFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    descriptorIndexingFeatures.runtimeDescriptorArray = VK_TRUE;
    descriptorIndexingFeatures.descriptorBindingPartiallyBound = VK_TRUE;
    descriptorIndexingFeatures.descriptorBindingVariableDescriptorCount = VK_TRUE;
    descriptorIndexingFeatures.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;

    // Hook into the chain (note: we attach even if ext not enabled; driver will ignore unsupported)
    descriptorIndexingFeatures.pNext = features2.pNext;
    features2.pNext = &descriptorIndexingFeatures;

    // Note: VK_KHR_swapchain_maintenance1 + surface_maintenance1 are requested below.
    // Full feature structs are omitted here for header compatibility; when available
    // they can be chained to enable present fences etc.

    // Base required extensions
    std::vector<const char*> device_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME
    };

    // Query supported extensions so we can optionally enable newer ones like swapchain_maintenance1
    auto available_exts = physical_device_->enumerateDeviceExtensionProperties();
    auto has_ext = [&](const char* name) {
        return std::any_of(available_exts.begin(), available_exts.end(),
                           [&](const vk::ExtensionProperties& p) {
                               return std::strcmp(p.extensionName, name) == 0;
                           });
    };

    if (has_ext(VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME) &&
        has_ext(VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME)) {
        device_extensions.push_back(VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
        device_extensions.push_back(VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
        // TODO: when headers + driver fully support, also enable the feature bits and
        // use VkSwapchainPresentFenceInfoKHR on present for fence-based present sync.
    }

    if (has_ext(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)) {
        device_extensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    }

    vk::DeviceCreateInfo dci;
    dci.queueCreateInfoCount = static_cast<std::uint32_t>(queue_infos.size());
    dci.pQueueCreateInfos    = queue_infos.data();
    dci.pNext                = &features2;
    dci.enabledExtensionCount   = static_cast<std::uint32_t>(device_extensions.size());
    dci.ppEnabledExtensionNames = device_extensions.data();

    device_         = physical_device_->createDevice(dci);
    graphics_queue_ = vk::raii::Queue{*device_, graphics_family_, 0};
    present_queue_  = vk::raii::Queue{*device_, present_family_, 0};
}

void VulkanContext::create_command_pool() {
    vk::CommandPoolCreateInfo info;
    info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    info.queueFamilyIndex = graphics_family_;
    command_pool_ = device_->createCommandPool(info);
}

} // namespace kengine