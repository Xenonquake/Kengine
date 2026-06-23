#include "kengine/render/texture.hpp"
#include <cstring>
#include <stdexcept>

namespace kengine {

Texture::Texture(const vk::raii::Device& device, VmaAllocator allocator,
                 const vk::raii::Queue& graphicsQueue, const vk::raii::CommandPool& cmdPool,
                 const void* data, uint32_t width, uint32_t height,
                 vk::Format format)
    : width_(width), height_(height) {

    if (!data || width == 0 || height == 0) {
        throw std::runtime_error("Invalid texture data for Texture");
    }

    vk::DeviceSize dataSize = static_cast<vk::DeviceSize>(width) * height * 4; // assume RGBA8

    GpuImageDesc desc;
    desc.extent = vk::Extent2D{width, height};
    desc.format = format;
    desc.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
    desc.aspect = vk::ImageAspectFlagBits::eColor;
    image_ = GpuImage(device, allocator, desc);

    createSampler(device);

    // --- Proper staging with VMA (VMA owns the VkBuffer) ---
    VkBufferCreateInfo bufCI{};
    bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size = dataSize;
    bufCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    allocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer stagingVk = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;
    VmaAllocationInfo stagingAllocInfo{};
    VkResult res = vmaCreateBuffer(allocator, &bufCI, &allocCI, &stagingVk, &stagingAlloc, &stagingAllocInfo);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("vmaCreateBuffer failed for texture staging");
    }

    bool explicitlyMapped = false;
    if (stagingAllocInfo.pMappedData == nullptr) {
        // Explicit map as fallback (when AUTO didn't provide mapping)
        void* mapped = nullptr;
        res = vmaMapMemory(allocator, stagingAlloc, &mapped);
        if (res != VK_SUCCESS || mapped == nullptr) {
            vmaDestroyBuffer(allocator, stagingVk, stagingAlloc);
            throw std::runtime_error("Failed to map staging buffer for texture upload");
        }
        stagingAllocInfo.pMappedData = mapped;
        explicitlyMapped = true;
    }

    std::memcpy(stagingAllocInfo.pMappedData, data, dataSize);
    vmaFlushAllocation(allocator, stagingAlloc, 0, dataSize);

    if (explicitlyMapped) {
        vmaUnmapMemory(allocator, stagingAlloc);
        stagingAllocInfo.pMappedData = nullptr;
    }

    // One-time command buffer for the copy
    vk::CommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.commandPool = *cmdPool;
    cmdAlloc.level = vk::CommandBufferLevel::ePrimary;
    cmdAlloc.commandBufferCount = 1;
    auto cmdBufs = device.allocateCommandBuffers(cmdAlloc);
    vk::raii::CommandBuffer cmd(std::move(cmdBufs[0]));  // raii will reset on end of scope, but we submit

    cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    vk::ImageMemoryBarrier2 toDst{};
    toDst.image = image_.image();
    toDst.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
    toDst.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
    toDst.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
    toDst.oldLayout = vk::ImageLayout::eUndefined;
    toDst.newLayout = vk::ImageLayout::eTransferDstOptimal;
    toDst.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

    vk::DependencyInfo dep{};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &toDst;
    cmd.pipelineBarrier2(dep);

    vk::BufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.imageExtent = vk::Extent3D{width, height, 1};

    // Use raw handle for the staging buffer (VMA owns it)
    vk::Buffer stagingHandle{stagingVk};
    cmd.copyBufferToImage(stagingHandle, image_.image(), vk::ImageLayout::eTransferDstOptimal, copyRegion);

    vk::ImageMemoryBarrier2 toRead{};
    toRead.image = image_.image();
    toRead.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
    toRead.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
    toRead.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
    toRead.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
    toRead.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    toRead.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    toRead.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

    dep.pImageMemoryBarriers = &toRead;
    cmd.pipelineBarrier2(dep);

    cmd.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &*cmd;
    graphicsQueue.submit(submitInfo);
    graphicsQueue.waitIdle();

    vmaDestroyBuffer(allocator, stagingVk, stagingAlloc);
    // No raii buffer for staging to avoid double-destroy with VMA
}

void Texture::createSampler(const vk::raii::Device& device) {
    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;

    sampler_ = device.createSampler(samplerInfo);
}

} // namespace kengine

