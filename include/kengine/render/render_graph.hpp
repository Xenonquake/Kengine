#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace kengine {

enum class ResourceType { Buffer, Image, Swapchain };

enum class ImageLayout {
    Undefined,
    ColorAttachment,
    ShaderRead,
    TransferDst,
    Present
};

struct ResourceHandle {
    std::uint32_t id = 0;
};

struct RenderPassNode {
    std::string name;
    std::vector<ResourceHandle> reads;
    std::vector<ResourceHandle> writes;
    std::function<void()> execute;
};

struct ResourceDesc {
    std::string name;
    ResourceType type = ResourceType::Image;
    ImageLayout final_layout = ImageLayout::ShaderRead;
};

class RenderGraph {
public:
    ResourceHandle create_resource(const ResourceDesc& desc);
    void add_pass(RenderPassNode pass);
    void compile();
    void execute();

    void set_pass_execute(const std::string& name, std::function<void()> fn);

    const std::vector<RenderPassNode>& passes() const { return passes_; }

private:
    struct Barrier {
        ResourceHandle resource;
        ImageLayout from;
        ImageLayout to;
    };

    void topological_sort();
    void insert_barriers();

    std::unordered_map<std::uint32_t, ResourceDesc> resources_;
    std::vector<RenderPassNode> passes_;
    std::vector<RenderPassNode> sorted_passes_;
    std::vector<Barrier> barriers_;
    std::uint32_t next_id_ = 1;
};

} // namespace kengine