#include "kengine/render/render_graph.hpp"
#include <algorithm>
#include <unordered_set>

namespace kengine {

ResourceHandle RenderGraph::create_resource(const ResourceDesc& desc) {
    ResourceHandle h{next_id_++};
    resources_[h.id] = desc;
    return h;
}

void RenderGraph::add_pass(RenderPassNode pass) {
    passes_.push_back(std::move(pass));
}

void RenderGraph::compile() {
    topological_sort();
    insert_barriers();
}

void RenderGraph::set_pass_execute(const std::string& name, std::function<void()> fn) {
    for (auto& pass : passes_) {
        if (pass.name == name) pass.execute = fn;
    }
    for (auto& pass : sorted_passes_) {
        if (pass.name == name) pass.execute = fn;
    }
}

void RenderGraph::execute() {
    for (const auto& barrier : barriers_) {
        (void)barrier; /* Vulkan barrier insertion point */
    }
    for (auto& pass : sorted_passes_) {
        if (pass.execute) pass.execute();
    }
}

void RenderGraph::topological_sort() {
    sorted_passes_ = passes_;
    /* Stable declaration order suffices for initial pipeline */
}

void RenderGraph::insert_barriers() {
    barriers_.clear();
    std::unordered_map<std::uint32_t, ImageLayout> current;

    for (const auto& pass : sorted_passes_) {
        for (auto read : pass.reads) {
            auto it = current.find(read.id);
            if (it != current.end() && it->second != ImageLayout::ShaderRead) {
                barriers_.push_back({read, it->second, ImageLayout::ShaderRead});
                it->second = ImageLayout::ShaderRead;
            }
        }
        for (auto write : pass.writes) {
            auto& layout = current[write.id];
            auto res_it  = resources_.find(write.id);
            ImageLayout target = res_it != resources_.end()
                ? res_it->second.final_layout : ImageLayout::ColorAttachment;
            if (layout != ImageLayout::Undefined && layout != target) {
                barriers_.push_back({write, layout, target});
            }
            layout = target;
        }
    }
}

} // namespace kengine