#include "kengine/render/frame_graph.hpp"

namespace kengine {

FrameGraph::FrameGraph(const FrameGraphConfig& config) : config_(config) {}

void FrameGraph::build(std::uint32_t width, std::uint32_t height) {
    graph_ = RenderGraph{};

    color_    = graph_.create_resource({"scene_color", ResourceType::Image, ImageLayout::ColorAttachment});
    depth_    = graph_.create_resource({"scene_depth", ResourceType::Image, ImageLayout::ColorAttachment});
    bloom_a_  = graph_.create_resource({"bloom_a", ResourceType::Image, ImageLayout::ShaderRead});
    bloom_b_  = graph_.create_resource({"bloom_b", ResourceType::Image, ImageLayout::ShaderRead});
    resolved_ = graph_.create_resource({"resolved", ResourceType::Image, ImageLayout::Present});

    graph_.add_pass({
        "geometry",
        {},
        {color_, depth_},
        [] {}
    });

    if (config_.enable_bloom) {
        graph_.add_pass({
            "bloom_threshold",
            {color_},
            {bloom_a_},
            [] {}
        });
        graph_.add_pass({
            "bloom_blur",
            {bloom_a_},
            {bloom_b_},
            [] {}
        });
    }

    if (config_.enable_dof) {
        graph_.add_pass({
            "depth_of_field",
            {color_, depth_},
            {bloom_a_},
            [] {}
        });
    }

    if (config_.enable_aa) {
        graph_.add_pass({
            "taa_resolve",
            {color_},
            {resolved_},
            [] {}
        });
    } else {
        graph_.add_pass({
            "copy_to_resolved",
            {color_},
            {resolved_},
            [] {}
        });
    }

    if (config_.enable_sharpen) {
        graph_.add_pass({
            "sharpen",
            {resolved_},
            {color_},
            [] {}
        });
    }

    graph_.add_pass({
        "present",
        {config_.enable_sharpen ? color_ : resolved_},
        {},
        [] {}
    });

    (void)width;
    (void)height;
    graph_.compile();
}

void FrameGraph::set_pass_execute(const std::string& name, std::function<void()> fn) {
    graph_.set_pass_execute(name, std::move(fn));
}

void FrameGraph::execute() {
    graph_.execute();
}

} // namespace kengine