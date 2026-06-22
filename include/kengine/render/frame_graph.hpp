#pragma once

#include "kengine/render/render_graph.hpp"
#include <cstdint>
#include <functional>
#include <string>

namespace kengine {

struct FrameGraphConfig {
    bool enable_bloom     = true;
    bool enable_dof       = true;
    bool enable_aa        = true;
    bool enable_sharpen   = true;
    float bloom_threshold = 1.0f;
    float bloom_intensity = 0.8f;
    float dof_focus_distance = 10.0f;
    float dof_aperture  = 2.8f;
    float sharpen_amount = 0.5f;
};

class FrameGraph {
public:
    explicit FrameGraph(const FrameGraphConfig& config = {});

    void build(std::uint32_t width, std::uint32_t height);
    void execute();

    void set_pass_execute(const std::string& name, std::function<void()> fn);

    RenderGraph& graph() { return graph_; }

private:
    FrameGraphConfig config_;
    RenderGraph graph_;

    ResourceHandle color_;
    ResourceHandle depth_;
    ResourceHandle bloom_a_;
    ResourceHandle bloom_b_;
    ResourceHandle resolved_;
};

} // namespace kengine