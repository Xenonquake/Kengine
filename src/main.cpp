#include "kengine/app/engine.hpp"
#include <iostream>

int main() {
    kengine::EngineConfig config;
    config.title = "Kengine — Vulkan 4D Engine";
    config.window_width  = 1476;
    config.window_height = 830;
    config.vsync = true;
    config.frame_graph.enable_bloom   = true;
    config.frame_graph.enable_dof     = true;
    config.frame_graph.enable_aa      = true;
    config.frame_graph.enable_sharpen = true;

    kengine::Engine engine(config);
    engine.set_retro_style(kengine::RetroStyle::GeometryCore);
    if (!engine.init()) {
        std::cerr << "Failed to initialize Kengine\n";
        return 1;
    }

    std::cout << "Kengine running — press close or ESC to exit\n";
    engine.run();
    return 0;
}
