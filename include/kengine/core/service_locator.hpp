#pragma once

#include <memory>
#include <mutex>
#include <typeindex>
#include <unordered_map>

namespace kengine {

class ServiceLocator {
public:
    template<typename T, typename... Args>
    static void register_service(Args&&... args) {
        register_instance<T>(std::make_shared<T>(std::forward<Args>(args)...));
    }

    template<typename T>
    static void register_instance(std::shared_ptr<T> service) {
        auto& map = instance().services_;
        std::lock_guard lock(instance().mutex_);
        map[std::type_index(typeid(T))] = std::move(service);
    }

    template<typename T>
    static std::shared_ptr<T> get() {
        auto& map = instance().services_;
        std::lock_guard lock(instance().mutex_);
        auto it = map.find(std::type_index(typeid(T)));
        if (it == map.end()) return nullptr;
        return std::static_pointer_cast<T>(it->second);
    }

    template<typename T>
    static void unregister() {
        auto& map = instance().services_;
        std::lock_guard lock(instance().mutex_);
        map.erase(std::type_index(typeid(T)));
    }

private:
    static ServiceLocator& instance();

    std::unordered_map<std::type_index, std::shared_ptr<void>> services_;
    std::mutex mutex_;
};

} // namespace kengine