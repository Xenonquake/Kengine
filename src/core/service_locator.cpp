#include "kengine/core/service_locator.hpp"

namespace kengine {

ServiceLocator& ServiceLocator::instance() {
    static ServiceLocator locator;
    return locator;
}

} // namespace kengine