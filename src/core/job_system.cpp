#include "kengine/core/job_system.hpp"
#include <thread>

namespace kengine {

JobSystem::JobSystem(std::size_t worker_count) {
    if (worker_count == 0) {
        worker_count = std::max(1u, std::thread::hardware_concurrency() - 1);
    }
    workers_.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

JobSystem::~JobSystem() {
    shutdown_ = true;
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

void JobSystem::enqueue(std::function<void()> job) {
    {
        std::lock_guard lock(mutex_);
        queue_.push(std::move(job));
    }
    cv_.notify_one();
}

void JobSystem::wait_idle() {
    for (;;) {
        std::lock_guard lock(mutex_);
        if (queue_.empty() && active_jobs_ == 0) break;
    }
}

void JobSystem::worker_loop() {
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [this] { return shutdown_ || !queue_.empty(); });
            if (shutdown_ && queue_.empty()) return;
            job = std::move(queue_.front());
            queue_.pop();
            ++active_jobs_;
        }
        job();
        --active_jobs_;
    }
}

} // namespace kengine