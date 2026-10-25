#pragma once
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace behemoth {
// One persistent host thread per GPU keeps HIP/hipBLASLt thread-local state warm.
class RankWorker {
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::packaged_task<void()>> queue_;
    bool stopping_ = false;
    std::thread thread_;

  public:
    RankWorker()
        : thread_([this] {
              for (;;) {
                  std::packaged_task<void()> task;
                  {
                      std::unique_lock lock(mutex_);
                      cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                      if (queue_.empty())
                          return;
                      task = std::move(queue_.front());
                      queue_.pop_front();
                  }
                  task(); // packaged_task delivers exceptions through the future.
              }
          }) {}
    ~RankWorker() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable())
            thread_.join();
    }
    RankWorker(const RankWorker &) = delete;
    template <class F> std::future<void> submit(F function) {
        std::packaged_task<void()> task(std::move(function));
        auto result = task.get_future();
        {
            std::lock_guard lock(mutex_);
            if (stopping_)
                throw std::runtime_error("GPU worker is stopping");
            queue_.push_back(std::move(task));
        }
        cv_.notify_one();
        return result;
    }
};
} // namespace behemoth
