#include "parkin/cpu/thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <system_error>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <pthread.h>
#define PARKIN_HAS_FORK 1
#else
#define PARKIN_HAS_FORK 0
#endif

namespace parkin::cpu {

/** @brief One call of parallel_for(), shared by the threads that work on it. */
struct ThreadPool::Job {
  Job(const std::function<void(std::size_t)>& task_function, std::size_t count)
      : task(task_function), task_count(count) {}

  /// Owned by the caller of parallel_for(), which outlives every use: a
  /// thread calls it only for an index below task_count, and the caller
  /// waits until all of those have returned.
  const std::function<void(std::size_t)>& task;
  const std::size_t task_count;
  std::atomic<std::size_t> next_index{0};

  std::mutex mutex;
  std::condition_variable all_tasks_finished;
  std::size_t finished_count = 0;  ///< Guarded by mutex.
  std::exception_ptr first_error;  ///< Guarded by mutex.
};

ThreadPool::~ThreadPool() {
  {
    const std::lock_guard lock(mutex_);
    is_stopping_ = true;
  }
  job_is_pending_.notify_all();
  for (std::thread& worker : workers_) {
    worker.join();
  }
}

void ThreadPool::parallel_for(std::size_t task_count, std::size_t thread_count,
                              const std::function<void(std::size_t)>& task) {
  if (task_count == 0) {
    return;
  }
  if (thread_count <= 1 || task_count == 1) {
    for (std::size_t index = 0; index < task_count; ++index) {
      task(index);
    }
    return;
  }
  // No more threads than tasks, and the calling thread is one of them.
  const std::size_t wanted_worker_count = std::min(thread_count, task_count) - 1;

  const auto job = std::make_shared<Job>(task, task_count);
  {
    const std::lock_guard lock(mutex_);
    start_workers(wanted_worker_count);
    const std::size_t helper_count = std::min(wanted_worker_count, workers_.size());
    for (std::size_t helper = 0; helper < helper_count; ++helper) {
      pending_jobs_.push_back(job);
    }
  }
  job_is_pending_.notify_all();

  run_tasks(*job);
  {
    std::unique_lock lock(job->mutex);
    job->all_tasks_finished.wait(lock, [&job] { return job->finished_count == job->task_count; });
  }
  {
    // Entries no worker took in time would otherwise wait in the queue.
    const std::lock_guard lock(mutex_);
    std::erase(pending_jobs_, job);
  }
  if (job->first_error) {
    std::rethrow_exception(job->first_error);
  }
}

std::size_t ThreadPool::worker_count() const {
  const std::lock_guard lock(mutex_);
  return workers_.size();
}

void ThreadPool::start_workers(std::size_t worker_count) {
  while (workers_.size() < worker_count) {
    try {
      workers_.emplace_back([this] { run_worker(); });
    } catch (const std::system_error&) {
      // The system refuses another thread. The tasks then run on the threads
      // there are, and at least on the calling one.
      return;
    }
  }
}

void ThreadPool::run_worker() {
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock lock(mutex_);
      job_is_pending_.wait(lock, [this] { return is_stopping_ || !pending_jobs_.empty(); });
      if (is_stopping_) {
        return;
      }
      job = std::move(pending_jobs_.front());
      pending_jobs_.pop_front();
    }
    run_tasks(*job);
  }
}

void ThreadPool::run_tasks(Job& job) {
  for (;;) {
    const std::size_t index = job.next_index.fetch_add(1, std::memory_order_relaxed);
    if (index >= job.task_count) {
      return;
    }
    std::exception_ptr error;
    try {
      job.task(index);
    } catch (...) {
      // An exception must not leave a worker thread; the caller rethrows it.
      error = std::current_exception();
    }
    const std::lock_guard lock(job.mutex);
    if (error && !job.first_error) {
      job.first_error = error;
    }
    ++job.finished_count;
    if (job.finished_count == job.task_count) {
      job.all_tasks_finished.notify_all();
    }
  }
}

namespace {

std::atomic<ThreadPool*> shared_pool{nullptr};

#if PARKIN_HAS_FORK
/**
 * @brief Make a forked child create its own pool.
 *
 * The child has the parent's pool object but none of its threads, and a
 * mutex of that pool may have been locked by a thread that no longer exists.
 * The object is left alone and never used again.
 */
void forget_shared_pool() noexcept { shared_pool.store(nullptr, std::memory_order_release); }

[[nodiscard]] bool install_fork_handler() noexcept {
  return pthread_atfork(nullptr, nullptr, &forget_shared_pool) == 0;
}
#endif

}  // namespace

ThreadPool& shared_thread_pool() {
#if PARKIN_HAS_FORK
  [[maybe_unused]] static const bool is_fork_handler_installed = install_fork_handler();
#endif
  ThreadPool* pool = shared_pool.load(std::memory_order_acquire);
  if (pool != nullptr) {
    return *pool;
  }
  auto new_pool = std::make_unique<ThreadPool>();
  if (shared_pool.compare_exchange_strong(pool, new_pool.get(), std::memory_order_acq_rel)) {
    // Released on purpose: see the declaration.
    return *new_pool.release();
  }
  // Another thread installed its pool first. new_pool has started no thread
  // and is destroyed here.
  return *pool;
}

}  // namespace parkin::cpu
