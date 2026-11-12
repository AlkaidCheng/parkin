/**
 * @file thread_pool.hpp
 * @brief The pool of threads that kernels run on.
 */
#ifndef PARKIN_CPU_THREAD_POOL_HPP
#define PARKIN_CPU_THREAD_POOL_HPP

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace parkin::cpu {

/**
 * @brief Threads that wait for independent tasks and run them.
 *
 * The pool starts without threads and starts one whenever a call asks for
 * more than it has; a thread, once started, lives as long as the pool. The
 * calling thread always takes part in its own call, so a call completes even
 * when the system refuses to start a thread.
 *
 * @note Every member function may be called from several threads at once.
 */
class ThreadPool {
 public:
  ThreadPool() = default;
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  /** @brief Stop and join the threads. No call of parallel_for() may be running. */
  ~ThreadPool();

  /**
   * @brief Run task(0), ..., task(task_count - 1) and return when all are done.
   *
   * Each index is passed to @p task exactly once. Which thread runs an index,
   * and in which order, is unspecified, so the tasks must be independent.
   *
   * @param task_count Number of tasks.
   * @param thread_count Largest number of threads to run the tasks on, the
   *                     calling thread included. Zero counts as one.
   * @param task The work for one index. It may be called from several
   *             threads at once.
   * @throws The first exception a task throws, once every task that was
   *         started has returned. Whether the tasks not yet started still
   *         run is unspecified.
   */
  void parallel_for(std::size_t task_count, std::size_t thread_count,
                    const std::function<void(std::size_t)>& task);

  /** @brief Number of threads the pool has started. */
  [[nodiscard]] std::size_t worker_count() const;

 private:
  struct Job;

  /** @brief Start threads until there are @p worker_count. The caller holds mutex_. */
  void start_workers(std::size_t worker_count);
  void run_worker();
  static void run_tasks(Job& job);

  mutable std::mutex mutex_;
  std::condition_variable job_is_pending_;
  /// One entry for each worker that a running call still waits to be joined by.
  std::deque<std::shared_ptr<Job>> pending_jobs_;
  std::vector<std::thread> workers_;
  bool is_stopping_ = false;
};

/**
 * @brief The pool every kernel of this process runs on.
 *
 * The pool is created on first use and never destroyed: its threads may
 * still be waiting when the process exits, and joining them from a static
 * destructor can block on some systems. A child process created by fork()
 * gets a pool of its own, because threads do not survive a fork.
 */
[[nodiscard]] ThreadPool& shared_thread_pool();

}  // namespace parkin::cpu

#endif  // PARKIN_CPU_THREAD_POOL_HPP
