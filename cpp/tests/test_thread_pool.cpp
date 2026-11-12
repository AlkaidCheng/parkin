/**
 * @file test_thread_pool.cpp
 * @brief Tests of the pool of threads that kernels run on.
 *
 * Catch2's assertions are made on the thread that runs the test case. A task
 * therefore records what it sees, in an atomic or in a slot of its own, and
 * the test case checks the record once parallel_for() has returned.
 */
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "parkin/cpu/thread_pool.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using parkin::cpu::ThreadPool;

/**
 * @brief How long tasks wait for each other in a Meeting.
 *
 * Threads meet within microseconds. The limit only turns a pool that never
 * brings them together into a failed check instead of a test that hangs.
 */
constexpr std::chrono::seconds kMeetingTimeout{30};

void do_nothing(std::size_t /*index*/) {}

/** @brief How often parallel_for() passed each index to the task. */
[[nodiscard]] std::vector<int> count_calls(ThreadPool& pool, std::size_t task_count,
                                           std::size_t thread_count) {
  std::vector<std::atomic<int>> call_counts(task_count);
  pool.parallel_for(task_count, thread_count, [&call_counts](std::size_t index) {
    call_counts[index].fetch_add(1, std::memory_order_relaxed);
  });
  std::vector<int> result;
  result.reserve(task_count);
  for (const std::atomic<int>& call_count : call_counts) {
    result.push_back(call_count.load());
  }
  return result;
}

/**
 * @brief A point where tasks wait until a given number of them are there.
 *
 * Tasks that all get past it were inside parallel_for() at the same time, so
 * each ran on a thread of its own.
 */
class Meeting {
 public:
  explicit Meeting(std::size_t expected_count) : expected_count_(expected_count) {}

  /** @brief Arrive, and wait for the others. @return Whether all arrived in time. */
  [[nodiscard]] bool attend() {
    std::unique_lock lock(mutex_);
    thread_ids_.insert(std::this_thread::get_id());
    ++arrived_count_;
    if (arrived_count_ >= expected_count_) {
      everyone_arrived_.notify_all();
      return true;
    }
    return everyone_arrived_.wait_for(lock, kMeetingTimeout,
                                      [this] { return arrived_count_ >= expected_count_; });
  }

  /** @brief The threads that have arrived. */
  [[nodiscard]] std::set<std::thread::id> thread_ids() const {
    const std::lock_guard lock(mutex_);
    return thread_ids_;
  }

 private:
  const std::size_t expected_count_;
  mutable std::mutex mutex_;
  std::condition_variable everyone_arrived_;
  std::size_t arrived_count_ = 0;
  std::set<std::thread::id> thread_ids_;
};

// --- Which tasks run --------------------------------------------------------

TEST_CASE("Every index is passed to the task exactly once", "[thread_pool]") {
  ThreadPool pool;
  const std::array<std::size_t, 6> task_counts{0, 1, 2, 7, 64, 1000};
  const std::array<std::size_t, 6> thread_counts{0, 1, 2, 3, 8, 33};

  for (const std::size_t task_count : task_counts) {
    for (const std::size_t thread_count : thread_counts) {
      CAPTURE(task_count, thread_count);
      CHECK(count_calls(pool, task_count, thread_count) == std::vector<int>(task_count, 1));
    }
  }
}

TEST_CASE("Many short calls in a row lose no task", "[thread_pool]") {
  ThreadPool pool;
  constexpr std::size_t kCallCount = 2000;
  constexpr std::size_t kTaskCount = 16;
  std::atomic<std::size_t> call_count{0};

  for (std::size_t call = 0; call < kCallCount; ++call) {
    pool.parallel_for(kTaskCount, 8, [&call_count](std::size_t /*index*/) {
      call_count.fetch_add(1, std::memory_order_relaxed);
    });
  }

  CHECK(call_count.load() == kCallCount * kTaskCount);
}

// --- Which threads run them -------------------------------------------------

TEST_CASE("One thread or one task runs on the calling thread alone", "[thread_pool]") {
  ThreadPool pool;
  const std::thread::id caller = std::this_thread::get_id();
  const auto threads_of = [&pool](std::size_t task_count, std::size_t thread_count) {
    std::vector<std::thread::id> thread_ids(task_count);
    pool.parallel_for(task_count, thread_count, [&thread_ids](std::size_t index) {
      thread_ids[index] = std::this_thread::get_id();
    });
    return thread_ids;
  };

  CHECK(threads_of(5, 1) == std::vector<std::thread::id>(5, caller));
  CHECK(threads_of(5, 0) == std::vector<std::thread::id>(5, caller));
  CHECK(threads_of(1, 8) == std::vector<std::thread::id>(1, caller));
  CHECK(pool.worker_count() == 0);
}

TEST_CASE("A call runs on as many threads as it asks for, the calling one among them",
          "[thread_pool]") {
  ThreadPool pool;
  constexpr std::size_t kThreadCount = 4;
  Meeting meeting(kThreadCount);
  std::atomic<std::size_t> late_count{0};

  pool.parallel_for(kThreadCount, kThreadCount, [&meeting, &late_count](std::size_t /*index*/) {
    if (!meeting.attend()) {
      late_count.fetch_add(1);
    }
  });

  CHECK(late_count.load() == 0);
  const std::set<std::thread::id> thread_ids = meeting.thread_ids();
  CHECK(thread_ids.size() == kThreadCount);
  CHECK(thread_ids.contains(std::this_thread::get_id()));
  CHECK(pool.worker_count() == kThreadCount - 1);
}

TEST_CASE("A call runs on no more threads than it asks for", "[thread_pool]") {
  ThreadPool pool;
  std::mutex mutex;
  std::set<std::thread::id> thread_ids;

  pool.parallel_for(64, 3, [&mutex, &thread_ids](std::size_t /*index*/) {
    const std::lock_guard lock(mutex);
    thread_ids.insert(std::this_thread::get_id());
  });

  CHECK(thread_ids.size() <= 3);
  CHECK(pool.worker_count() == 2);
}

TEST_CASE("The pool starts a thread only when a call needs one more", "[thread_pool]") {
  ThreadPool pool;
  CHECK(pool.worker_count() == 0);

  pool.parallel_for(8, 3, do_nothing);
  CHECK(pool.worker_count() == 2);
  pool.parallel_for(8, 2, do_nothing);
  CHECK(pool.worker_count() == 2);
  pool.parallel_for(8, 5, do_nothing);
  CHECK(pool.worker_count() == 4);
}

TEST_CASE("A call asks for no more threads than it has tasks", "[thread_pool]") {
  ThreadPool pool;

  pool.parallel_for(3, 16, do_nothing);

  CHECK(pool.worker_count() == 2);
}

TEST_CASE("A pool joins its threads when it is destroyed", "[thread_pool]") {
  // A thread may still be starting when its pool is destroyed.
  for (int repeat = 0; repeat < 50; ++repeat) {
    ThreadPool pool;
    pool.parallel_for(8, 4, do_nothing);
  }
  const ThreadPool never_used;

  SUCCEED("every destructor returned");
}

// --- Exceptions -------------------------------------------------------------

TEST_CASE("The exception of a task reaches the caller once the running tasks have returned",
          "[thread_pool]") {
  ThreadPool pool;
  std::atomic<int> running_count{0};
  const auto task = [&running_count](std::size_t index) {
    running_count.fetch_add(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    running_count.fetch_sub(1);
    if (index == 5) {
      throw std::runtime_error("task 5 failed");
    }
  };

  for (const std::size_t thread_count : {std::size_t{1}, std::size_t{4}}) {
    CAPTURE(thread_count);
    std::string message;
    try {
      pool.parallel_for(64, thread_count, task);
    } catch (const std::runtime_error& error) {
      message = error.what();
    }

    CHECK(message == "task 5 failed");
    CHECK(running_count.load() == 0);
  }
}

TEST_CASE("One exception reaches the caller when several tasks throw", "[thread_pool]") {
  ThreadPool pool;
  std::string message;

  try {
    pool.parallel_for(32, 4, [](std::size_t index) {
      throw std::runtime_error("task " + std::to_string(index) + " failed");
    });
  } catch (const std::runtime_error& error) {
    message = error.what();
  }

  CHECK(message.starts_with("task "));
  CHECK(message.ends_with(" failed"));
}

TEST_CASE("An exception that is not a std::exception reaches the caller unchanged",
          "[thread_pool]") {
  ThreadPool pool;
  int thrown_value = 0;

  try {
    pool.parallel_for(8, 4, [](std::size_t index) {
      if (index == 3) {
        throw 42;
      }
    });
  } catch (int value) {
    thrown_value = value;
  }

  CHECK(thrown_value == 42);
}

TEST_CASE("The pool runs the next call after a task has thrown", "[thread_pool]") {
  ThreadPool pool;
  CHECK_THROWS_AS(
      pool.parallel_for(8, 4, [](std::size_t /*index*/) { throw std::runtime_error("failed"); }),
      std::runtime_error);

  CHECK(count_calls(pool, 100, 4) == std::vector<int>(100, 1));
}

// --- Several callers --------------------------------------------------------

TEST_CASE("Calls from several threads at once all complete", "[thread_pool]") {
  ThreadPool pool;
  constexpr std::size_t kCallerCount = 8;
  constexpr std::size_t kCallCount = 50;
  constexpr std::size_t kTaskCount = 100;
  std::vector<std::atomic<std::size_t>> sums(kCallerCount);

  std::vector<std::thread> callers;
  callers.reserve(kCallerCount);
  for (std::size_t caller = 0; caller < kCallerCount; ++caller) {
    callers.emplace_back([&pool, &sums, caller] {
      for (std::size_t call = 0; call < kCallCount; ++call) {
        pool.parallel_for(kTaskCount, 4, [&sums, caller](std::size_t index) {
          sums[caller].fetch_add(index + 1, std::memory_order_relaxed);
        });
      }
    });
  }
  for (std::thread& caller : callers) {
    caller.join();
  }

  const std::size_t expected_sum = kCallCount * (kTaskCount * (kTaskCount + 1) / 2);
  for (const std::atomic<std::size_t>& sum : sums) {
    CHECK(sum.load() == expected_sum);
  }
}

TEST_CASE("A task may call the pool it runs on", "[thread_pool]") {
  ThreadPool pool;
  std::atomic<std::size_t> call_count{0};

  pool.parallel_for(4, 4, [&pool, &call_count](std::size_t /*index*/) {
    pool.parallel_for(8, 4, [&call_count](std::size_t /*inner_index*/) {
      call_count.fetch_add(1, std::memory_order_relaxed);
    });
  });

  CHECK(call_count.load() == std::size_t{32});
}

// --- The shared pool --------------------------------------------------------

TEST_CASE("The shared pool is one object for every thread", "[thread_pool]") {
  // Each test case runs in a process of its own under CTest, so these threads
  // are the first to ask for the pool, and they ask at the same moment.
  constexpr std::size_t kThreadCount = 8;
  std::vector<ThreadPool*> pools(kThreadCount, nullptr);
  std::atomic<bool> may_start{false};

  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (std::size_t thread = 0; thread < kThreadCount; ++thread) {
    threads.emplace_back([&pools, &may_start, thread] {
      while (!may_start.load()) {
        std::this_thread::yield();
      }
      pools[thread] = &parkin::cpu::shared_thread_pool();
    });
  }
  may_start.store(true);
  for (std::thread& thread : threads) {
    thread.join();
  }

  CHECK(pools == std::vector<ThreadPool*>(kThreadCount, &parkin::cpu::shared_thread_pool()));
}

TEST_CASE("The shared pool runs tasks", "[thread_pool]") {
  CHECK(count_calls(parkin::cpu::shared_thread_pool(), 100, 4) == std::vector<int>(100, 1));
}

#if defined(__unix__) || defined(__APPLE__)
TEST_CASE("A forked child gets a shared pool of its own", "[thread_pool][fork]") {
  ThreadPool& parent_pool = parkin::cpu::shared_thread_pool();
  // The parent's pool has threads, which do not exist in the child.
  parent_pool.parallel_for(8, 4, do_nothing);
  REQUIRE(parent_pool.worker_count() == 3);

  const pid_t child = fork();
  REQUIRE(child >= 0);
  if (child == 0) {
    // The child makes no Catch2 assertion and leaves through _exit(), so that
    // it neither reports for the parent nor runs the parent's destructors.
    ThreadPool& child_pool = parkin::cpu::shared_thread_pool();
    int exit_status = 0;
    if (&child_pool == &parent_pool) {
      exit_status = 1;
    } else if (child_pool.worker_count() != 0) {
      exit_status = 2;
    } else {
      std::atomic<std::size_t> sum{0};
      child_pool.parallel_for(100, 4, [&sum](std::size_t index) {
        sum.fetch_add(index + 1, std::memory_order_relaxed);
      });
      if (sum.load() != 5050) {
        exit_status = 3;
      } else if (child_pool.worker_count() != 3) {
        exit_status = 4;
      }
    }
    _exit(exit_status);
  }

  int wait_status = 0;
  REQUIRE(waitpid(child, &wait_status, 0) == child);
  REQUIRE(WIFEXITED(wait_status));
  // 1: the parent's pool; 2: it has threads; 3: a wrong result; 4: no threads started.
  CHECK(WEXITSTATUS(wait_status) == 0);
}
#endif

}  // namespace
