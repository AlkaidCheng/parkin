/**
 * @file map_executor.hpp
 * @brief Running a map kernel over whole columns, on several threads.
 */
#ifndef PARKIN_CPU_MAP_EXECUTOR_HPP
#define PARKIN_CPU_MAP_EXECUTOR_HPP

#include <cstddef>
#include <span>

#include "parkin/cpu/column_view.hpp"
#include "parkin/cpu/instruction_set.hpp"
#include "parkin/cpu/map_kernels.hpp"

namespace parkin::cpu {

/** @brief How run_map() executes a kernel. */
struct MapOptions {
  /// Largest number of threads to use, the calling thread included.
  std::size_t thread_count = 1;
  /// Number of elements a thread takes at a time, rounded up to a multiple of
  /// kMapChunkSize. An array of at most this many elements runs on the
  /// calling thread alone.
  std::size_t block_size = 1;
  InstructionSet instruction_set = InstructionSet::kScalar;
};

/**
 * @brief Apply a map kernel to every element of its input columns.
 *
 * The array is divided into blocks, which the threads take one after another
 * until none is left. For a given instruction set, the result does not depend
 * on the number of threads or on the block size.
 *
 * @param kernel The kernel to apply.
 * @param float_type The element type of every input column and of @p out.
 * @param inputs One column per input of the kernel, each of
 *               @p element_count elements.
 * @param out Where the results go: @p element_count adjacent elements,
 *            aligned for the element type, that overlap no input.
 * @param element_count Number of elements in each column.
 * @param options Threads, block size and instruction set.
 * @throws std::invalid_argument If the number of inputs is not the kernel's,
 *         if the thread count or the block size is zero, or if the
 *         instruction set is not one of supported_instruction_sets().
 * @note The memory behind @p inputs and @p out must stay valid and
 *       unmodified by others for the duration of the call; nothing is kept
 *       after it returns.
 */
void run_map(MapKernel kernel, FloatType float_type, std::span<const ColumnView> inputs,
             std::byte* out, std::size_t element_count, const MapOptions& options);

}  // namespace parkin::cpu

#endif  // PARKIN_CPU_MAP_EXECUTOR_HPP
