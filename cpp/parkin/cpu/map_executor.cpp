#include "parkin/cpu/map_executor.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "parkin/cpu/map_loop.hpp"
#include "parkin/cpu/thread_pool.hpp"

namespace parkin::cpu {

void run_map(MapKernel kernel, FloatType float_type, std::span<const ColumnView> inputs,
             std::byte* out, std::size_t element_count, const MapOptions& options) {
  const std::size_t input_count = map_kernel_input_count(kernel);
  if (inputs.size() != input_count) {
    throw std::invalid_argument("run_map: the kernel reads " + std::to_string(input_count) +
                                " columns, got " + std::to_string(inputs.size()));
  }
  if (options.thread_count == 0) {
    throw std::invalid_argument("run_map: thread_count must be at least 1");
  }
  if (options.block_size == 0) {
    throw std::invalid_argument("run_map: block_size must be at least 1");
  }
  const MapBlockFunction run_block =
      map_block_function(kernel, float_type, options.instruction_set);
  if (element_count == 0) {
    return;
  }

  // Blocks are whole chunks, so that chunks start at the same elements
  // whatever the block size is. A block size beyond the array is capped
  // first, which keeps the rounding within the range of std::size_t.
  const std::size_t wanted_block_size = std::min(options.block_size, element_count);
  const std::size_t chunks_per_block = (wanted_block_size - 1) / kMapChunkSize + 1;
  const std::size_t block_size = chunks_per_block * kMapChunkSize;
  const std::size_t block_count = (element_count - 1) / block_size + 1;

  const ColumnView* const columns = inputs.data();
  shared_thread_pool().parallel_for(
      block_count, options.thread_count,
      [run_block, columns, out, element_count, block_size](std::size_t block) {
        const std::size_t begin = block * block_size;
        const std::size_t end = std::min(element_count, begin + block_size);
        run_block(columns, &out, begin, end);
      });
}

}  // namespace parkin::cpu
