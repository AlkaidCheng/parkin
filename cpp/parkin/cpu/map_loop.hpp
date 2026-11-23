/**
 * @file map_loop.hpp
 * @brief The loop that applies a map kernel to a range of elements.
 *
 * A map kernel gives one output element per input element. The kernel is a
 * formula on values; this file supplies the loads, the stores and the loop,
 * once for scalars and once for the SIMD batches of an instruction set.
 */
#ifndef PARKIN_CPU_MAP_LOOP_HPP
#define PARKIN_CPU_MAP_LOOP_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include <xsimd/xsimd.hpp>

#include "parkin/cpu/column_view.hpp"

namespace parkin::cpu {

/**
 * @brief Number of elements a map loop handles at a time.
 *
 * A column that cannot be loaded in place is copied, a chunk at a time, into
 * scratch memory on the stack. Eight double columns of this size take 16 KiB,
 * a small part of the 512 KiB stack that macOS, the system with the smallest
 * default, gives a secondary thread (Table 2-1 of
 * https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/Multithreading/CreatingThreads/CreatingThreads.html).
 *
 * Chunks start at multiples of this size counted from the first element of
 * the whole array, and the size is a multiple of every vector width. The
 * vector an element is computed in therefore depends on its index alone, not
 * on how the array is divided among threads. That matters because xsimd's
 * sine picks its range reduction from the largest argument in the vector
 * (trigo_reducer::reduce in
 * https://github.com/xtensor-stack/xsimd/blob/14.3.0/include/xsimd/arch/common/xsimd_common_trigo.hpp),
 * so the last bit of an element's result can depend on its neighbours.
 */
inline constexpr std::size_t kMapChunkSize = 256;

/** @brief Loads and stores for a loop that handles one element at a time. */
template <class T>
struct ScalarLanes {
  using Scalar = T;
  using Vector = T;
  static constexpr std::size_t kLaneCount = 1;

  [[nodiscard]] static Vector load(const T* source) noexcept { return *source; }
  static void store(Vector value, T* target) noexcept { *target = value; }
};

/** @brief Loads and stores for a loop that handles one xsimd batch at a time. */
template <class T, class Arch>
struct SimdLanes {
  using Scalar = T;
  using Vector = xsimd::batch<T, Arch>;
  static constexpr std::size_t kLaneCount = Vector::size;

  [[nodiscard]] static Vector load(const T* source) noexcept {
    return Vector::load_unaligned(source);
  }
  static void store(const Vector& value, T* target) noexcept { value.store_unaligned(target); }
};

namespace detail {

/** @brief Whether the elements of a column are adjacent and aligned for T. */
template <class T>
[[nodiscard]] bool is_loadable_in_place(const ColumnView& column) noexcept {
  return column.stride == static_cast<std::ptrdiff_t>(sizeof(T)) &&
         reinterpret_cast<std::uintptr_t>(column.data) % alignof(T) == 0;
}

/** @brief Copy @p count elements of a column, starting at @p first, to adjacent memory. */
template <class T>
void copy_column(const std::byte* first, std::ptrdiff_t stride, std::size_t count,
                 T* target) noexcept {
  for (std::size_t i = 0; i < count; ++i) {
    std::memcpy(target + i, first + static_cast<std::ptrdiff_t>(i) * stride, sizeof(T));
  }
}

/** @brief Apply a kernel to the vectors that start @p offset elements into each source. */
template <class Kernel, class Lanes, std::size_t... Index>
[[nodiscard]] typename Lanes::Vector apply_kernel(
    const std::array<const typename Lanes::Scalar*, sizeof...(Index)>& sources, std::size_t offset,
    std::index_sequence<Index...> /*indices*/) noexcept {
  return Kernel{}(Lanes::load(sources[Index] + offset)...);
}

}  // namespace detail

/**
 * @brief Apply a map kernel to the elements [begin, end) of its input columns.
 *
 * Whole chunks are loaded in place from columns that allow it and stored
 * straight to the output. Other columns, and the last chunk when it is
 * partial, go through scratch memory on the stack; a partial chunk is padded
 * to a whole vector with copies of its last element, so every element is
 * computed by the same vector code.
 *
 * @tparam Kernel A kernel with a constant kInputCount and a call operator
 *                that takes kInputCount values and returns one.
 * @tparam Lanes ScalarLanes or SimdLanes for the element type.
 * @param inputs Kernel::kInputCount columns of at least @p end elements.
 * @param out Adjacent elements of the element type, aligned for it, of which
 *            [begin, end) are written. It must not overlap an input.
 * @param begin First element to compute, a multiple of kMapChunkSize.
 * @param end One past the last element to compute.
 */
template <class Kernel, class Lanes>
void map_block(const ColumnView* inputs, std::byte* out, std::size_t begin,
               std::size_t end) noexcept {
  using T = typename Lanes::Scalar;
  constexpr std::size_t kInputCount = Kernel::kInputCount;
  constexpr std::size_t kLaneCount = Lanes::kLaneCount;
  static_assert(kMapChunkSize % kLaneCount == 0, "a chunk must hold a whole number of vectors");

  std::array<std::array<T, kMapChunkSize>, kInputCount> input_scratch{};
  std::array<T, kMapChunkSize> output_scratch{};
  std::array<const T*, kInputCount> sources{};

  for (std::size_t chunk_begin = begin; chunk_begin < end; chunk_begin += kMapChunkSize) {
    const std::size_t count = std::min(kMapChunkSize, end - chunk_begin);
    const bool is_whole_chunk = count == kMapChunkSize;
    const std::size_t padded_count = (count + kLaneCount - 1) / kLaneCount * kLaneCount;

    for (std::size_t column = 0; column < kInputCount; ++column) {
      const ColumnView& view = inputs[column];
      const std::byte* first = view.data + static_cast<std::ptrdiff_t>(chunk_begin) * view.stride;
      if (is_whole_chunk && detail::is_loadable_in_place<T>(view)) {
        sources[column] = reinterpret_cast<const T*>(first);
        continue;
      }
      T* scratch = input_scratch[column].data();
      detail::copy_column(first, view.stride, count, scratch);
      std::fill(scratch + count, scratch + padded_count, scratch[count - 1]);
      sources[column] = scratch;
    }

    T* target = is_whole_chunk ? reinterpret_cast<T*>(out) + chunk_begin : output_scratch.data();
    for (std::size_t offset = 0; offset < padded_count; offset += kLaneCount) {
      Lanes::store(detail::apply_kernel<Kernel, Lanes>(sources, offset,
                                                       std::make_index_sequence<kInputCount>{}),
                   target + offset);
    }
    if (!is_whole_chunk) {
      std::memcpy(out + chunk_begin * sizeof(T), output_scratch.data(), count * sizeof(T));
    }
  }
}

}  // namespace parkin::cpu

#endif  // PARKIN_CPU_MAP_LOOP_HPP
