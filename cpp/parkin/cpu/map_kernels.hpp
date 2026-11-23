/**
 * @file map_kernels.hpp
 * @brief The map kernels compiled into the library, and their loops.
 */
#ifndef PARKIN_CPU_MAP_KERNELS_HPP
#define PARKIN_CPU_MAP_KERNELS_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

#include "parkin/cpu/column_view.hpp"
#include "parkin/cpu/instruction_set.hpp"

namespace parkin::cpu {

/** @brief A kernel that gives one output element per input element. */
enum class MapKernel : std::uint8_t {
  kPairMassPtEtaPhiM,  ///< kernels::PairMassPtEtaPhiM, named "pair_mass_ptetaphim".
};

/** @brief The floating-point type of a kernel's inputs and output. */
enum class FloatType : std::uint8_t {
  kFloat32,
  kFloat64,
};

/**
 * @brief A loop that applies one kernel to the elements [begin, end).
 *
 * The arguments are those of map_block() in map_loop.hpp.
 */
using MapBlockFunction = void (*)(const ColumnView* inputs, std::byte* out, std::size_t begin,
                                  std::size_t end) noexcept;

/** @brief Find a map kernel by name, or nothing when no kernel has that name. */
[[nodiscard]] std::optional<MapKernel> find_map_kernel(std::string_view name) noexcept;

/** @brief Number of input columns a map kernel reads. */
[[nodiscard]] std::size_t map_kernel_input_count(MapKernel kernel) noexcept;

/**
 * @brief The loop of a map kernel for an element type and an instruction set.
 *
 * @throws std::invalid_argument If @p instruction_set is not one of
 *         supported_instruction_sets().
 */
[[nodiscard]] MapBlockFunction map_block_function(MapKernel kernel, FloatType float_type,
                                                  InstructionSet instruction_set);

}  // namespace parkin::cpu

#endif  // PARKIN_CPU_MAP_KERNELS_HPP
