/**
 * @file instruction_set.hpp
 * @brief The instruction sets kernels are compiled for.
 */
#ifndef PARKIN_CPU_INSTRUCTION_SET_HPP
#define PARKIN_CPU_INSTRUCTION_SET_HPP

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace parkin::cpu {

/** @brief A way of executing a kernel on the host processor. */
enum class InstructionSet : std::uint8_t {
  kScalar,  ///< One element at a time, with the standard library's math functions.
  kSse2,    ///< SSE2, which every x86-64 processor has.
  kNeon64,  ///< NEON, which every 64-bit Arm processor has.
};

/** @brief The name of an instruction set, as xsimd spells it: "scalar", "sse2", "neon64". */
[[nodiscard]] std::string_view instruction_set_name(InstructionSet instruction_set) noexcept;

/**
 * @brief List the instruction sets a kernel can run with in this process.
 *
 * @return The sets that the library was compiled for and that the running
 *         processor has, from the narrowest to the widest. The first is
 *         always InstructionSet::kScalar.
 */
[[nodiscard]] std::vector<InstructionSet> supported_instruction_sets();

/** @brief The last, and widest, of supported_instruction_sets(). */
[[nodiscard]] InstructionSet widest_instruction_set();

/**
 * @brief Find a supported instruction set by name.
 *
 * @return The set, or nothing when @p name is not the name of one of
 *         supported_instruction_sets().
 */
[[nodiscard]] std::optional<InstructionSet> find_instruction_set(std::string_view name);

}  // namespace parkin::cpu

#endif  // PARKIN_CPU_INSTRUCTION_SET_HPP
