#include "parkin/cpu/instruction_set.hpp"

#include <xsimd/xsimd.hpp>

namespace parkin::cpu {

std::string_view instruction_set_name(InstructionSet instruction_set) noexcept {
  switch (instruction_set) {
    case InstructionSet::kScalar:
      return "scalar";
    case InstructionSet::kSse2:
      return "sse2";
    case InstructionSet::kNeon64:
      return "neon64";
  }
  return "";
}

std::vector<InstructionSet> supported_instruction_sets() {
  // The SIMD sets listed here are part of the baseline the library is
  // compiled for, so the running processor has them.
  std::vector<InstructionSet> instruction_sets{InstructionSet::kScalar};
#if XSIMD_WITH_SSE2
  instruction_sets.push_back(InstructionSet::kSse2);
#endif
#if XSIMD_WITH_NEON64
  instruction_sets.push_back(InstructionSet::kNeon64);
#endif
  return instruction_sets;
}

InstructionSet widest_instruction_set() { return supported_instruction_sets().back(); }

std::optional<InstructionSet> find_instruction_set(std::string_view name) {
  for (const InstructionSet instruction_set : supported_instruction_sets()) {
    if (instruction_set_name(instruction_set) == name) {
      return instruction_set;
    }
  }
  return std::nullopt;
}

}  // namespace parkin::cpu
