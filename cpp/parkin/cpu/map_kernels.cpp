#include "parkin/cpu/map_kernels.hpp"

#include <stdexcept>
#include <string>

#include <xsimd/xsimd.hpp>

#include "parkin/cpu/map_loop.hpp"
#include "parkin/kernels/pair_mass.hpp"

namespace parkin::cpu {

namespace {

#if XSIMD_WITH_SSE2
template <class T>
using Sse2Lanes = SimdLanes<T, xsimd::sse2>;
#endif

#if XSIMD_WITH_NEON64
template <class T>
using Neon64Lanes = SimdLanes<T, xsimd::neon64>;
#endif

/** @brief The loop of one kernel for an element type, with the lanes of one instruction set. */
template <class Kernel, template <class> class LanesOf>
[[nodiscard]] MapBlockFunction block_function(FloatType float_type) noexcept {
  switch (float_type) {
    case FloatType::kFloat32:
      return &map_block<Kernel, LanesOf<float>>;
    case FloatType::kFloat64:
      return &map_block<Kernel, LanesOf<double>>;
  }
  return nullptr;
}

/** @brief The loop of any kernel for an element type, with the lanes of one instruction set. */
template <template <class> class LanesOf>
[[nodiscard]] MapBlockFunction block_function(MapKernel kernel, FloatType float_type) noexcept {
  switch (kernel) {
    case MapKernel::kPairMassPtEtaPhiM:
      return block_function<kernels::PairMassPtEtaPhiM, LanesOf>(float_type);
  }
  return nullptr;
}

}  // namespace

std::optional<MapKernel> find_map_kernel(std::string_view name) noexcept {
  if (name == "pair_mass_ptetaphim") {
    return MapKernel::kPairMassPtEtaPhiM;
  }
  return std::nullopt;
}

std::size_t map_kernel_input_count(MapKernel kernel) noexcept {
  switch (kernel) {
    case MapKernel::kPairMassPtEtaPhiM:
      return kernels::PairMassPtEtaPhiM::kInputCount;
  }
  return 0;
}

MapBlockFunction map_block_function(MapKernel kernel, FloatType float_type,
                                    InstructionSet instruction_set) {
  switch (instruction_set) {
    case InstructionSet::kScalar:
      return block_function<ScalarLanes>(kernel, float_type);
    case InstructionSet::kSse2:
#if XSIMD_WITH_SSE2
      return block_function<Sse2Lanes>(kernel, float_type);
#else
      break;
#endif
    case InstructionSet::kNeon64:
#if XSIMD_WITH_NEON64
      return block_function<Neon64Lanes>(kernel, float_type);
#else
      break;
#endif
  }
  throw std::invalid_argument("map kernels are not compiled for the instruction set '" +
                              std::string(instruction_set_name(instruction_set)) + "'");
}

}  // namespace parkin::cpu
