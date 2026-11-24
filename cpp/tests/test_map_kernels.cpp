/**
 * @file test_map_kernels.cpp
 * @brief Tests of the map kernels compiled into the library and of their loops.
 */
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "parkin/cpu/column_view.hpp"
#include "parkin/cpu/instruction_set.hpp"
#include "parkin/cpu/map_kernels.hpp"
#include "parkin/cpu/map_loop.hpp"
#include "test_support.hpp"

namespace {

using parkin::cpu::ColumnView;
using parkin::cpu::InstructionSet;
using parkin::cpu::kMapChunkSize;
using parkin::cpu::MapKernel;
using parkin::tests::kernel_columns;
using parkin::tests::kFloatTypeOf;
using parkin::tests::largest_ulp_distance;
using parkin::tests::pair_mass_block;
using parkin::tests::PairColumns;
using parkin::tests::run_block;
using parkin::tests::sample_pairs;

/// The largest difference allowed, in units in the last place (ULP), between
/// the loops of two instruction sets, whose math functions differ, and
/// between a loop and the kernel called from this file, which the compiler
/// may evaluate in another order. The largest measured was 4, with Apple's
/// and GNU's libraries on 64-bit Arm and on x86-64, and the bound is twice
/// that.
constexpr std::uint64_t kLoopUlpBound = 8;

/** @brief The pair mass of every pair, computed by the loop of an instruction set. */
template <class T>
[[nodiscard]] std::vector<T> masses_with(InstructionSet instruction_set,
                                         const PairColumns<T>& pairs) {
  std::vector<T> masses(pairs.size());
  run_block<T>(pair_mass_block(kFloatTypeOf<T>, instruction_set), kernel_columns(pairs), masses, 0,
               pairs.size());
  return masses;
}

TEST_CASE("The pair-mass kernel is found by its name", "[map_kernels]") {
  CHECK(parkin::cpu::find_map_kernel("pair_mass_ptetaphim") ==
        std::optional(MapKernel::kPairMassPtEtaPhiM));
  CHECK(parkin::cpu::find_map_kernel("") == std::nullopt);
  CHECK(parkin::cpu::find_map_kernel("PAIR_MASS_PTETAPHIM") == std::nullopt);
  CHECK(parkin::cpu::map_kernel_input_count(MapKernel::kPairMassPtEtaPhiM) == 8);
}

TEMPLATE_TEST_CASE("The scalar loop computes what the kernel computes", "[map_kernels]", float,
                   double) {
  const PairColumns<TestType> pairs = sample_pairs<TestType>(1000);
  std::vector<TestType> expected(pairs.size());
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    expected[index] = parkin::tests::pair_mass(
        pairs.pt1[index], pairs.eta1[index], pairs.phi1[index], pairs.mass1[index],
        pairs.pt2[index], pairs.eta2[index], pairs.phi2[index], pairs.mass2[index]);
  }

  CHECK(largest_ulp_distance(masses_with(InstructionSet::kScalar, pairs), expected) <=
        kLoopUlpBound);
}

TEMPLATE_TEST_CASE("The loop of every supported instruction set agrees with the scalar one",
                   "[map_kernels]", float, double) {
  const PairColumns<TestType> pairs = sample_pairs<TestType>(1000);
  const std::vector<TestType> scalar_masses = masses_with(InstructionSet::kScalar, pairs);

  for (const InstructionSet instruction_set : parkin::cpu::supported_instruction_sets()) {
    CAPTURE(parkin::cpu::instruction_set_name(instruction_set));
    CHECK(largest_ulp_distance(masses_with(instruction_set, pairs), scalar_masses) <=
          kLoopUlpBound);
  }
}

TEMPLATE_TEST_CASE(
    "A result does not depend on how the range is divided or how the columns are stored",
    "[map_kernels]", float, double) {
  constexpr std::size_t kCount = 1000;
  const PairColumns<TestType> pairs = sample_pairs<TestType>(kCount);
  const std::array<ColumnView, 8> columns = kernel_columns(pairs);
  // The eight components of a pair next to each other, as in an array of records.
  std::vector<TestType> records(8 * kCount);
  std::array<ColumnView, 8> record_columns{};
  for (std::size_t column = 0; column < columns.size(); ++column) {
    const auto* const values = reinterpret_cast<const TestType*>(columns[column].data);
    for (std::size_t index = 0; index < kCount; ++index) {
      records[8 * index + column] = values[index];
    }
    record_columns[column] = {reinterpret_cast<const std::byte*>(records.data() + column),
                              static_cast<std::ptrdiff_t>(8 * sizeof(TestType))};
  }

  for (const InstructionSet instruction_set : parkin::cpu::supported_instruction_sets()) {
    CAPTURE(parkin::cpu::instruction_set_name(instruction_set));
    const auto block = pair_mass_block(kFloatTypeOf<TestType>, instruction_set);
    std::vector<TestType> whole(kCount);
    run_block<TestType>(block, columns, whole, 0, kCount);

    std::vector<TestType> divided(kCount);
    run_block<TestType>(block, columns, divided, 0, 2 * kMapChunkSize);
    run_block<TestType>(block, columns, divided, 2 * kMapChunkSize, 3 * kMapChunkSize);
    run_block<TestType>(block, columns, divided, 3 * kMapChunkSize, kCount);
    CHECK(divided == whole);

    std::vector<TestType> from_records(kCount);
    run_block<TestType>(block, record_columns, from_records, 0, kCount);
    CHECK(from_records == whole);
  }
}

TEST_CASE("An instruction set the processor lacks has no loop", "[map_kernels]") {
  const std::vector<InstructionSet> supported = parkin::cpu::supported_instruction_sets();
  for (const InstructionSet instruction_set :
       {InstructionSet::kScalar, InstructionSet::kSse2, InstructionSet::kNeon64}) {
    CAPTURE(parkin::cpu::instruction_set_name(instruction_set));
    const bool is_supported =
        std::find(supported.begin(), supported.end(), instruction_set) != supported.end();
    if (is_supported) {
      CHECK(pair_mass_block(parkin::cpu::FloatType::kFloat64, instruction_set) != nullptr);
    } else {
      CHECK_THROWS_AS(pair_mass_block(parkin::cpu::FloatType::kFloat64, instruction_set),
                      std::invalid_argument);
    }
  }
}

}  // namespace
