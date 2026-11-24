/**
 * @file test_map_executor.cpp
 * @brief Tests of running a map kernel over whole columns, on several threads.
 */
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "parkin/cpu/column_view.hpp"
#include "parkin/cpu/instruction_set.hpp"
#include "parkin/cpu/map_executor.hpp"
#include "parkin/cpu/map_kernels.hpp"
#include "test_support.hpp"

namespace {

using parkin::cpu::ColumnView;
using parkin::cpu::InstructionSet;
using parkin::cpu::MapOptions;
using parkin::tests::kernel_columns;
using parkin::tests::kFloatTypeOf;
using parkin::tests::pair_mass_block;
using parkin::tests::pair_masses_by_executor;
using parkin::tests::PairColumns;
using parkin::tests::run_block;
using parkin::tests::run_pair_mass;
using parkin::tests::sample_pairs;

/// 39 whole chunks of the map loop and a partial one.
constexpr std::size_t kPairCount = 10007;

TEMPLATE_TEST_CASE("The executor computes what the loop computes over the whole range",
                   "[map_executor]", float, double) {
  const PairColumns<TestType> pairs = sample_pairs<TestType>(kPairCount);
  const std::array<ColumnView, 8> columns = kernel_columns(pairs);

  for (const InstructionSet instruction_set : parkin::cpu::supported_instruction_sets()) {
    CAPTURE(parkin::cpu::instruction_set_name(instruction_set));
    std::vector<TestType> expected(kPairCount);
    run_block(pair_mass_block(kFloatTypeOf<TestType>, instruction_set), columns, expected, 0,
              kPairCount);

    CHECK(pair_masses_by_executor(pairs, MapOptions{1, 4096, instruction_set}) == expected);
  }
}

TEMPLATE_TEST_CASE("A result does not depend on the threads or on the block size", "[map_executor]",
                   float, double) {
  const PairColumns<TestType> pairs = sample_pairs<TestType>(kPairCount);

  for (const InstructionSet instruction_set : parkin::cpu::supported_instruction_sets()) {
    CAPTURE(parkin::cpu::instruction_set_name(instruction_set));
    const std::vector<TestType> expected =
        pair_masses_by_executor(pairs, MapOptions{1, 4096, instruction_set});

    // A block of one element is a single chunk, and one beyond the array is the whole array.
    for (const auto& [thread_count, block_size] :
         {std::pair<std::size_t, std::size_t>{1, 1}, {2, 1}, {3, 1000}, {8, 256}, {33, 100000}}) {
      CAPTURE(thread_count, block_size);
      CHECK(pair_masses_by_executor(pairs, MapOptions{thread_count, block_size, instruction_set}) ==
            expected);
    }
  }
}

TEMPLATE_TEST_CASE("Columns shorter than a chunk, and empty ones, are computed", "[map_executor]",
                   float, double) {
  const MapOptions options{4, 256, InstructionSet::kScalar};
  const PairColumns<TestType> few = sample_pairs<TestType>(5);
  const std::array<ColumnView, 8> columns = kernel_columns(few);
  std::vector<TestType> expected(few.size());
  run_block(pair_mass_block(kFloatTypeOf<TestType>, InstructionSet::kScalar), columns, expected, 0,
            few.size());

  CHECK(pair_masses_by_executor(few, options) == expected);
  CHECK(pair_masses_by_executor(sample_pairs<TestType>(0), options).empty());
}

TEST_CASE("The executor refuses what it cannot run", "[map_executor]") {
  const PairColumns<double> pairs = sample_pairs<double>(8);
  const std::array<ColumnView, 8> columns = kernel_columns(pairs);
  std::vector<double> masses(pairs.size());
  const MapOptions options{1, 4096, InstructionSet::kScalar};

  // A missing column, no thread, a block of no element.
  CHECK_THROWS_AS(run_pair_mass<double>(std::span(columns).first(7), masses, options),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      run_pair_mass<double>(columns, masses, MapOptions{0, 4096, InstructionSet::kScalar}),
      std::invalid_argument);
  CHECK_THROWS_AS(run_pair_mass<double>(columns, masses, MapOptions{1, 0, InstructionSet::kScalar}),
                  std::invalid_argument);
}

}  // namespace
