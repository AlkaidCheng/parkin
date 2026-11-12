/**
 * @file test_instruction_set.cpp
 * @brief Tests of the instruction sets kernels are compiled for.
 */
#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "parkin/cpu/instruction_set.hpp"

namespace {

using parkin::cpu::InstructionSet;

TEST_CASE("The supported instruction sets start with scalar and do not repeat",
          "[instruction_set]") {
  const std::vector<InstructionSet> supported = parkin::cpu::supported_instruction_sets();

  REQUIRE_FALSE(supported.empty());
  CHECK(supported.front() == InstructionSet::kScalar);
  for (const InstructionSet instruction_set : supported) {
    CHECK(std::count(supported.begin(), supported.end(), instruction_set) == 1);
  }
}

TEST_CASE("The widest instruction set is the last supported one", "[instruction_set]") {
  CHECK(parkin::cpu::widest_instruction_set() == parkin::cpu::supported_instruction_sets().back());
}

TEST_CASE("A supported instruction set is found by its name", "[instruction_set]") {
  for (const InstructionSet instruction_set : parkin::cpu::supported_instruction_sets()) {
    const std::string name(parkin::cpu::instruction_set_name(instruction_set));
    CAPTURE(name);

    CHECK_FALSE(name.empty());
    CHECK(parkin::cpu::find_instruction_set(name) == std::optional(instruction_set));
  }
}

TEST_CASE("A name that is not a supported instruction set finds nothing", "[instruction_set]") {
  CHECK(parkin::cpu::find_instruction_set("avx9000") == std::nullopt);
  CHECK(parkin::cpu::find_instruction_set("") == std::nullopt);
  CHECK(parkin::cpu::find_instruction_set("SCALAR") == std::nullopt);
}

}  // namespace
