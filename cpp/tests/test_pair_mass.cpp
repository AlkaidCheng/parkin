/**
 * @file test_pair_mass.cpp
 * @brief Tests of the kernel for the invariant mass of a pair of particles.
 */
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <xsimd/xsimd.hpp>

#include "test_support.hpp"

namespace {

using parkin::tests::largest_ulp_distance;
using parkin::tests::PairColumns;
using parkin::tests::sample_pairs;
using parkin::tests::ulp_distance;

/// The largest difference allowed, in units in the last place (ULP), between
/// a mass and a value it must equal. The largest measured was 2, with Apple's
/// and GNU's libraries on 64-bit Arm and on x86-64, and the bound is twice
/// that.
constexpr std::uint64_t kValueUlpBound = 4;

/// The same between the kernel on a batch and on a scalar, whose math
/// functions differ. The largest measured was 4, and the bound is twice that.
constexpr std::uint64_t kBatchUlpBound = 8;

/** @brief The pair mass of eight components: (pT, eta, phi, mass) of each particle. */
template <class V>
[[nodiscard]] V pair_mass_of(const std::array<V, 8>& components) noexcept {
  return parkin::tests::pair_mass(components[0], components[1], components[2], components[3],
                                  components[4], components[5], components[6], components[7]);
}

/** @brief The pair mass of each pair, computed one pair at a time. */
template <class T>
[[nodiscard]] std::vector<T> masses_on_scalars(const PairColumns<T>& pairs) {
  std::vector<T> masses(pairs.size());
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    masses[index] = pair_mass_of<T>({pairs.pt1[index], pairs.eta1[index], pairs.phi1[index],
                                     pairs.mass1[index], pairs.pt2[index], pairs.eta2[index],
                                     pairs.phi2[index], pairs.mass2[index]});
  }
  return masses;
}

/** @brief The pair mass of each pair, computed a batch at a time. The count is a whole number of
 * batches. */
template <class T>
[[nodiscard]] std::vector<T> masses_on_batches(const PairColumns<T>& pairs) {
  using Batch = xsimd::batch<T>;
  REQUIRE(pairs.size() % Batch::size == 0);
  std::vector<T> masses(pairs.size());
  for (std::size_t offset = 0; offset < pairs.size(); offset += Batch::size) {
    const auto load = [offset](const std::vector<T>& column) {
      return Batch::load_unaligned(column.data() + offset);
    };
    pair_mass_of<Batch>({load(pairs.pt1), load(pairs.eta1), load(pairs.phi1), load(pairs.mass1),
                         load(pairs.pt2), load(pairs.eta2), load(pairs.phi2), load(pairs.mass2)})
        .store_unaligned(masses.data() + offset);
  }
  return masses;
}

// --- Values -----------------------------------------------------------------

TEMPLATE_TEST_CASE("Two massless particles back to back have the mass 2 pT", "[pair_mass]", float,
                   double) {
  const auto pi = static_cast<TestType>(3.14159265358979323846);

  const TestType mass = pair_mass_of<TestType>({45, 0, 0, 0, 45, 0, pi, 0});

  CHECK(ulp_distance(mass, TestType(90)) <= kValueUlpBound);
}

TEST_CASE("The muon pairs of the README have the masses printed there", "[pair_mass]") {
  // The README prints eight decimals.
  constexpr double kPrintedPrecision = 1e-8;

  CHECK(std::abs(pair_mass_of<double>({45.0, 0.3, 0.1, 0.106, 41.5, -0.4, 3.0, 0.106}) -
                 91.18459273) < kPrintedPrecision);
  CHECK(std::abs(pair_mass_of<double>({38.2, -1.2, 2.9, 0.106, 27.9, -1.0, -0.4, 0.106}) -
                 65.41600255) < kPrintedPrecision);
}

TEMPLATE_TEST_CASE("A particle paired with itself has twice its mass", "[pair_mass]", float,
                   double) {
  PairColumns<TestType> pairs = sample_pairs<TestType>(512);
  pairs.pt2 = pairs.pt1;
  pairs.eta2 = pairs.eta1;
  pairs.phi2 = pairs.phi1;
  pairs.mass2 = pairs.mass1;
  std::vector<TestType> twice_the_mass(pairs.size());
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    twice_the_mass[index] = TestType(2) * pairs.mass1[index];
  }

  // A collinear pair is where a formula that subtracts energies loses the mass.
  CHECK(largest_ulp_distance(masses_on_scalars(pairs), twice_the_mass) <= kValueUlpBound);
}

TEMPLATE_TEST_CASE("The mass does not depend on the order of the two particles", "[pair_mass]",
                   float, double) {
  const PairColumns<TestType> pairs = sample_pairs<TestType>(512);
  const PairColumns<TestType> swapped{pairs.pt2, pairs.eta2, pairs.phi2, pairs.mass2,
                                      pairs.pt1, pairs.eta1, pairs.phi1, pairs.mass1};

  CHECK(largest_ulp_distance(masses_on_scalars(pairs), masses_on_scalars(swapped)) <=
        kValueUlpBound);
}

// --- Conventions ------------------------------------------------------------

TEMPLATE_TEST_CASE("A particle paired with a null momentum keeps its mass and its sign",
                   "[pair_mass]", float, double) {
  const auto with_null_momentum = [](TestType mass) {
    return pair_mass_of<TestType>({TestType(5), TestType(0.5), TestType(1), mass, 0, 0, 0, 0});
  };

  CHECK(ulp_distance(with_null_momentum(3), TestType(3)) <= kValueUlpBound);
  CHECK(with_null_momentum(0) == TestType(0));
  // A negative mass denotes a spacelike momentum.
  CHECK(ulp_distance(with_null_momentum(-4), TestType(-4)) <= kValueUlpBound);
}

TEMPLATE_TEST_CASE("A pair with a negative squared mass gets a negative mass", "[pair_mass]", float,
                   double) {
  // Two equal spacelike momenta: m^2 = -16 - 16 + 2 (E^2 - |p|^2) = -64.
  const TestType mass = pair_mass_of<TestType>({5, 0, 0, -4, 5, 0, 0, -4});

  CHECK(ulp_distance(mass, TestType(-8)) <= kValueUlpBound);
}

TEMPLATE_TEST_CASE("A NaN in any component gives NaN", "[pair_mass]", float, double) {
  using Batch = xsimd::batch<TestType>;
  const std::array<TestType, 8> finite{TestType(45),    TestType(0.3),  TestType(0.1),
                                       TestType(0.106), TestType(41.5), TestType(-0.4),
                                       TestType(3.0),   TestType(0.106)};
  REQUIRE(std::isfinite(pair_mass_of<TestType>(finite)));

  for (std::size_t position = 0; position < finite.size(); ++position) {
    CAPTURE(position);
    std::array<TestType, 8> components = finite;
    components[position] = std::numeric_limits<TestType>::quiet_NaN();
    std::array<Batch, 8> batches;
    for (std::size_t index = 0; index < components.size(); ++index) {
      batches[index] = Batch(components[index]);
    }

    CHECK(std::isnan(pair_mass_of<TestType>(components)));
    CHECK(std::isnan(pair_mass_of<Batch>(batches).get(0)));
  }
}

// --- Scalars and batches ----------------------------------------------------

TEMPLATE_TEST_CASE("The kernel gives the same mass on a batch as on a scalar", "[pair_mass]", float,
                   double) {
  const PairColumns<TestType> pairs = sample_pairs<TestType>(2048);

  CHECK(largest_ulp_distance(masses_on_batches(pairs), masses_on_scalars(pairs)) <= kBatchUlpBound);
}

}  // namespace
