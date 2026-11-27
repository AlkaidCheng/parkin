/**
 * @file test_pair_mass.cpp
 * @brief Tests of the kernel for the invariant mass of a pair of particles.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <xsimd/xsimd.hpp>

#include "parkin/kernels/coordinates.hpp"
#include "parkin/kernels/pair_mass.hpp"
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

// --- Coordinate systems -----------------------------------------------------

using parkin::kernels::PairMass;

/// The eight coordinate systems: one chart of each part of a four-momentum.
using AllSystems =
    std::tuple<parkin::kernels::PxPyPzE, parkin::kernels::PxPyPzM, parkin::kernels::PxPyEtaE,
               parkin::kernels::PxPyEtaM, parkin::kernels::PtPhiPzE, parkin::kernels::PtPhiPzM,
               parkin::kernels::PtEtaPhiE, parkin::kernels::PtEtaPhiM>;

/// The largest error of the squared mass of a pair allowed, in units of
/// eps (E1 + E2)^2: the bound the library states for every coordinate
/// system, since an energy and Cartesian momenta hold a mass only as a
/// difference of numbers of that size. The largest measured here was 1.9,
/// with Apple's and GNU's libraries on 64-bit Arm and on x86-64.
constexpr double kSquaredErrorBound = 16;

/** @brief A momentum given by its Cartesian components and its mass; a negative mass is spacelike.
 */
struct CartesianMomentum {
  double px;
  double py;
  double pz;
  double mass;
};

/** @brief A component of a momentum, by name, computed in double precision. */
[[nodiscard]] double component_of(const CartesianMomentum& momentum, std::string_view name) {
  const double pt = std::hypot(momentum.px, momentum.py);
  const double momentum_squared = pt * pt + momentum.pz * momentum.pz;
  if (name == "px") {
    return momentum.px;
  }
  if (name == "py") {
    return momentum.py;
  }
  if (name == "pz") {
    return momentum.pz;
  }
  if (name == "pt") {
    return pt;
  }
  if (name == "phi") {
    return std::atan2(momentum.py, momentum.px);
  }
  if (name == "eta") {
    return std::asinh(momentum.pz / pt);
  }
  if (name == "energy") {
    return std::sqrt(momentum.mass * std::abs(momentum.mass) + momentum_squared);
  }
  if (name == "mass") {
    return momentum.mass;
  }
  FAIL("no component named " << name);
  return 0;
}

/** @brief The components of a momentum in a coordinate system, in the order it lists them, rounded
 * to T. */
template <class System, class T>
[[nodiscard]] std::array<T, System::kComponentCount> components_in(
    const CartesianMomentum& momentum) {
  std::array<T, System::kComponentCount> components{};
  for (std::size_t index = 0; index < components.size(); ++index) {
    components[index] = static_cast<T>(component_of(momentum, System::kComponentNames[index]));
  }
  return components;
}

/** @brief The component names of a coordinate system, for messages. */
template <class System>
[[nodiscard]] std::string system_name() {
  std::string name;
  for (const std::string_view component : System::kComponentNames) {
    name += (name.empty() ? "" : ",") + std::string(component);
  }
  return name;
}

/** @brief The components of two particles, the first in System1 and the second in System2. */
template <class System1, class System2, class T>
[[nodiscard]] std::array<T, PairMass<System1, System2>::kInputCount> pair_components(
    const CartesianMomentum& first, const CartesianMomentum& second) {
  std::array<T, PairMass<System1, System2>::kInputCount> components{};
  const auto first_components = components_in<System1, T>(first);
  const auto second_components = components_in<System2, T>(second);
  std::copy(first_components.begin(), first_components.end(), components.begin());
  std::copy(second_components.begin(), second_components.end(),
            components.begin() + System1::kComponentCount);
  return components;
}

/** @brief The same values, each in a batch of its own. */
template <class T, std::size_t kCount>
[[nodiscard]] std::array<xsimd::batch<T>, kCount> as_batches(const std::array<T, kCount>& values) {
  std::array<xsimd::batch<T>, kCount> batches;
  for (std::size_t index = 0; index < kCount; ++index) {
    batches[index] = xsimd::batch<T>(values[index]);
  }
  return batches;
}

/** @brief The energy of a momentum, in double precision. */
[[nodiscard]] double energy_of(const CartesianMomentum& momentum) {
  return component_of(momentum, "energy");
}

/** @brief The error of a mass as a squared mass, in units of eps (E1 + E2)^2. */
template <class T>
[[nodiscard]] double squared_mass_error(T mass, double mass_squared, double total_energy) {
  const auto computed = static_cast<double>(mass);
  const double epsilon = std::numeric_limits<T>::epsilon();
  return std::abs(computed * std::abs(computed) - mass_squared) /
         (epsilon * total_energy * total_energy);
}

/** @brief Check the mass of a pair given in System1 and System2, on a scalar and on a batch. */
template <class T, class System1, class System2>
void check_pair_mass(const CartesianMomentum& first, const CartesianMomentum& second,
                     double mass_squared) {
  using Kernel = PairMass<System1, System2>;
  const auto components = pair_components<System1, System2, T>(first, second);
  const double total_energy = energy_of(first) + energy_of(second);
  CAPTURE(system_name<System1>(), system_name<System2>());

  CHECK(squared_mass_error(Kernel{}(components)[0], mass_squared, total_energy) <=
        kSquaredErrorBound);
  CHECK(squared_mass_error(Kernel{}(as_batches(components))[0].get(0), mass_squared,
                           total_energy) <= kSquaredErrorBound);
}

template <class T, class System1, class... Systems2>
void check_with_every_second_system(std::type_identity<std::tuple<Systems2...>> /*systems*/,
                                    const CartesianMomentum& first, const CartesianMomentum& second,
                                    double mass_squared) {
  (check_pair_mass<T, System1, Systems2>(first, second, mass_squared), ...);
}

/** @brief Check the mass of a pair in each of the 64 ordered pairs of coordinate systems. */
template <class T, class... Systems>
void check_every_pair_of_systems(const CartesianMomentum& first, const CartesianMomentum& second,
                                 double mass_squared) {
  (check_with_every_second_system<T, Systems>(std::type_identity<std::tuple<Systems...>>{}, first,
                                              second, mass_squared),
   ...);
}

template <class T, class... Systems>
void check_every_pair_of_systems(std::type_identity<std::tuple<Systems...>> /*systems*/,
                                 const CartesianMomentum& first, const CartesianMomentum& second,
                                 double mass_squared) {
  check_every_pair_of_systems<T, Systems...>(first, second, mass_squared);
}

TEMPLATE_TEST_CASE("Every pair of coordinate systems gives the mass of an exact pair",
                   "[pair_mass]", float, double) {
  // (3, 4, 12) with the mass 84 has |p| = 13 and E = 85, and (6, 8, 24) with
  // the mass 168 has |p| = 26 and E = 170: one velocity, so m = 84 + 168.
  const CartesianMomentum heavy{3, 4, 12, 84};
  check_every_pair_of_systems<TestType>(std::type_identity<AllSystems>{}, heavy, {6, 8, 24, 168},
                                        252.0 * 252.0);
  // A massless particle collinear with the first: m^2 = 111^2 - 39^2.
  check_every_pair_of_systems<TestType>(std::type_identity<AllSystems>{}, heavy, {6, 8, 24, 0},
                                        10800.0);
  // A massless particle in another direction: m^2 = 7056 + 2 (85 26 - 288).
  check_every_pair_of_systems<TestType>(std::type_identity<AllSystems>{}, heavy, {-8, 6, 24, 0},
                                        10900.0);
  // Two spacelike momenta, E = 3 for |p| = 5 and E = 12 for |p| = 13, whose
  // sum is spacelike: m^2 = 15^2 - 234.
  check_every_pair_of_systems<TestType>(std::type_identity<AllSystems>{}, {3, 4, 0, -4},
                                        {0, 5, 12, -5}, -9.0);
}

/** @brief Check that a NaN in any component of a pair in System gives NaN. */
template <class T, class System>
void check_nan_in_every_component() {
  using Kernel = PairMass<System, System>;
  const auto finite =
      pair_components<System, System, T>(CartesianMomentum{3, 4, 12, 84}, {-8, 6, 24, 0});
  CAPTURE(system_name<System>());
  REQUIRE(std::isfinite(Kernel{}(finite)[0]));
  for (std::size_t position = 0; position < finite.size(); ++position) {
    CAPTURE(position);
    auto components = finite;
    components[position] = std::numeric_limits<T>::quiet_NaN();

    CHECK(std::isnan(Kernel{}(components)[0]));
    CHECK(std::isnan(Kernel{}(as_batches(components))[0].get(0)));
  }
}

template <class T, class... Systems>
void check_nan_in_every_system(std::type_identity<std::tuple<Systems...>> /*systems*/) {
  (check_nan_in_every_component<T, Systems>(), ...);
}

TEMPLATE_TEST_CASE("A NaN in any component gives NaN in every coordinate system", "[pair_mass]",
                   float, double) {
  check_nan_in_every_system<TestType>(std::type_identity<AllSystems>{});
}

/// The largest difference allowed between a squared mass on a batch and on a
/// scalar, in units of eps (E1 + E2)^2, the scale on which a sum of energies
/// and momenta holds it. The largest measured was 4.6, with Apple's and
/// GNU's libraries on 64-bit Arm and on x86-64, and the bound is twice that,
/// rounded up.
constexpr double kBatchSquaredBound = 10;

/** @brief The largest difference between squared masses of pairs on batches and on scalars, in
 * System. */
template <class T, class System>
[[nodiscard]] double largest_batch_difference(const PairColumns<T>& pairs) {
  using Kernel = PairMass<System, System>;
  using Batch = xsimd::batch<T>;
  REQUIRE(pairs.size() % Batch::size == 0);
  const auto cartesian = [](double pt, double eta, double phi, double mass) {
    return CartesianMomentum{pt * std::cos(phi), pt * std::sin(phi), pt * std::sinh(eta), mass};
  };
  std::vector<std::array<T, Kernel::kInputCount>> components(pairs.size());
  std::vector<double> total_energies(pairs.size());
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    const CartesianMomentum first =
        cartesian(pairs.pt1[index], pairs.eta1[index], pairs.phi1[index], pairs.mass1[index]);
    const CartesianMomentum second =
        cartesian(pairs.pt2[index], pairs.eta2[index], pairs.phi2[index], pairs.mass2[index]);
    components[index] = pair_components<System, System, T>(first, second);
    total_energies[index] = energy_of(first) + energy_of(second);
  }
  double largest = 0;
  for (std::size_t offset = 0; offset < pairs.size(); offset += Batch::size) {
    std::array<Batch, Kernel::kInputCount> batches;
    for (std::size_t column = 0; column < Kernel::kInputCount; ++column) {
      std::array<T, Batch::size> lanes{};
      for (std::size_t lane = 0; lane < Batch::size; ++lane) {
        lanes[lane] = components[offset + lane][column];
      }
      batches[column] = Batch::load_unaligned(lanes.data());
    }
    const Batch on_batch = Kernel{}(batches)[0];
    for (std::size_t lane = 0; lane < Batch::size; ++lane) {
      const T on_scalar = Kernel{}(components[offset + lane])[0];
      const double scalar_squared =
          static_cast<double>(on_scalar) * std::abs(static_cast<double>(on_scalar));
      largest = std::max(largest, squared_mass_error(on_batch.get(lane), scalar_squared,
                                                     total_energies[offset + lane]));
    }
  }
  return largest;
}

template <class T, class... Systems>
void check_batches_in_every_system(std::type_identity<std::tuple<Systems...>> /*systems*/,
                                   const PairColumns<T>& pairs) {
  (
      [&pairs] {
        CAPTURE(system_name<Systems>());
        CHECK(largest_batch_difference<T, Systems>(pairs) <= kBatchSquaredBound);
      }(),
      ...);
}

TEMPLATE_TEST_CASE("Every coordinate system gives the same mass on a batch as on a scalar",
                   "[pair_mass]", float, double) {
  check_batches_in_every_system<TestType>(std::type_identity<AllSystems>{},
                                          sample_pairs<TestType>(2048));
}

}  // namespace
