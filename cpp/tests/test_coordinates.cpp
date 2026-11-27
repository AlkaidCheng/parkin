/**
 * @file test_coordinates.cpp
 * @brief Tests of the charts of a four-momentum and of what is computed from one chart.
 */
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <xsimd/xsimd.hpp>

#include "parkin/kernels/coordinates.hpp"
#include "test_support.hpp"

namespace {

using namespace parkin::kernels;
using parkin::tests::largest_ulp_distance;
using parkin::tests::PairColumns;
using parkin::tests::sample_pairs;
using parkin::tests::ulp_distance;

/// The largest difference allowed, in units in the last place (ULP), between
/// a component and a value it must equal. The largest measured was 2, with
/// Apple's and GNU's libraries on 64-bit Arm and on x86-64, and the bound is
/// twice that.
constexpr std::uint64_t kValueUlpBound = 4;

/// The same between a conversion on a batch and on a scalar, whose math
/// functions differ. The largest measured was 4, and the bound is twice that.
constexpr std::uint64_t kBatchUlpBound = 8;

/**
 * @brief One momentum in every chart: (px, py, pz) = (3, 4, 12) with the mass 84.
 *
 * Its transverse momentum is 5, its momentum 13 and its energy 85, all exact
 * in both precisions.
 */
template <class T>
struct Reference {
  T px = T(3);
  T py = T(4);
  T pz = T(12);
  T pt = T(5);
  T phi = std::atan2(T(4), T(3));
  T eta = std::asinh(T(12) / T(5));
  T momentum = T(13);
  T energy = T(85);
  T mass = T(84);
  T mass_squared = T(7056);
};

/** @brief A component of the reference momentum, by name. */
template <class T>
[[nodiscard]] T component_of(const Reference<T>& reference, std::string_view name) {
  if (name == "px") {
    return reference.px;
  }
  if (name == "py") {
    return reference.py;
  }
  if (name == "pz") {
    return reference.pz;
  }
  if (name == "pt") {
    return reference.pt;
  }
  if (name == "phi") {
    return reference.phi;
  }
  if (name == "eta") {
    return reference.eta;
  }
  if (name == "energy") {
    return reference.energy;
  }
  if (name == "mass") {
    return reference.mass;
  }
  FAIL("no component named " << name);
  return T(0);
}

/** @brief The components of the reference momentum in a coordinate system, in the order it lists
 * them. */
template <class System, class T>
[[nodiscard]] std::array<T, System::kComponentCount> components_in(const Reference<T>& reference) {
  std::array<T, System::kComponentCount> components{};
  for (std::size_t index = 0; index < components.size(); ++index) {
    components[index] = component_of(reference, System::kComponentNames[index]);
  }
  return components;
}

/** @brief Check that a system converts the reference momentum to its common coordinates. */
template <class System, class T>
void check_conversion_of_reference() {
  const Reference<T> reference;
  const std::array<T, 5> expected{reference.px, reference.py, reference.pz, reference.energy,
                                  reference.mass_squared};
  const std::array<T, System::kComponentCount> components = components_in<System, T>(reference);

  const std::array<T, 5> on_scalars = ToCommonCoordinates<System>{}(components);
  std::array<xsimd::batch<T>, System::kComponentCount> batches;
  for (std::size_t index = 0; index < components.size(); ++index) {
    batches[index] = xsimd::batch<T>(components[index]);
  }
  const std::array<xsimd::batch<T>, 5> on_batches = ToCommonCoordinates<System>{}(batches);

  for (std::size_t output = 0; output < expected.size(); ++output) {
    CAPTURE(output);
    CHECK(ulp_distance(on_scalars[output], expected[output]) <= kValueUlpBound);
    CHECK(ulp_distance(on_batches[output].get(0), expected[output]) <= kValueUlpBound);
  }
}

TEST_CASE("A coordinate system lists its components in the customary order", "[coordinates]") {
  using Names4 = std::array<std::string_view, 4>;
  CHECK(PxPyPzE::kComponentNames == Names4{"px", "py", "pz", "energy"});
  CHECK(PxPyPzM::kComponentNames == Names4{"px", "py", "pz", "mass"});
  CHECK(PxPyEtaE::kComponentNames == Names4{"px", "py", "eta", "energy"});
  CHECK(PxPyEtaM::kComponentNames == Names4{"px", "py", "eta", "mass"});
  CHECK(PtPhiPzE::kComponentNames == Names4{"pt", "phi", "pz", "energy"});
  CHECK(PtPhiPzM::kComponentNames == Names4{"pt", "phi", "pz", "mass"});
  CHECK(PtEtaPhiE::kComponentNames == Names4{"pt", "eta", "phi", "energy"});
  CHECK(PtEtaPhiM::kComponentNames == Names4{"pt", "eta", "phi", "mass"});
  CHECK(CommonCoordinates::kComponentNames ==
        std::array<std::string_view, 5>{"px", "py", "pz", "energy", "mass_squared"});
}

TEST_CASE("A coordinate system reads each chart from the place of its components",
          "[coordinates]") {
  // In (pt, eta, phi, mass) the longitudinal component lies between the two
  // azimuthal ones.
  const std::array<double, 4> polar{5.0, 0.3, 0.1, 84.0};
  const auto particle = PtEtaPhiM::read(polar.data());
  CHECK(particle.azimuthal.pt == 5.0);
  CHECK(particle.azimuthal.phi == 0.1);
  CHECK(particle.longitudinal.eta == 0.3);
  CHECK(particle.temporal.mass == 84.0);

  const std::array<double, 4> cartesian{3.0, 4.0, 12.0, 85.0};
  const auto same = PxPyPzE::read(cartesian.data());
  CHECK(same.azimuthal.px == 3.0);
  CHECK(same.azimuthal.py == 4.0);
  CHECK(same.longitudinal.pz == 12.0);
  CHECK(same.temporal.energy == 85.0);

  CHECK(PtEtaPhiM::position_of("phi") == 2);
  CHECK(PtEtaPhiM::position_of("eta") == 1);
}

TEMPLATE_TEST_CASE("Each chart of a part gives the same momentum", "[coordinates]", float, double) {
  using T = TestType;
  const Reference<T> reference;
  const PxPy<T> cartesian_transverse{reference.px, reference.py};
  const PtPhi<T> polar_transverse{reference.pt, reference.phi};

  CHECK(transverse_momentum(cartesian_transverse) == reference.pt);
  CHECK(transverse_momentum(polar_transverse) == reference.pt);
  CHECK(cartesian(cartesian_transverse).px == reference.px);
  CHECK(ulp_distance(cartesian(polar_transverse).px, reference.px) <= kValueUlpBound);
  CHECK(ulp_distance(cartesian(polar_transverse).py, reference.py) <= kValueUlpBound);

  CHECK(longitudinal_momentum(Pz<T>{reference.pz}, reference.pt) == reference.pz);
  CHECK(ulp_distance(longitudinal_momentum(Eta<T>{reference.eta}, reference.pt), reference.pz) <=
        kValueUlpBound);
  CHECK(momentum(Pz<T>{reference.pz}, reference.pt) == reference.momentum);
  CHECK(ulp_distance(momentum(Eta<T>{reference.eta}, reference.pt), reference.momentum) <=
        kValueUlpBound);

  // The energy exceeds the momentum by 72, and 72 * (85 + 13) is the squared mass.
  for (const EnergyTerms<T>& terms :
       {energy_terms(Mass<T>{reference.mass}, reference.momentum),
        energy_terms(Energy<T>{reference.energy}, reference.momentum)}) {
    CHECK(terms.energy == reference.energy);
    CHECK(terms.energy_minus_momentum == T(72));
    CHECK(terms.mass_squared == reference.mass_squared);
  }
}

TEMPLATE_TEST_CASE("A negative mass is a spacelike momentum", "[coordinates]", float, double) {
  using T = TestType;

  // m^2 = -16 and |p| = 5: the energy is 3, below the momentum.
  const EnergyTerms<T> spacelike = energy_terms(Mass<T>{T(-4)}, T(5));
  CHECK(spacelike.mass_squared == T(-16));
  CHECK(spacelike.energy == T(3));
  CHECK(spacelike.energy_minus_momentum == T(-2));

  // m^2 = -25 and |p| = 3 have no real energy, which is then zero.
  const EnergyTerms<T> without_energy = energy_terms(Mass<T>{T(-5)}, T(3));
  CHECK(without_energy.mass_squared == T(-25));
  CHECK(without_energy.energy == T(0));

  // An energy below the momentum gives a negative squared mass.
  CHECK(energy_terms(Energy<T>{T(3)}, T(5)).mass_squared == T(-16));
}

TEMPLATE_TEST_CASE("A null momentum has zero energy terms", "[coordinates]", float, double) {
  using T = TestType;
  for (const EnergyTerms<T>& terms :
       {energy_terms(Mass<T>{T(0)}, T(0)), energy_terms(Energy<T>{T(0)}, T(0))}) {
    CHECK(terms.energy == T(0));
    CHECK(terms.energy_minus_momentum == T(0));
    CHECK(terms.mass_squared == T(0));
  }
}

TEMPLATE_TEST_CASE("Every coordinate system converts to the same common coordinates",
                   "[coordinates]", float, double) {
  using T = TestType;
  check_conversion_of_reference<PxPyPzE, T>();
  check_conversion_of_reference<PxPyPzM, T>();
  check_conversion_of_reference<PxPyEtaE, T>();
  check_conversion_of_reference<PxPyEtaM, T>();
  check_conversion_of_reference<PtPhiPzE, T>();
  check_conversion_of_reference<PtPhiPzM, T>();
  check_conversion_of_reference<PtEtaPhiE, T>();
  check_conversion_of_reference<PtEtaPhiM, T>();
}

TEMPLATE_TEST_CASE("A conversion on batches agrees with the one on scalars", "[coordinates]", float,
                   double) {
  using T = TestType;
  using Batch = xsimd::batch<T>;
  using System = PtEtaPhiM;
  constexpr std::size_t kCount = 4096;
  REQUIRE(kCount % Batch::size == 0);
  const PairColumns<T> pairs = sample_pairs<T>(kCount);

  std::array<std::vector<T>, 5> on_scalars;
  std::array<std::vector<T>, 5> on_batches;
  for (std::size_t output = 0; output < 5; ++output) {
    on_scalars[output].resize(kCount);
    on_batches[output].resize(kCount);
  }
  for (std::size_t index = 0; index < kCount; ++index) {
    const std::array<T, 5> converted = ToCommonCoordinates<System>{}(std::array<T, 4>{
        pairs.pt1[index], pairs.eta1[index], pairs.phi1[index], pairs.mass1[index]});
    for (std::size_t output = 0; output < 5; ++output) {
      on_scalars[output][index] = converted[output];
    }
  }
  for (std::size_t index = 0; index < kCount; index += Batch::size) {
    const std::array<Batch, 5> converted = ToCommonCoordinates<System>{}(std::array<Batch, 4>{
        Batch::load_unaligned(&pairs.pt1[index]), Batch::load_unaligned(&pairs.eta1[index]),
        Batch::load_unaligned(&pairs.phi1[index]), Batch::load_unaligned(&pairs.mass1[index])});
    for (std::size_t output = 0; output < 5; ++output) {
      converted[output].store_unaligned(&on_batches[output][index]);
    }
  }

  for (std::size_t output = 0; output < 5; ++output) {
    CAPTURE(output);
    CHECK(largest_ulp_distance(on_batches[output], on_scalars[output]) <= kBatchUlpBound);
  }
}

}  // namespace
