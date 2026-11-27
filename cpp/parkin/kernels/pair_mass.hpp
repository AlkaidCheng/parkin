/**
 * @file pair_mass.hpp
 * @brief The squared mass of a pair of particles, and the kernel for its mass.
 */
#ifndef PARKIN_KERNELS_PAIR_MASS_HPP
#define PARKIN_KERNELS_PAIR_MASS_HPP

#include <array>
#include <cstddef>

#include "parkin/kernels/coordinates.hpp"
#include "parkin/kernels/math.hpp"

namespace parkin::kernels {

// The squared mass of a pair is m1^2 + m2^2 + 2 (E1 E2 - p1.p2). Evaluated as
// written, the last term subtracts two numbers of the size of the squared
// energy, which costs a light or nearly collinear pair its precision.
// pair_mass_squared() is overloaded on the charts of the two momenta, and
// each overload is the cheapest form that keeps the precision those charts
// hold.

namespace detail {

/** @brief Whether a temporal chart holds the mass of a particle. */
template <class Temporal>
inline constexpr bool kHoldsMass = false;
template <class V>
inline constexpr bool kHoldsMass<Mass<V>> = true;
template <class V>
inline constexpr bool kHoldsMass<EnergyMassSquared<V>> = true;

/** @brief The energy and the squared mass of a particle, for the form of cartesian momenta. */
template <class V>
struct MassShell {
  V energy;
  V energy_squared;
  V mass_squared;
};

template <class V>
[[nodiscard]] MassShell<V> mass_shell(const Mass<V>& temporal, V momentum_squared) noexcept {
  const V zero(math::ScalarType<V>(0));
  const V mass_squared = temporal.mass * math::abs(temporal.mass);
  // A spacelike momentum with -m^2 > |p|^2 has no real energy; it is given
  // E = 0, as vector does.
  const V sum = mass_squared + momentum_squared;
  const V energy_squared = math::select(sum > zero, sum, zero);
  return {math::sqrt(energy_squared), energy_squared, mass_squared};
}

template <class V>
[[nodiscard]] MassShell<V> mass_shell(const EnergyMassSquared<V>& temporal,
                                      V /*momentum_squared*/) noexcept {
  return {temporal.energy, temporal.energy * temporal.energy, temporal.mass_squared};
}

/** @brief The momentum pz along the z axis of a particle. */
template <class Azimuthal, class Temporal, class V>
[[nodiscard]] V longitudinal_momentum(
    const Momentum<Azimuthal, Pz<V>, Temporal>& particle) noexcept {
  return particle.longitudinal.pz;
}
template <class Azimuthal, class Temporal, class V>
[[nodiscard]] V longitudinal_momentum(
    const Momentum<Azimuthal, Eta<V>, Temporal>& particle) noexcept {
  return transverse_momentum(particle.azimuthal) * math::sinh(particle.longitudinal.eta);
}

/** @brief |pT1 + pT2|^2, the squared transverse momentum of a pair. */
template <class V>
[[nodiscard]] V transverse_sum_squared(const PxPy<V>& first, const PxPy<V>& second) noexcept {
  const V px = first.px + second.px;
  const V py = first.py + second.py;
  return px * px + py * py;
}
template <class V>
[[nodiscard]] V transverse_sum_squared(const PtPhi<V>& first, const PtPhi<V>& second) noexcept {
  const V product = first.pt * second.pt;
  return first.pt * first.pt + second.pt * second.pt +
         (product + product) * math::cos(first.phi - second.phi);
}
template <class Azimuthal1, class Azimuthal2>
[[nodiscard]] auto transverse_sum_squared(const Azimuthal1& first,
                                          const Azimuthal2& second) noexcept {
  return transverse_sum_squared(cartesian(first), cartesian(second));
}

}  // namespace detail

/**
 * @brief Squared invariant mass of two particles, each held as one chart of each part.
 *
 * This form serves every pair of charts. The term E1 E2 - p1.p2 is the sum of
 * three parts, one for each part of a four-momentum, none of which subtracts
 * nearly equal numbers beyond what the coordinates themselves impose:
 *
 *   temporal      E1 E2 - |p1| |p2| = |p1| d2 + |p2| d1 + d1 d2,  d = E - |p|
 *   longitudinal  |p1| |p2| - pT1 pT2 - pz1 pz2    (longitudinal_gap)
 *   azimuthal     pT1 pT2 - pT1.pT2                (azimuthal_gap)
 *
 * For two massless particles given with their pseudorapidities and azimuthal
 * angles this is the familiar m^2 = 2 pT1 pT2 [cosh(eta1 - eta2) -
 * cos(phi1 - phi2)], written so that a collinear pair keeps its precision.
 *
 * @param first,second The two momenta, each a Momentum of charts.
 * @return The squared mass, negative for a spacelike sum. A NaN in any
 *         component gives NaN.
 */
template <class Momentum1, class Momentum2>
[[nodiscard]] auto pair_mass_squared(const Momentum1& first, const Momentum2& second) noexcept {
  const auto pt1 = transverse_momentum(first.azimuthal);
  const auto pt2 = transverse_momentum(second.azimuthal);
  const auto momentum1 = momentum(first.longitudinal, pt1);
  const auto momentum2 = momentum(second.longitudinal, pt2);
  const auto terms1 = energy_terms(first.temporal, momentum1);
  const auto terms2 = energy_terms(second.temporal, momentum2);

  const auto temporal_term = momentum1 * terms2.energy_minus_momentum +
                             momentum2 * terms1.energy_minus_momentum +
                             terms1.energy_minus_momentum * terms2.energy_minus_momentum;
  const auto spatial_term =
      longitudinal_gap(first.longitudinal, pt1, momentum1, second.longitudinal, pt2, momentum2) +
      azimuthal_gap(first.azimuthal, pt1, second.azimuthal, pt2);
  const auto dot_product = temporal_term + spatial_term;
  return terms1.mass_squared + terms2.mass_squared + (dot_product + dot_product);
}

/**
 * @brief The same for two particles given with their energies.
 *
 * An energy holds a mass only as the difference E - |p|, so no form is more
 * precise than a rounding error of the squared energy of the pair, and the
 * definition (E1 + E2)^2 - |p1 + p2|^2 reaches that with one subtraction.
 */
template <class Azimuthal1, class Longitudinal1, class Azimuthal2, class Longitudinal2, class V>
[[nodiscard]] V pair_mass_squared(
    const Momentum<Azimuthal1, Longitudinal1, Energy<V>>& first,
    const Momentum<Azimuthal2, Longitudinal2, Energy<V>>& second) noexcept {
  const V energy = first.temporal.energy + second.temporal.energy;
  const V pz = detail::longitudinal_momentum(first) + detail::longitudinal_momentum(second);
  return energy * energy -
         (detail::transverse_sum_squared(first.azimuthal, second.azimuthal) + pz * pz);
}

/**
 * @brief The same for two particles given by cartesian momenta and a chart that holds the mass.
 *
 * For p1.p2 > 0 the term E1 E2 - p1.p2 is its product with the conjugate
 * E1 E2 + p1.p2 over that conjugate, and by Lagrange's identity the product
 * is a sum without a difference:
 *
 *   E1 E2 - p1.p2 = (m1^2 E2^2 + |p1|^2 m2^2 + |p1 x p2|^2) / (E1 E2 + p1.p2)
 *
 * The cross product has the precision of the components, as in the general
 * form, and the whole costs one division and no square root beyond the
 * energies. The quotient is taken first, so that no intermediate exceeds a
 * squared energy. For p1.p2 <= 0 the difference E1 E2 - p1.p2 is a sum.
 */
template <class Temporal1, class Temporal2, class V>
  requires detail::kHoldsMass<Temporal1> && detail::kHoldsMass<Temporal2>
[[nodiscard]] V pair_mass_squared(const Momentum<PxPy<V>, Pz<V>, Temporal1>& first,
                                  const Momentum<PxPy<V>, Pz<V>, Temporal2>& second) noexcept {
  const V zero(math::ScalarType<V>(0));
  const V one(math::ScalarType<V>(1));
  const V px1 = first.azimuthal.px;
  const V py1 = first.azimuthal.py;
  const V pz1 = first.longitudinal.pz;
  const V px2 = second.azimuthal.px;
  const V py2 = second.azimuthal.py;
  const V pz2 = second.longitudinal.pz;

  const V momentum1_squared = px1 * px1 + py1 * py1 + pz1 * pz1;
  const V momentum2_squared = px2 * px2 + py2 * py2 + pz2 * pz2;
  const auto shell1 = detail::mass_shell(first.temporal, momentum1_squared);
  const auto shell2 = detail::mass_shell(second.temporal, momentum2_squared);

  const V energy_product = shell1.energy * shell2.energy;
  const V dot_product = px1 * px2 + py1 * py2 + pz1 * pz2;
  const V cross_x = py1 * pz2 - pz1 * py2;
  const V cross_y = pz1 * px2 - px1 * pz2;
  const V cross_z = px1 * py2 - py1 * px2;

  const V scale = one / (energy_product + dot_product);
  const V quotient =
      shell1.mass_squared * (shell2.energy_squared * scale) +
      momentum1_squared * (shell2.mass_squared * scale) +
      ((cross_x * scale) * cross_x + (cross_y * scale) * cross_y + (cross_z * scale) * cross_z);
  // The identity needs E^2 = m^2 + |p|^2 for both particles, which fails
  // only for an energy that was set to zero.
  const V gap = math::select(dot_product > zero && energy_product > zero, quotient,
                             energy_product - dot_product);
  return shell1.mass_squared + shell2.mass_squared + (gap + gap);
}

/**
 * @brief Invariant mass of two particles, each given in a coordinate system.
 *
 * The mass is the root of pair_mass_squared(). A pair whose squared mass is
 * negative gets the mass -sqrt(-m^2), as in ROOT's TLorentzVector::M and
 * vector's `mass`. A NaN in any input gives NaN.
 *
 * @tparam System1,System2 The Coordinates of the first and of the second particle.
 */
template <class System1, class System2>
struct PairMass {
  static constexpr std::size_t kInputCount = System1::kComponentCount + System2::kComponentCount;
  static constexpr std::size_t kOutputCount = 1;

  /**
   * @tparam V A floating-point scalar or an xsimd batch of one.
   * @param components The components of the first particle in System1, then
   *                   those of the second in System2. Momenta, energies and
   *                   masses share one unit; angles are in radians.
   * @return The invariant mass of the pair, in the same unit.
   */
  template <class V>
  [[nodiscard]] std::array<V, kOutputCount> operator()(
      const std::array<V, kInputCount>& components) const noexcept {
    const V mass_squared =
        pair_mass_squared(System1::read(components.data()),
                          System2::read(components.data() + System1::kComponentCount));
    return {math::copysign(math::sqrt(math::abs(mass_squared)), mass_squared)};
  }
};

/** @brief Invariant mass of two particles given in (pT, eta, phi, mass). */
using PairMassPtEtaPhiM = PairMass<PtEtaPhiM, PtEtaPhiM>;

}  // namespace parkin::kernels

#endif  // PARKIN_KERNELS_PAIR_MASS_HPP
