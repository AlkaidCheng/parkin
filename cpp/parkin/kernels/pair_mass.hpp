/**
 * @file pair_mass.hpp
 * @brief Kernels for the invariant mass of a pair of particles.
 */
#ifndef PARKIN_KERNELS_PAIR_MASS_HPP
#define PARKIN_KERNELS_PAIR_MASS_HPP

#include <array>
#include <cstddef>

#include "parkin/kernels/math.hpp"

namespace parkin::kernels {

/**
 * @brief Invariant mass of two particles given in (pT, eta, phi, mass).
 *
 * The squared mass of the pair is m1^2 + m2^2 + 2 (E1 E2 - p1.p2). Its last
 * term is evaluated as the sum of two parts, neither of which subtracts
 * nearly equal numbers:
 *
 *   E1 E2 - |p1| |p2| = |p1| d2 + |p2| d1 + d1 d2,  d = E - |p| = m^2 / (E + |p|)
 *   |p1| |p2| - p1.p2 = 2 pT1 pT2 [sinh^2((eta1 - eta2) / 2) + sin^2((phi1 - phi2) / 2)]
 *
 * with |p| = pT cosh(eta) and E = sqrt(m^2 + |p|^2). For two massless
 * particles this is the familiar m^2 = 2 pT1 pT2 [cosh(eta1 - eta2) -
 * cos(phi1 - phi2)], written so that a collinear pair keeps its precision.
 *
 * A negative mass denotes a spacelike four-momentum with m^2 = -mass^2, and a
 * pair whose squared mass is negative gets the mass -sqrt(-m^2). ROOT's
 * TLorentzVector::M and vector's `mass` use the same convention.
 *
 * A NaN in any input gives NaN.
 */
struct PairMassPtEtaPhiM {
  /** @brief Number of input columns: four per particle. */
  static constexpr std::size_t kInputCount = 8;
  /** @brief Number of output columns. */
  static constexpr std::size_t kOutputCount = 1;

  /**
   * @tparam V A floating-point scalar or an xsimd batch of one.
   * @param components (pT, eta, phi, mass) of the first particle, then of the
   *                   second: transverse momenta, non-negative;
   *                   pseudorapidities; azimuthal angles in radians, in any
   *                   range; masses, in the unit of the transverse momenta.
   * @return The invariant mass of the pair, in the same unit.
   */
  template <class V>
  [[nodiscard]] std::array<V, kOutputCount> operator()(
      const std::array<V, kInputCount>& components) const noexcept {
    using T = math::ScalarType<V>;
    const V pt1 = components[0];
    const V eta1 = components[1];
    const V phi1 = components[2];
    const V mass1 = components[3];
    const V pt2 = components[4];
    const V eta2 = components[5];
    const V phi2 = components[6];
    const V mass2 = components[7];
    const V zero(T(0));
    const V half(T(0.5));

    const V momentum1 = pt1 * math::cosh(eta1);
    const V momentum2 = pt2 * math::cosh(eta2);
    const V mass1_squared = mass1 * math::abs(mass1);
    const V mass2_squared = mass2 * math::abs(mass2);

    // A spacelike input with -m^2 > |p|^2 has no real energy; it is given
    // E = 0, as vector does.
    const V energy1_squared = mass1_squared + momentum1 * momentum1;
    const V energy2_squared = mass2_squared + momentum2 * momentum2;
    const V energy1 = math::sqrt(math::select(energy1_squared > zero, energy1_squared, zero));
    const V energy2 = math::sqrt(math::select(energy2_squared > zero, energy2_squared, zero));

    // E - |p|. The sum E + |p| is zero only for a null four-momentum, whose
    // E - |p| is zero as well.
    const V energy_plus_momentum1 = energy1 + momentum1;
    const V energy_plus_momentum2 = energy2 + momentum2;
    const V energy_minus_momentum1 =
        math::select(energy_plus_momentum1 > zero, mass1_squared / energy_plus_momentum1, zero);
    const V energy_minus_momentum2 =
        math::select(energy_plus_momentum2 > zero, mass2_squared / energy_plus_momentum2, zero);

    const V magnitude_term = momentum1 * energy_minus_momentum2 +
                             momentum2 * energy_minus_momentum1 +
                             energy_minus_momentum1 * energy_minus_momentum2;

    const V sinh_half_delta_eta = math::sinh(half * (eta1 - eta2));
    const V sin_half_delta_phi = math::sin(half * (phi1 - phi2));
    const V angle_term =
        pt1 * pt2 *
        (sinh_half_delta_eta * sinh_half_delta_eta + sin_half_delta_phi * sin_half_delta_phi);

    const V dot_product = magnitude_term + (angle_term + angle_term);
    const V mass_squared = mass1_squared + mass2_squared + (dot_product + dot_product);
    return {math::copysign(math::sqrt(math::abs(mass_squared)), mass_squared)};
  }
};

}  // namespace parkin::kernels

#endif  // PARKIN_KERNELS_PAIR_MASS_HPP
