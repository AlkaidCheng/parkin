/**
 * @file coordinates.hpp
 * @brief The coordinates a four-momentum is given in, and what kernels compute from them.
 *
 * A four-momentum has three independent parts, and each part can be given in
 * more than one chart:
 *
 *   azimuthal     the momentum in the transverse plane   (px, py) or (pt, phi)
 *   longitudinal  the momentum along the z axis          pz or eta
 *   temporal      the energy                             energy or mass
 *
 * A coordinate system is one chart of each part, so these charts give eight
 * systems, and a further chart of one part gives a system with every chart of
 * the other two. The decomposition is that of vector (the Azimuthal,
 * Longitudinal and Temporal classes of
 * https://github.com/scikit-hep/vector/blob/v1.8.1/src/vector/_methods.py).
 *
 * A kernel names no coordinate system. It is a template on the systems of its
 * arguments and is written with the functions of this file, which are
 * overloaded on the chart. A function of one chart costs what that chart
 * needs.
 */
#ifndef PARKIN_KERNELS_COORDINATES_HPP
#define PARKIN_KERNELS_COORDINATES_HPP

#include <array>
#include <cstddef>
#include <string_view>

#include "parkin/kernels/math.hpp"

namespace parkin::kernels {

// --- Charts -----------------------------------------------------------------
//
// A chart holds the components of one part of a momentum, as values of a
// kernel's value type V: a floating-point scalar or an xsimd batch of one.

/** @brief Azimuthal chart: the Cartesian components of the transverse momentum. */
template <class V>
struct PxPy {
  static constexpr std::size_t kComponentCount = 2;
  static constexpr std::string_view kComponentNames = "px,py";
  V px;
  V py;
};

/** @brief Azimuthal chart: the transverse momentum and the azimuthal angle, in radians. */
template <class V>
struct PtPhi {
  static constexpr std::size_t kComponentCount = 2;
  static constexpr std::string_view kComponentNames = "pt,phi";
  V pt;
  V phi;
};

/** @brief Longitudinal chart: the momentum along the z axis. */
template <class V>
struct Pz {
  static constexpr std::size_t kComponentCount = 1;
  static constexpr std::string_view kComponentNames = "pz";
  V pz;
};

/** @brief Longitudinal chart: the pseudorapidity, asinh(pz / pT). */
template <class V>
struct Eta {
  static constexpr std::size_t kComponentCount = 1;
  static constexpr std::string_view kComponentNames = "eta";
  V eta;
};

/** @brief Temporal chart: the energy. */
template <class V>
struct Energy {
  static constexpr std::size_t kComponentCount = 1;
  static constexpr std::string_view kComponentNames = "energy";
  V energy;
};

/**
 * @brief Temporal chart: the mass.
 *
 * A negative mass denotes a spacelike four-momentum with m^2 = -mass^2, as in
 * ROOT's TLorentzVector::M and vector's `mass`.
 */
template <class V>
struct Mass {
  static constexpr std::size_t kComponentCount = 1;
  static constexpr std::string_view kComponentNames = "mass";
  V mass;
};

/**
 * @brief Temporal chart: the energy together with the squared mass.
 *
 * The two are redundant. The chart exists for CommonCoordinates, where it
 * keeps what either of Energy and Mass determines well: a momentum given with
 * its mass keeps that mass however large its energy is.
 */
template <class V>
struct EnergyMassSquared {
  static constexpr std::size_t kComponentCount = 2;
  static constexpr std::string_view kComponentNames = "energy,mass_squared";
  V energy;
  V mass_squared;
};

// --- Coordinate systems -----------------------------------------------------

/** @brief A four-momentum as one chart of each part. */
template <class Azimuthal, class Longitudinal, class Temporal>
struct Momentum {
  Azimuthal azimuthal;
  Longitudinal longitudinal;
  Temporal temporal;
};

namespace detail {

/** @brief A chart from its components, the first of which is at @p components. */
template <class Chart, class V>
[[nodiscard]] Chart read_chart(const V* components) noexcept {
  if constexpr (Chart::kComponentCount == 1) {
    return Chart{components[0]};
  } else {
    return Chart{components[0], components[1]};
  }
}

}  // namespace detail

/**
 * @brief A coordinate system: one chart of each part of a four-momentum.
 *
 * Its components are those of the azimuthal chart, then of the longitudinal
 * one, then of the temporal one.
 */
template <template <class> class AzimuthalChart, template <class> class LongitudinalChart,
          template <class> class TemporalChart>
struct Coordinates {
  /** @brief Number of components of a momentum in this system. */
  static constexpr std::size_t kComponentCount = AzimuthalChart<float>::kComponentCount +
                                                 LongitudinalChart<float>::kComponentCount +
                                                 TemporalChart<float>::kComponentCount;

  /** @brief The comma-separated component names of each chart, in the order of the components. */
  static constexpr std::array<std::string_view, 3> kChartComponentNames = {
      AzimuthalChart<float>::kComponentNames, LongitudinalChart<float>::kComponentNames,
      TemporalChart<float>::kComponentNames};

  /** @brief A momentum from its kComponentCount components, the first of which is at @p components.
   */
  template <class V>
  [[nodiscard]] static Momentum<AzimuthalChart<V>, LongitudinalChart<V>, TemporalChart<V>> read(
      const V* components) noexcept {
    const V* const longitudinal = components + AzimuthalChart<V>::kComponentCount;
    const V* const temporal = longitudinal + LongitudinalChart<V>::kComponentCount;
    return {detail::read_chart<AzimuthalChart<V>>(components),
            detail::read_chart<LongitudinalChart<V>>(longitudinal),
            detail::read_chart<TemporalChart<V>>(temporal)};
  }
};

/**
 * @brief The system the arguments of a kernel are brought to when they are
 *        given in different systems: (px, py, pz, energy, mass squared).
 *
 * Every system converts into it without the loss a conversion into another
 * system of four components would cause. See EnergyMassSquared.
 */
using CommonCoordinates = Coordinates<PxPy, Pz, EnergyMassSquared>;

// --- Azimuthal part ---------------------------------------------------------

/** @brief The magnitude pT of the transverse momentum. */
template <class V>
[[nodiscard]] V transverse_momentum(const PxPy<V>& azimuthal) noexcept {
  return math::sqrt(azimuthal.px * azimuthal.px + azimuthal.py * azimuthal.py);
}

template <class V>
[[nodiscard]] V transverse_momentum(const PtPhi<V>& azimuthal) noexcept {
  return azimuthal.pt;
}

/** @brief The Cartesian components of the transverse momentum. */
template <class V>
[[nodiscard]] PxPy<V> cartesian(const PxPy<V>& azimuthal) noexcept {
  return azimuthal;
}

template <class V>
[[nodiscard]] PxPy<V> cartesian(const PtPhi<V>& azimuthal) noexcept {
  return {azimuthal.pt * math::cos(azimuthal.phi), azimuthal.pt * math::sin(azimuthal.phi)};
}

// --- Longitudinal part ------------------------------------------------------

/** @brief The momentum pz along the z axis, for a transverse momentum of magnitude @p pt. */
template <class V>
[[nodiscard]] V longitudinal_momentum(const Pz<V>& longitudinal, V /*pt*/) noexcept {
  return longitudinal.pz;
}

template <class V>
[[nodiscard]] V longitudinal_momentum(const Eta<V>& longitudinal, V pt) noexcept {
  return pt * math::sinh(longitudinal.eta);
}

/** @brief The magnitude |p| of the momentum, for a transverse momentum of magnitude @p pt. */
template <class V>
[[nodiscard]] V momentum(const Pz<V>& longitudinal, V pt) noexcept {
  return math::sqrt(pt * pt + longitudinal.pz * longitudinal.pz);
}

template <class V>
[[nodiscard]] V momentum(const Eta<V>& longitudinal, V pt) noexcept {
  return pt * math::cosh(longitudinal.eta);
}

// --- Temporal part ----------------------------------------------------------

/** @brief The energy of a particle and the two quantities that locate it on its mass shell. */
template <class V>
struct EnergyTerms {
  V energy;
  V energy_minus_momentum;  ///< E - |p|, which is m^2 / (E + |p|).
  V mass_squared;           ///< Negative for a spacelike four-momentum.
};

namespace detail {

/**
 * @brief E - |p| as m^2 / (E + |p|), which subtracts nothing.
 *
 * The sum E + |p| is zero only for a null four-momentum, whose E - |p| is
 * zero as well.
 */
template <class V>
[[nodiscard]] V energy_minus_momentum(V energy, V momentum, V mass_squared) noexcept {
  const V zero(math::ScalarType<V>(0));
  const V energy_plus_momentum = energy + momentum;
  return math::select(energy_plus_momentum > zero, mass_squared / energy_plus_momentum, zero);
}

}  // namespace detail

/**
 * @brief The energy terms of a particle whose momentum has magnitude @p momentum.
 *
 * Given the mass, the energy follows without loss. Given the energy, the
 * squared mass is the product (E - |p|) (E + |p|), whose first factor is as
 * precise as the energy and the momentum determine it.
 */
template <class V>
[[nodiscard]] EnergyTerms<V> energy_terms(const Mass<V>& temporal, V momentum) noexcept {
  const V zero(math::ScalarType<V>(0));
  const V mass_squared = temporal.mass * math::abs(temporal.mass);
  // A spacelike momentum with -m^2 > |p|^2 has no real energy; it is given
  // E = 0, as vector does.
  const V energy_squared = mass_squared + momentum * momentum;
  const V energy = math::sqrt(math::select(energy_squared > zero, energy_squared, zero));
  return {energy, detail::energy_minus_momentum(energy, momentum, mass_squared), mass_squared};
}

template <class V>
[[nodiscard]] EnergyTerms<V> energy_terms(const Energy<V>& temporal, V momentum) noexcept {
  const V energy_minus_momentum = temporal.energy - momentum;
  return {temporal.energy, energy_minus_momentum,
          energy_minus_momentum * (temporal.energy + momentum)};
}

// --- Conversion -------------------------------------------------------------

/**
 * @brief Kernel that converts a momentum from a coordinate system into CommonCoordinates.
 *
 * @tparam System A Coordinates.
 */
template <class System>
struct ToCommonCoordinates {
  static constexpr std::size_t kInputCount = System::kComponentCount;
  static constexpr std::size_t kOutputCount = CommonCoordinates::kComponentCount;

  /**
   * @tparam V A floating-point scalar or an xsimd batch of one.
   * @param components The components of a momentum in System.
   * @return Its px, py, pz, energy and squared mass.
   */
  template <class V>
  [[nodiscard]] std::array<V, kOutputCount> operator()(
      const std::array<V, kInputCount>& components) const noexcept {
    const auto particle = System::read(components.data());
    const V pt = transverse_momentum(particle.azimuthal);
    const PxPy<V> transverse = cartesian(particle.azimuthal);
    const EnergyTerms<V> terms =
        energy_terms(particle.temporal, momentum(particle.longitudinal, pt));
    return {transverse.px, transverse.py, longitudinal_momentum(particle.longitudinal, pt),
            terms.energy, terms.mass_squared};
  }
};

}  // namespace parkin::kernels

#endif  // PARKIN_KERNELS_COORDINATES_HPP
