/**
 * @file test_support.hpp
 * @brief What the tests of the kernels and of their loops share.
 */
#ifndef PARKIN_TESTS_TEST_SUPPORT_HPP
#define PARKIN_TESTS_TEST_SUPPORT_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <vector>

#include "parkin/cpu/column_view.hpp"
#include "parkin/cpu/instruction_set.hpp"
#include "parkin/cpu/map_executor.hpp"
#include "parkin/cpu/map_kernels.hpp"
#include "parkin/kernels/pair_mass.hpp"

namespace parkin::tests {

// --- Values -----------------------------------------------------------------

/** @brief @p count values spread evenly over [first, last], both included. */
template <class T>
[[nodiscard]] std::vector<T> evenly_spaced(double first, double last, std::size_t count) {
  std::vector<T> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    const double fraction = static_cast<double>(index) / static_cast<double>(count - 1);
    values[index] = static_cast<T>(first + (last - first) * fraction);
  }
  return values;
}

/**
 * @brief @p count pseudo-random values in [low, high), the same on every system.
 *
 * The standard fixes the sequence of std::mt19937 but not the results of the
 * distributions, so its 32-bit numbers are scaled by hand.
 */
template <class T>
[[nodiscard]] std::vector<T> uniform_values(std::uint32_t seed, std::size_t count, double low,
                                            double high) {
  constexpr double kGeneratorRange = 4294967296.0;  // 2^32
  std::mt19937 generator(seed);
  std::vector<T> values(count);
  for (T& value : values) {
    value =
        static_cast<T>(low + (high - low) * (static_cast<double>(generator()) / kGeneratorRange));
  }
  return values;
}

/**
 * @brief Number of representable values of T that separate two values.
 *
 * It is zero for equal values, for +0 and -0, and for two NaN. A NaN and a
 * number are as far apart as can be.
 */
template <class T>
[[nodiscard]] std::uint64_t ulp_distance(T first, T second) noexcept {
  if (std::isnan(first) || std::isnan(second)) {
    return std::isnan(first) && std::isnan(second) ? 0 : std::numeric_limits<std::uint64_t>::max();
  }
  // Read as a sign and a magnitude, the bits of a float count the floats from
  // zero. This maps them to an integer that grows with the value. The bits
  // are copied out, which every compiler of the build can do; std::bit_cast
  // is missing from the standard library of GCC 10.
  const auto position = [](T value) noexcept -> std::int64_t {
    if constexpr (sizeof(T) == sizeof(std::int32_t)) {
      std::int32_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      return bits < 0 ? std::int64_t{std::numeric_limits<std::int32_t>::min()} - bits : bits;
    } else {
      std::int64_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      return bits < 0 ? std::numeric_limits<std::int64_t>::min() - bits : bits;
    }
  };
  const std::int64_t low = std::min(position(first), position(second));
  const std::int64_t high = std::max(position(first), position(second));
  // Unsigned, so that the difference of two values of opposite sign fits.
  return static_cast<std::uint64_t>(high) - static_cast<std::uint64_t>(low);
}

/** @brief The largest ulp_distance() of the elements of two ranges of one length. */
template <class T>
[[nodiscard]] std::uint64_t largest_ulp_distance(const std::vector<T>& first,
                                                 const std::vector<T>& second) noexcept {
  std::uint64_t largest =
      first.size() == second.size() ? 0 : std::numeric_limits<std::uint64_t>::max();
  for (std::size_t index = 0; index < std::min(first.size(), second.size()); ++index) {
    largest = std::max(largest, ulp_distance(first[index], second[index]));
  }
  return largest;
}

// --- Columns ----------------------------------------------------------------

/** @brief The FloatType of the element type T, which is float or double. */
template <class T>
inline constexpr cpu::FloatType kFloatTypeOf =
    sizeof(T) == sizeof(float) ? cpu::FloatType::kFloat32 : cpu::FloatType::kFloat64;

/** @brief A column that reads the adjacent elements of a vector. */
template <class T>
[[nodiscard]] cpu::ColumnView column_of(const std::vector<T>& values) noexcept {
  return {reinterpret_cast<const std::byte*>(values.data()),
          static_cast<std::ptrdiff_t>(sizeof(T))};
}

/** @brief Pairs of particles in (pT, eta, phi, mass), one vector per component. */
template <class T>
struct PairColumns {
  std::vector<T> pt1, eta1, phi1, mass1;
  std::vector<T> pt2, eta2, phi2, mass2;

  [[nodiscard]] std::size_t size() const noexcept { return pt1.size(); }
};

/**
 * @brief @p count pairs of particles, the same on every system.
 *
 * Transverse momenta from 0.1 to 1000, pseudorapidities within 4, azimuthal
 * angles within pi, and masses from 0.001 to 100 of which a third are zero.
 */
template <class T>
[[nodiscard]] PairColumns<T> sample_pairs(std::size_t count) {
  constexpr double kPi = 3.14159265358979323846;
  const auto powers_of_ten = [count](std::uint32_t seed, double low, double high) {
    std::vector<T> values = uniform_values<T>(seed, count, low, high);
    for (T& value : values) {
      value = static_cast<T>(std::pow(10.0, static_cast<double>(value)));
    }
    return values;
  };
  const auto masses = [count, &powers_of_ten](std::uint32_t seed) {
    std::vector<T> values = powers_of_ten(seed, -3, 2);
    for (std::size_t index = 0; index < count; index += 3) {
      values[index] = T(0);
    }
    return values;
  };
  return {powers_of_ten(1, -1, 3),
          uniform_values<T>(2, count, -4, 4),
          uniform_values<T>(3, count, -kPi, kPi),
          masses(4),
          powers_of_ten(5, -1, 3),
          uniform_values<T>(6, count, -4, 4),
          uniform_values<T>(7, count, -kPi, kPi),
          masses(8)};
}

// --- The interface of the kernels -------------------------------------------
//
// The tests reach the pair-mass kernel, its loops, the executor and their
// arguments through these functions alone, so a change of that interface is
// followed here.

/** @brief The pair mass of two particles in (pT, eta, phi, mass), on scalars or batches. */
template <class V>
[[nodiscard]] V pair_mass(V pt1, V eta1, V phi1, V mass1, V pt2, V eta2, V phi2, V mass2) noexcept {
  return kernels::PairMassPtEtaPhiM{}(
      std::array<V, 8>{pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2})[0];
}

/** @brief The columns of pairs, in the order the loops of the pair-mass kernel read them. */
template <class T>
[[nodiscard]] std::array<cpu::ColumnView, 8> kernel_columns(const PairColumns<T>& pairs) noexcept {
  return {column_of(pairs.pt1),   column_of(pairs.eta1), column_of(pairs.phi1),
          column_of(pairs.mass1), column_of(pairs.pt2),  column_of(pairs.eta2),
          column_of(pairs.phi2),  column_of(pairs.mass2)};
}

/** @brief The loop of the pair-mass kernel for an element type and an instruction set. */
[[nodiscard]] inline cpu::MapBlockFunction pair_mass_block(cpu::FloatType float_type,
                                                           cpu::InstructionSet instruction_set) {
  return cpu::map_block_function(cpu::MapKernel::kPairMassPtEtaPhiM, float_type, instruction_set);
}

/** @brief Apply a loop to the elements [begin, end) of columns and write the results to @p out. */
template <class T>
void run_block(cpu::MapBlockFunction block, std::span<const cpu::ColumnView> columns,
               std::vector<T>& out, std::size_t begin, std::size_t end) noexcept {
  std::byte* const output = reinterpret_cast<std::byte*>(out.data());
  block(columns.data(), &output, begin, end);
}

/** @brief Run the pair-mass kernel over whole columns with the map executor. */
template <class T>
void run_pair_mass(std::span<const cpu::ColumnView> columns, std::vector<T>& masses,
                   const cpu::MapOptions& options) {
  cpu::run_map(cpu::MapKernel::kPairMassPtEtaPhiM, kFloatTypeOf<T>, columns,
               reinterpret_cast<std::byte*>(masses.data()), masses.size(), options);
}

/** @brief The pair mass of every pair, computed by the map executor. */
template <class T>
[[nodiscard]] std::vector<T> pair_masses_by_executor(const PairColumns<T>& pairs,
                                                     const cpu::MapOptions& options) {
  const std::array<cpu::ColumnView, 8> columns = kernel_columns(pairs);
  std::vector<T> masses(pairs.size());
  run_pair_mass<T>(columns, masses, options);
  return masses;
}

}  // namespace parkin::tests

#endif  // PARKIN_TESTS_TEST_SUPPORT_HPP
