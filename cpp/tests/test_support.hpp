/**
 * @file test_support.hpp
 * @brief What the tests of the kernels and of their loops share.
 */
#ifndef PARKIN_TESTS_TEST_SUPPORT_HPP
#define PARKIN_TESTS_TEST_SUPPORT_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "parkin/cpu/column_view.hpp"

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

/** @brief A column that reads the adjacent elements of a vector. */
template <class T>
[[nodiscard]] cpu::ColumnView column_of(const std::vector<T>& values) noexcept {
  return {reinterpret_cast<const std::byte*>(values.data()),
          static_cast<std::ptrdiff_t>(sizeof(T))};
}

}  // namespace parkin::tests

#endif  // PARKIN_TESTS_TEST_SUPPORT_HPP
