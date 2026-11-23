/**
 * @file test_math.cpp
 * @brief Tests of the math functions kernels are written with.
 *
 * A kernel calls the same function on a scalar and on a SIMD batch, so a
 * function must mean the same on both.
 */
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <xsimd/xsimd.hpp>

#include "parkin/kernels/math.hpp"
#include "test_support.hpp"

namespace {

namespace math = parkin::math;
using parkin::tests::evenly_spaced;
using parkin::tests::largest_ulp_distance;

/// The largest difference allowed, in units in the last place (ULP), between
/// a function of xsimd and the one of the standard library. Each is within a
/// unit or two of the exact value. The largest difference measured was 2,
/// with Apple's and GNU's libraries on 64-bit Arm and on x86-64, and the bound
/// is twice that.
constexpr std::uint64_t kUlpBound = 4;

/// The same for the square root. Both are exact in double precision. In
/// single precision xsimd 14.3.0 refines an estimate on 64-bit Arm (sqrt in
/// https://github.com/xtensor-stack/xsimd/blob/14.3.0/include/xsimd/arch/xsimd_neon.hpp),
/// which was 2 ULP off at most; the bound is twice that.
constexpr std::uint64_t kSquareRootUlpBound = 4;

/** @brief Apply a function to values, one scalar at a time. */
template <class T, class Function>
[[nodiscard]] std::vector<T> on_scalars(const std::vector<T>& arguments, Function function) {
  std::vector<T> results(arguments.size());
  std::transform(arguments.begin(), arguments.end(), results.begin(), function);
  return results;
}

/** @brief Apply a function to values, one batch at a time. The count is a whole number of batches.
 */
template <class T, class Function>
[[nodiscard]] std::vector<T> on_batches(const std::vector<T>& arguments, Function function) {
  using Batch = xsimd::batch<T>;
  REQUIRE(arguments.size() % Batch::size == 0);
  std::vector<T> results(arguments.size());
  for (std::size_t offset = 0; offset < arguments.size(); offset += Batch::size) {
    function(Batch::load_unaligned(arguments.data() + offset))
        .store_unaligned(results.data() + offset);
  }
  return results;
}

TEMPLATE_TEST_CASE("A function without rounding gives the same on a batch as on a scalar", "[math]",
                   float, double) {
  const std::vector<TestType> values = evenly_spaced<TestType>(-6, 6, 512);
  const auto magnitude = [](const auto& x) { return math::abs(x); };
  const auto negated_sign = [](const auto& x) { return math::copysign(x, -x); };
  const auto positive_part = [](const auto& x) {
    using V = std::remove_cvref_t<decltype(x)>;
    const V zero(math::ScalarType<V>(0));
    return math::select(x > zero, x, zero);
  };

  CHECK(on_batches(values, magnitude) == on_scalars(values, magnitude));
  CHECK(on_batches(values, negated_sign) == on_scalars(values, negated_sign));
  CHECK(on_batches(values, positive_part) == on_scalars(values, positive_part));
}

TEMPLATE_TEST_CASE("The square root on a batch is the one on a scalar", "[math]", float, double) {
  const std::vector<TestType> values = evenly_spaced<TestType>(0, 1000, 512);
  const auto square_root = [](const auto& x) { return math::sqrt(x); };

  CHECK(largest_ulp_distance(on_batches(values, square_root), on_scalars(values, square_root)) <=
        kSquareRootUlpBound);
}

TEMPLATE_TEST_CASE(
    "A trigonometric or hyperbolic function on a batch agrees with the one on a scalar", "[math]",
    float, double) {
  // Angles beyond a full turn either way, and the range of a pseudorapidity.
  const std::vector<TestType> angles = evenly_spaced<TestType>(-6.5, 6.5, 2048);
  const std::vector<TestType> rapidities = evenly_spaced<TestType>(-6, 6, 2048);
  const auto sine = [](const auto& x) { return math::sin(x); };
  const auto hyperbolic_sine = [](const auto& x) { return math::sinh(x); };
  const auto hyperbolic_cosine = [](const auto& x) { return math::cosh(x); };

  CHECK(largest_ulp_distance(on_batches(angles, sine), on_scalars(angles, sine)) <= kUlpBound);
  CHECK(largest_ulp_distance(on_batches(rapidities, hyperbolic_sine),
                             on_scalars(rapidities, hyperbolic_sine)) <= kUlpBound);
  CHECK(largest_ulp_distance(on_batches(rapidities, hyperbolic_cosine),
                             on_scalars(rapidities, hyperbolic_cosine)) <= kUlpBound);
}

}  // namespace
