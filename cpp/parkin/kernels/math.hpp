/**
 * @file math.hpp
 * @brief The math functions kernels are written with, for scalars and SIMD batches.
 *
 * A kernel calls these functions and no others, so one kernel template serves
 * every value type: a scalar, where they forward to the C++ standard library,
 * and an xsimd batch, where they forward to xsimd. Replacing the
 * implementation of a function here replaces it in every kernel.
 */
#ifndef PARKIN_KERNELS_MATH_HPP
#define PARKIN_KERNELS_MATH_HPP

#include <cmath>
#include <concepts>

#include <xsimd/xsimd.hpp>

namespace parkin::math {

/**
 * @brief The scalar type of a kernel's value type.
 *
 * A kernel builds its constants from it, as in `V(ScalarType<V>(0.5))`, so
 * that a constant is never narrowed implicitly.
 */
template <class V>
struct ScalarTypeOf {
  using Type = V;
};

template <class T, class Arch>
struct ScalarTypeOf<xsimd::batch<T, Arch>> {
  using Type = T;
};

template <class V>
using ScalarType = typename ScalarTypeOf<V>::Type;

/** @brief Absolute value. */
template <std::floating_point T>
[[nodiscard]] inline T abs(T x) noexcept {
  return std::fabs(x);
}

template <class T, class Arch>
[[nodiscard]] inline xsimd::batch<T, Arch> abs(const xsimd::batch<T, Arch>& x) noexcept {
  return xsimd::abs(x);
}

/** @brief The magnitude of @p magnitude with the sign of @p sign. */
template <std::floating_point T>
[[nodiscard]] inline T copysign(T magnitude, T sign) noexcept {
  return std::copysign(magnitude, sign);
}

template <class T, class Arch>
[[nodiscard]] inline xsimd::batch<T, Arch> copysign(const xsimd::batch<T, Arch>& magnitude,
                                                    const xsimd::batch<T, Arch>& sign) noexcept {
  return xsimd::copysign(magnitude, sign);
}

/** @brief Cosine of an angle in radians. */
template <std::floating_point T>
[[nodiscard]] inline T cos(T x) noexcept {
  return std::cos(x);
}

template <class T, class Arch>
[[nodiscard]] inline xsimd::batch<T, Arch> cos(const xsimd::batch<T, Arch>& x) noexcept {
  return xsimd::cos(x);
}

/** @brief Hyperbolic cosine. */
template <std::floating_point T>
[[nodiscard]] inline T cosh(T x) noexcept {
  return std::cosh(x);
}

template <class T, class Arch>
[[nodiscard]] inline xsimd::batch<T, Arch> cosh(const xsimd::batch<T, Arch>& x) noexcept {
  return xsimd::cosh(x);
}

/**
 * @brief Choose between two values element by element.
 *
 * @param condition The result of a comparison of two values of the kernel's
 *                  value type.
 * @return @p if_true where @p condition holds and @p if_false elsewhere.
 */
template <std::floating_point T>
[[nodiscard]] inline T select(bool condition, T if_true, T if_false) noexcept {
  return condition ? if_true : if_false;
}

template <class T, class Arch>
[[nodiscard]] inline xsimd::batch<T, Arch> select(const xsimd::batch_bool<T, Arch>& condition,
                                                  const xsimd::batch<T, Arch>& if_true,
                                                  const xsimd::batch<T, Arch>& if_false) noexcept {
  return xsimd::select(condition, if_true, if_false);
}

/** @brief Sine of an angle in radians. */
template <std::floating_point T>
[[nodiscard]] inline T sin(T x) noexcept {
  return std::sin(x);
}

template <class T, class Arch>
[[nodiscard]] inline xsimd::batch<T, Arch> sin(const xsimd::batch<T, Arch>& x) noexcept {
  return xsimd::sin(x);
}

/** @brief Hyperbolic sine. */
template <std::floating_point T>
[[nodiscard]] inline T sinh(T x) noexcept {
  return std::sinh(x);
}

template <class T, class Arch>
[[nodiscard]] inline xsimd::batch<T, Arch> sinh(const xsimd::batch<T, Arch>& x) noexcept {
  return xsimd::sinh(x);
}

/** @brief Square root. */
template <std::floating_point T>
[[nodiscard]] inline T sqrt(T x) noexcept {
  return std::sqrt(x);
}

template <class T, class Arch>
[[nodiscard]] inline xsimd::batch<T, Arch> sqrt(const xsimd::batch<T, Arch>& x) noexcept {
  return xsimd::sqrt(x);
}

}  // namespace parkin::math

#endif  // PARKIN_KERNELS_MATH_HPP
