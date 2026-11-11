/**
 * @file build_info.hpp
 * @brief Facts about how the library was compiled.
 */
#ifndef PARKIN_CPU_BUILD_INFO_HPP
#define PARKIN_CPU_BUILD_INFO_HPP

#include <string>

namespace parkin::cpu {

/**
 * @brief Name the compiler that built the library.
 *
 * @return CMake's compiler id and version separated by a space, for example
 *         "GNU 14.2.0".
 */
[[nodiscard]] std::string compiler();

/**
 * @brief Give the version of the xsimd headers the library was compiled with.
 *
 * @return The version as "major.minor.patch".
 */
[[nodiscard]] std::string xsimd_version();

}  // namespace parkin::cpu

#endif  // PARKIN_CPU_BUILD_INFO_HPP
