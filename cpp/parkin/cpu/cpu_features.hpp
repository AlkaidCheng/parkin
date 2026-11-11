/**
 * @file cpu_features.hpp
 * @brief Detection of the SIMD instruction sets of the running CPU.
 */
#ifndef PARKIN_CPU_CPU_FEATURES_HPP
#define PARKIN_CPU_CPU_FEATURES_HPP

#include <string>
#include <vector>

namespace parkin::cpu {

/**
 * @brief List the SIMD instruction sets the running CPU supports.
 *
 * The CPU is queried once per process. The names are those of xsimd's
 * architectures, written as in "sse2", "sse4.2", "fma3", "avx2", "avx512f",
 * "neon64" and "sve256".
 *
 * @return The supported instruction sets, from the oldest to the newest of
 *         each processor family. Empty on a processor family other than
 *         x86-64 and Arm.
 */
[[nodiscard]] std::vector<std::string> detected_cpu_features();

}  // namespace parkin::cpu

#endif  // PARKIN_CPU_CPU_FEATURES_HPP
