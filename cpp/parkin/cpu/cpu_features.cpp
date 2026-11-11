#include "parkin/cpu/cpu_features.hpp"

#include <array>
#include <string_view>

#include <xsimd/xsimd.hpp>

namespace parkin::cpu {

namespace {

struct CpuFeature {
  std::string_view name;
  bool is_supported;
};

}  // namespace

std::vector<std::string> detected_cpu_features() {
  const auto arch = xsimd::available_architectures();
  // xsimd reports one flag for FMA3 on x86-64 (fma3_sse42); fma3_avx and
  // fma3_avx2 combine it with the AVX flags listed here.
  const std::array features{
      CpuFeature{"sse2", arch.sse2 != 0},         CpuFeature{"sse3", arch.sse3 != 0},
      CpuFeature{"ssse3", arch.ssse3 != 0},       CpuFeature{"sse4.1", arch.sse4_1 != 0},
      CpuFeature{"sse4.2", arch.sse4_2 != 0},     CpuFeature{"avx", arch.avx != 0},
      CpuFeature{"fma3", arch.fma3_sse42 != 0},   CpuFeature{"avx2", arch.avx2 != 0},
      CpuFeature{"avx512f", arch.avx512f != 0},   CpuFeature{"avx512cd", arch.avx512cd != 0},
      CpuFeature{"avx512dq", arch.avx512dq != 0}, CpuFeature{"avx512bw", arch.avx512bw != 0},
      CpuFeature{"avx512vl", arch.avx512vl != 0}, CpuFeature{"neon", arch.neon != 0},
      CpuFeature{"neon64", arch.neon64 != 0},     CpuFeature{"sve128", arch.sve128 != 0},
      CpuFeature{"sve256", arch.sve256 != 0},     CpuFeature{"sve512", arch.sve512 != 0},
  };

  std::vector<std::string> supported;
  for (const CpuFeature& feature : features) {
    if (feature.is_supported) {
      supported.emplace_back(feature.name);
    }
  }
  return supported;
}

}  // namespace parkin::cpu
