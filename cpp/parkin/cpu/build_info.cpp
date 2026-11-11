#include "parkin/cpu/build_info.hpp"

#include <xsimd/xsimd.hpp>

namespace parkin::cpu {

std::string compiler() {
  // Both macros are string literals defined by CMakeLists.txt.
  return std::string(PARKIN_CXX_COMPILER_ID) + " " + PARKIN_CXX_COMPILER_VERSION;
}

std::string xsimd_version() {
  return std::to_string(XSIMD_VERSION_MAJOR) + "." + std::to_string(XSIMD_VERSION_MINOR) + "." +
         std::to_string(XSIMD_VERSION_PATCH);
}

}  // namespace parkin::cpu
