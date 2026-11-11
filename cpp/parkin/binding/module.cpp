/**
 * @file module.cpp
 * @brief The Python module parkin._core: bindings of the C++ core.
 */
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "parkin/cpu/build_info.hpp"
#include "parkin/cpu/cpu_features.hpp"

NB_MODULE(_core, core) {
  core.doc() = "Compiled core of parkin.";

  core.def("compiler", &parkin::cpu::compiler,
           "Return the id and version of the compiler that built the core.");
  core.def("xsimd_version", &parkin::cpu::xsimd_version,
           "Return the version of the xsimd headers the core was compiled with.");
  core.def("detected_cpu_features", &parkin::cpu::detected_cpu_features,
           "Return the SIMD instruction sets the running CPU supports.");
}
