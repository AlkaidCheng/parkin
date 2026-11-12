# 3.26 is the first CMake whose FindPython provides Development.SABIModule,
# the component nanobind uses for stable-ABI extension modules
# (https://cmake.org/cmake/help/latest/module/FindPython.html).
cmake_minimum_required(VERSION 3.26...4.3)

project(parkin LANGUAGES CXX)

# scikit-build-core builds the Python module. The core is C++ alone, so it and
# its C++ tests also build without Python and nanobind:
#   cmake --workflow --preset cpp-tests
option(PARKIN_BUILD_PYTHON_MODULE "Build the Python module parkin._core" ON)
option(PARKIN_BUILD_TESTS "Build the C++ tests of the core" OFF)

if(PARKIN_BUILD_PYTHON_MODULE AND NOT SKBUILD)
  message(WARNING "\
The Python module is built by scikit-build-core. Build and install it with
  pip install .
from the repository root rather than by running CMake directly.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PARKIN_WARNINGS_AS_ERRORS "Treat compiler warnings in parkin's own sources as errors" OFF)

# Applies the project's warning level to a target that compiles parkin's own
# sources.
function(parkin_set_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
    if(PARKIN_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    if(PARKIN_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()

include(cmake/xsimd.cmake)

# The host library: C++ only, with no Python or binding header.
add_library(
  parkin_cpu STATIC
  cpp/parkin/cpu/build_info.cpp
  cpp/parkin/cpu/cpu_features.cpp
  cpp/parkin/cpu/instruction_set.cpp)
target_include_directories(parkin_cpu PUBLIC cpp)
target_link_libraries(parkin_cpu PRIVATE xsimd)
target_compile_definitions(
  parkin_cpu PRIVATE PARKIN_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
                     PARKIN_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}")
parkin_set_warnings(parkin_cpu)

if(PARKIN_BUILD_PYTHON_MODULE)
  find_package(
    Python 3.11 REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)
  find_package(nanobind CONFIG REQUIRED)

  # The Python module parkin._core. STABLE_ABI takes effect on CPython 3.12 and
  # later. NB_SUPPRESS_WARNINGS keeps nanobind's headers out of parkin's warning
  # level; nanobind_add_module accepts it from nanobind 2.5.0 on (the entry of
  # that version in https://nanobind.readthedocs.io/en/latest/changelog.html).
  set(parkin_module_options STABLE_ABI)
  if(nanobind_VERSION VERSION_GREATER_EQUAL 2.5.0)
    list(APPEND parkin_module_options NB_SUPPRESS_WARNINGS)
  endif()
  nanobind_add_module(_core ${parkin_module_options} cpp/parkin/binding/module.cpp)
  target_link_libraries(_core PRIVATE parkin_cpu)
  parkin_set_warnings(_core)

  install(TARGETS _core LIBRARY DESTINATION parkin)
endif()

if(PARKIN_BUILD_TESTS)
  enable_testing()
  include(cmake/catch2.cmake)
  add_subdirectory(cpp/tests)
endif()
