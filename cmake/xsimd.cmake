# Provides the header-only target `xsimd` (https://github.com/xtensor-stack/xsimd).
#
# PARKIN_XSIMD_TESTED_VERSION is the xsimd release the build is tested with.
# An installed copy (a conda-forge or system package) is used when it is at
# least that version; otherwise PARKIN_XSIMD_GIT_TAG is downloaded at configure
# time. FetchContent needs an exact tag, branch or commit, so the tag defaults
# to the tested release; pass -DPARKIN_XSIMD_GIT_TAG=master to build against
# xsimd's development branch.

set(PARKIN_XSIMD_TESTED_VERSION 14.3.0)
set(PARKIN_XSIMD_GIT_TAG
    "${PARKIN_XSIMD_TESTED_VERSION}"
    CACHE STRING "xsimd tag, branch or commit downloaded when no installed copy is found")

find_package(xsimd ${PARKIN_XSIMD_TESTED_VERSION} CONFIG QUIET)

if(xsimd_FOUND)
  message(STATUS "parkin: using the installed xsimd ${xsimd_VERSION}")
else()
  message(STATUS "parkin: downloading xsimd ${PARKIN_XSIMD_GIT_TAG}")
  include(FetchContent)
  # SOURCE_SUBDIR names a directory xsimd does not have, so its own CMake
  # project, with its install rules, is not added to this build: only the
  # headers are needed, and they must stay out of the wheel.
  FetchContent_Declare(
    xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG ${PARKIN_XSIMD_GIT_TAG}
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR parkin-uses-headers-only)
  FetchContent_MakeAvailable(xsimd)
  add_library(xsimd INTERFACE)
  target_include_directories(xsimd SYSTEM INTERFACE "${xsimd_SOURCE_DIR}/include")
endif()
