# Provides the target `Catch2::Catch2WithMain` and the command
# `catch_discover_tests` for the C++ tests (https://github.com/catchorg/Catch2).
#
# PARKIN_CATCH2_TESTED_VERSION is the Catch2 release the tests are built with.
# An installed copy is used when it is at least that version; otherwise that
# release is downloaded at configure time and built with the tests.

set(PARKIN_CATCH2_TESTED_VERSION 3.16.1)

find_package(Catch2 ${PARKIN_CATCH2_TESTED_VERSION} CONFIG QUIET)

if(Catch2_FOUND)
  message(STATUS "parkin: using the installed Catch2 ${Catch2_VERSION}")
else()
  message(STATUS "parkin: downloading Catch2 ${PARKIN_CATCH2_TESTED_VERSION}")
  include(FetchContent)
  # SYSTEM keeps Catch2's headers out of parkin's warning level.
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v${PARKIN_CATCH2_TESTED_VERSION}
    GIT_SHALLOW TRUE
    SYSTEM)
  FetchContent_MakeAvailable(Catch2)
  # The module that defines catch_discover_tests.
  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
endif()

include(Catch)
