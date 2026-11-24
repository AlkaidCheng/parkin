# The C++ tests of the core. They cover what the Python tests cannot reach,
# or reach only through other code.
add_executable(
  parkin_cpp_tests
  test_instruction_set.cpp
  test_map_executor.cpp
  test_map_kernels.cpp
  test_map_loop.cpp
  test_math.cpp
  test_pair_mass.cpp
  test_thread_pool.cpp)
# The kernels are templates in headers that include xsimd's.
target_link_libraries(parkin_cpp_tests PRIVATE parkin_cpu xsimd Catch2::Catch2WithMain)
parkin_set_warnings(parkin_cpp_tests)

# Each test case becomes a CTest test. A case takes well under a second, so
# one that is still running after two minutes has hung, and CTest stops it.
catch_discover_tests(parkin_cpp_tests PROPERTIES TIMEOUT 120)
