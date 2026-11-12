# The C++ tests of the core. They cover what the Python tests cannot reach,
# or reach only through other code.
add_executable(parkin_cpp_tests test_instruction_set.cpp)
target_link_libraries(parkin_cpp_tests PRIVATE parkin_cpu Catch2::Catch2WithMain)
parkin_set_warnings(parkin_cpp_tests)

# Each test case becomes a CTest test. A case takes well under a second, so
# one that is still running after two minutes has hung, and CTest stops it.
catch_discover_tests(parkin_cpp_tests PROPERTIES TIMEOUT 120)
