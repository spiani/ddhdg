set(_target tests)

set(_test_files
		linear_potential.cpp
		log_potential.cpp
		)

add_executable(${_target} main.cpp ${_test_files})
DEAL_II_SETUP_TARGET(${_target})
target_link_libraries(${_target} DDHDG ${GTEST_LIBRARY})
gtest_add_tests(TARGET ${_target})
