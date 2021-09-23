set(_target tests)

find_package(GTest)

if (NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG release-1.10.0
  )
  FetchContent_MakeAvailable(googletest)
endif()

set(_test_files
        adimensionalizer.cpp
        cell_face_tau_computer.cpp
        copy_on_trace.cpp
        current_equation.cpp
        disabling_components.cpp
        laplacians.cpp
        linear_potential.cpp
        local_charge_neutrality.cpp
        log_potential.cpp
        neumann_boundary_conditions.cpp
        set_component.cpp
        solution_transfer.cpp
        recombination_terms.cpp
        thermodynamic_equilibrium.cpp
        )

add_executable(${_target} main.cpp ${_test_files})
DEAL_II_SETUP_TARGET(${_target})
target_link_libraries(${_target} DDHDG GTest::GTest)

# include(GoogleTest)
# gtest_discover_tests(tests)
gtest_add_tests(TARGET ${_target})
