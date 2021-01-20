set(_target tests)

set(_test_files
        adimensionalizer.cpp
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
target_link_libraries(${_target} DDHDG ${GTEST_LIBRARY})

include_directories(${GTEST_INCLUDE_DIRS})

gtest_add_tests(TARGET ${_target})
