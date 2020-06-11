set(_target tests)

set(_test_files
        adimensionalizer.cpp
        disabling_components.cpp
        set_component.cpp
        linear_potential.cpp
        local_charge_neutrality.cpp
        log_potential.cpp
        neumann_boundary_conditions.cpp
        solution_transfer.cpp
        thermodynamic_equilibrium.cpp
        )

add_executable(${_target} main.cpp ${_test_files})
DEAL_II_SETUP_TARGET(${_target})
target_link_libraries(${_target} DDHDG ${GTEST_LIBRARY})
gtest_add_tests(TARGET ${_target})
