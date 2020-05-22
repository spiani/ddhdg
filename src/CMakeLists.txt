add_library(DDHDG SHARED
        adimensionalizer.cpp
        boundary_conditions.cpp
        electron_mobility.cpp
        function_tools.cpp
        nonlinear_iteration_results.cpp
        np_solver.cpp
        permittivity.cpp
        problem.cpp
        recombination_term.cpp
        solver.cpp
        thermodynamic_equilibrium.cpp
        )

if (WITH_PYTHON_INTERFACE)
    add_subdirectory(py_interface)
endif ()

DEAL_II_SETUP_TARGET(DDHDG)
