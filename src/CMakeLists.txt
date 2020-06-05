add_library(DDHDG SHARED
        adimensionalizer.cpp
        assembling_methods_1d.cpp
        assembling_methods_2d.cpp
        assembling_methods_3d.cpp
        boundary_conditions.cpp
        electron_mobility.cpp
        error_estimation.cpp
        function_tools.cpp
        nonlinear_iteration_results.cpp
        np_solver.cpp
        problem.cpp
        recombination_term.cpp
        solver.cpp
        thermodynamic_equilibrium.cpp
        )

if (WITH_PYTHON_INTERFACE)
    add_subdirectory(py_interface)
endif ()

DEAL_II_SETUP_TARGET(DDHDG)
