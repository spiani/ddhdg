add_library(DDHDG SHARED
        adimensionalizer.cpp
        assembling_methods_1d.cpp
        assembling_methods_2d.cpp
        assembling_methods_3d.cpp
        boundary_conditions.cpp
        copy_on_trace.cpp
        electron_mobility.cpp
        error_estimation.cpp
        function_tools.cpp
        local_condenser.cpp
        nonlinear_iteration_results.cpp
        np_solver.cpp
        problem.cpp
        project_component.cpp
        recombination_term.cpp
        solver.cpp
        thermodynamic_equilibrium.cpp
        )

target_link_libraries(DDHDG ${Boost_LOG_LIBRARY} ${Boost_LOG_SETUP_LIBRARY})

if (WITH_PYTHON_INTERFACE)
    add_subdirectory(pyddhdg)
endif ()

DEAL_II_SETUP_TARGET(DDHDG)
