add_library(DDHDG SHARED
        ddhdg.cpp
        boundary_conditions.cpp
        electron_mobility.cpp
        function_tools.cpp
        permittivity.cpp
        problem.cpp
        recombination_term.cpp
        )

if (WITH_PYTHON_INTERFACE)
    add_subdirectory(py_interface)
endif ()

DEAL_II_SETUP_TARGET(DDHDG)
