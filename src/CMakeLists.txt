add_library(DDHDG SHARED ddhdg.cpp boundary_conditions.cpp permittivity.cpp electron_mobility.cpp recombination_term.cpp function_tools.cpp)

if(WITH_PYTHON_INTERFACE)
  add_subdirectory(py_interface)
endif()

DEAL_II_SETUP_TARGET(DDHDG)
