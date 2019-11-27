add_library(DDHDG SHARED ddhdg.cpp boundary_conditions.cpp permittivity.cpp electron_mobility.cpp)

DEAL_II_SETUP_TARGET(DDHDG)
