add_library(pyddhdg SHARED pyddhdg.cpp)
target_link_libraries(pyddhdg DDHDG)
DEAL_II_SETUP_TARGET(pyddhdg)

find_package(pybind11 REQUIRED)
add_subdirectory(1d)
add_subdirectory(2d)
add_subdirectory(3d)

