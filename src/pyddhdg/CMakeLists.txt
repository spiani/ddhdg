find_package(pybind11 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(pyddhdg STATIC pyddhdg.cpp)
target_include_directories(pyddhdg PRIVATE ${pybind11_INCLUDE_DIRS})
target_link_libraries(pyddhdg DDHDG Eigen3::Eigen)
DEAL_II_SETUP_TARGET(pyddhdg)

set_target_properties(pyddhdg PROPERTIES CXX_VISIBILITY_PRESET hidden)

add_library(pyddhdg_common MODULE pyddhdg_common.cpp)

IF (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # Intel (at least up to 19.0.5) confuses an external TBB installation with
    # our selected one if we use -Isystem includes. Work around this by using
    # normal includes.
    # See https://github.com/dealii/dealii/issues/8374 for details.
    TARGET_INCLUDE_DIRECTORIES(pyddhdg_common PRIVATE ${DEAL_II_INCLUDE_DIRS})
ELSE ()
    TARGET_INCLUDE_DIRECTORIES(pyddhdg_common SYSTEM PRIVATE ${DEAL_II_INCLUDE_DIRS})
ENDIF ()

target_link_libraries(pyddhdg_common pyddhdg pybind11::module)
set_target_properties(pyddhdg_common PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")

add_subdirectory(pyddhdg1d)
add_subdirectory(pyddhdg2d)
add_subdirectory(pyddhdg3d)

add_custom_command(
        TARGET pyddhdg POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/src/pyddhdg/pyddhdg.py
        ${CMAKE_CURRENT_BINARY_DIR}/__init__.py)
