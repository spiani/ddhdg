add_library(pyddhdg3d MODULE pyddhdg3d.cpp)

IF (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # Intel (at least up to 19.0.5) confuses an external TBB installation with
    # our selected one if we use -Isystem includes. Work around this by using
    # normal includes.
    # See https://github.com/dealii/dealii/issues/8374 for details.
    TARGET_INCLUDE_DIRECTORIES(pyddhdg3d PRIVATE ${DEAL_II_INCLUDE_DIRS})
ELSE ()
    TARGET_INCLUDE_DIRECTORIES(pyddhdg3d SYSTEM PRIVATE ${DEAL_II_INCLUDE_DIRS})
ENDIF ()

add_dependencies(pyddhdg3d dealii-python-bindings pyddhdg_common)

target_link_libraries(pyddhdg3d PRIVATE pybind11::module)
target_link_libraries(pyddhdg3d PRIVATE DDHDG)
target_link_libraries(pyddhdg3d PRIVATE pyddhdg)
set_target_properties(pyddhdg3d PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")
