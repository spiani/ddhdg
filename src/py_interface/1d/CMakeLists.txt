add_library(pyddhdg1d MODULE pyddhdg1d.cpp)

IF(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  # Intel (at least up to 19.0.5) confuses an external TBB installation with
  # our selected one if we use -Isystem includes. Work around this by using
  # normal includes.
  # See https://github.com/dealii/dealii/issues/8374 for details.
  TARGET_INCLUDE_DIRECTORIES(pyddhdg1d PRIVATE ${DEAL_II_INCLUDE_DIRS})
ELSE()
  TARGET_INCLUDE_DIRECTORIES(pyddhdg1d SYSTEM PRIVATE ${DEAL_II_INCLUDE_DIRS})
ENDIF()

target_link_libraries(pyddhdg1d PRIVATE pybind11::module)
target_link_libraries(pyddhdg1d PRIVATE DDHDG)
target_link_libraries(pyddhdg1d PRIVATE pyddhdg)
set_target_properties(pyddhdg1d PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                           SUFFIX "${PYTHON_MODULE_EXTENSION}")
