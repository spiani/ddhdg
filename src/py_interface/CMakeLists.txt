find_package(pybind11 REQUIRED)

add_library(pyddhdg MODULE pyddhdg.cpp)

IF(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  # Intel (at least up to 19.0.5) confuses an external TBB installation with
  # our selected one if we use -Isystem includes. Work around this by using
  # normal includes.
  # See https://github.com/dealii/dealii/issues/8374 for details.
  TARGET_INCLUDE_DIRECTORIES(pyddhdg PRIVATE ${DEAL_II_INCLUDE_DIRS})
ELSE()
  TARGET_INCLUDE_DIRECTORIES(pyddhdg SYSTEM PRIVATE ${DEAL_II_INCLUDE_DIRS})
ENDIF()

target_link_libraries(pyddhdg PRIVATE pybind11::module)
target_link_libraries(pyddhdg PRIVATE DDHDG)
set_target_properties(pyddhdg PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")
