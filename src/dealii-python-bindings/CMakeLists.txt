## ---------------------------------------------------------------------
##
## Copyright (C) 2016 - 2020 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE.md at
## the top level directory of deal.II.
##
## ---------------------------------------------------------------------

find_package(pybind11 REQUIRED)

SET(_src
        wrappers.cc
        export_tria_accessor.cc
        export_cell_accessor.cc
        export_point.cc
        export_triangulation.cc
        export_mapping.cc
        export_manifold.cc
        export_quadrature.cc
        cell_accessor_wrapper.cc
        tria_accessor_wrapper.cc
        point_wrapper.cc
        triangulation_wrapper.cc
        mapping_wrapper.cc
        manifold_wrapper.cc
        quadrature_wrapper.cc
        )

add_library(dealii-python-bindings MODULE ${_src})

IF (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # Intel (at least up to 19.0.5) confuses an external TBB installation with
    # our selected one if we use -Isystem includes. Work around this by using
    # normal includes.
    # See https://github.com/dealii/dealii/issues/8374 for details.
    TARGET_INCLUDE_DIRECTORIES(dealii-python-bindings PRIVATE ${DEAL_II_INCLUDE_DIRS})
ELSE ()
    TARGET_INCLUDE_DIRECTORIES(dealii-python-bindings SYSTEM PRIVATE ${DEAL_II_INCLUDE_DIRS})
ENDIF ()

set_target_properties(dealii-python-bindings PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        "${CMAKE_BINARY_DIR}/src/PyDealII/")

set_target_properties(dealii-python-bindings PROPERTIES OUTPUT_NAME "PyDealII")

target_link_libraries(dealii-python-bindings pybind11::module DDHDG)
set_target_properties(dealii-python-bindings PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")
