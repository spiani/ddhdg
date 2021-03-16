// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#ifdef USE_BOOST_PYTHON
#  include <boost/python.hpp>
#endif

#ifdef USE_PYBIND11
#  include <pybind11/pybind11.h>
#endif

#include <dealii-python-bindings/mapping_wrapper.h>

DEAL_II_NAMESPACE_OPEN

namespace python
{
  const char transform_unit_to_real_cell_docstring[] =
    " Map the point p on the unit cell to the corresponding point       \n"
    " on the real cell.                                                 \n";


  const char transform_real_to_unit_cell_docstring[] =
    " Map the point p on the real cell to the corresponding point       \n"
    " on the unit cell.                                                 \n";


  const char project_real_point_to_unit_point_on_face_docstring[] =
    " Transform the point on the real cell to the corresponding point   \n"
    " on the unit cell, and then projects it to a dim-1 point on the    \n"
    " face with the given face number face_no. Ideally the point is     \n"
    " near the face face_no, but any point in the cell can technically  \n"
    " be projected. The returned point is of dimension dim with         \n"
    " dim-1 coodinate value explicitly set to zero.                     \n";

#ifdef USE_BOOST_PYTHON
  void
  export_mapping()
  {
    boost::python::class_<MappingQGenericWrapper>(
      "MappingQGeneric",
      boost::python::init<const int, const int, const int>(
        boost::python::args("dim", "spacedim", "degree")))
      .def("transform_real_to_unit_cell",
           &MappingQGenericWrapper::transform_real_to_unit_cell,
           transform_real_to_unit_cell_docstring,
           boost::python::args("self", "cell", "point"))
      .def("transform_unit_to_real_cell",
           &MappingQGenericWrapper::transform_unit_to_real_cell,
           transform_unit_to_real_cell_docstring,
           boost::python::args("self", "cell", "point"))
      .def("project_real_point_to_unit_point_on_face",
           &MappingQGenericWrapper::project_real_point_to_unit_point_on_face,
           project_real_point_to_unit_point_on_face_docstring,
           boost::python::args("self", "cell", "face_no", "point"));
  }
#endif

#ifdef USE_PYBIND11
  void
  export_mapping(pybind11::module &m)
  {
    pybind11::class_<MappingQGenericWrapper>(m, "MappingQGeneric")
      .def(pybind11::init<const int, const int, const int>(),
           pybind11::arg("dim"),
           pybind11::arg("spacedim"),
           pybind11::arg("degree"))
      .def("transform_real_to_unit_cell",
           &MappingQGenericWrapper::transform_real_to_unit_cell,
           transform_real_to_unit_cell_docstring,
           pybind11::arg("cell"),
           pybind11::arg("point"))
      .def("transform_unit_to_real_cell",
           &MappingQGenericWrapper::transform_unit_to_real_cell,
           transform_unit_to_real_cell_docstring,
           pybind11::arg("cell"),
           pybind11::arg("point"))
      .def("project_real_point_to_unit_point_on_face",
           &MappingQGenericWrapper::project_real_point_to_unit_point_on_face,
           project_real_point_to_unit_point_on_face_docstring,
           pybind11::arg("cell"),
           pybind11::arg("face_no"),
           pybind11::arg("point"));
  }
#endif
} // namespace python

DEAL_II_NAMESPACE_CLOSE
