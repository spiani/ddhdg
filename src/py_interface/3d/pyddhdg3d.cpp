#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "py_interface/pyddhdg.h"

namespace py = pybind11;

#define DIM 3

namespace pyddhdg
{
  PYBIND11_MODULE(pyddhdg3d, m)
  {
    m.doc() = "A python interface for ddhdg in 3D";

    py::class_<dealii::Point<DIM>>(m, "Point")
      .def(py::init<double, double, double>());

#include "py_interface/pyddhdg_interface.h"
  }
} // namespace pyddhdg
