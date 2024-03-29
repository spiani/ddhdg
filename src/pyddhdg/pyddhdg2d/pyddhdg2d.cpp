#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "pyddhdg/pyddhdg.h"

namespace py = pybind11;

#define DIM 2

namespace pyddhdg
{
  PYBIND11_MODULE(pyddhdg2d, m)
  {
    m.doc() = "A python interface for ddhdg in 2D";

    py::class_<dealii::Point<DIM>>(m, "Point").def(py::init<double, double>());

#include "pyddhdg/pyddhdg_interface.h"
  }
} // namespace pyddhdg
