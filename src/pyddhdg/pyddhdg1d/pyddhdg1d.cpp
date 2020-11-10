#include <pybind11/stl.h>

#include "pyddhdg/pyddhdg.h"

namespace py = pybind11;

#define DIM 1

namespace pyddhdg
{
  PYBIND11_MODULE(pyddhdg1d, m)
  {
    m.doc() = "A python interface for ddhdg in 1D";

    py::class_<dealii::Point<DIM>>(m, "Point").def(py::init<double>());

#include "pyddhdg/pyddhdg_interface.h"
  }
} // namespace pyddhdg
