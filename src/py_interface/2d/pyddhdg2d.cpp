#include <pybind11/pybind11.h>

#include "py_interface/pyddhdg.h"

namespace py = pybind11;

#define DIM 2

namespace pyddhdg
{
  PYBIND11_MODULE(pyddhdg2d, m)
  {
    m.doc() = "A python interface for ddhdg in 2D";
#include "py_interface/pyddhdg_interface.h"
  }
} // namespace pyddhdg