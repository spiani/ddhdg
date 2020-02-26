#include <pybind11/pybind11.h>

#include "py_interface/pyddhdg.h"

namespace py = pybind11;

#define DIM 3

namespace pyddhdg
{
  PYBIND11_MODULE(pyddhdg3d, m)
  {
    m.doc() = "A python interface for ddhdg in 3D";
#include "py_interface/pyddhdg_interface.h"
  }
} // namespace pyddhdg
