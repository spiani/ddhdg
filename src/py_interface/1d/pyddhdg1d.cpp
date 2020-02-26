#include <pybind11/pybind11.h>

#include "py_interface/pyddhdg.h"

namespace py = pybind11;

#define DIM 1

namespace pyddhdg
{
  PYBIND11_MODULE(pyddhdg1d, m)
  {
    m.doc() = "A python interface for ddhdg in 1D";
#include "py_interface/pyddhdg_interface.h"
  }
} // namespace pyddhdg
