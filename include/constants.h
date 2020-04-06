#pragma once

#include <cmath>

namespace Ddhdg::Constants
{
  const double PI = M_PI;
  const double E  = std::exp(1.0);

  const double EPSILON0 = 8.85418781762E-12;
  const double Q        = -1.602176634E-19;
  const double KB       = 1.380649E-23;

  const std::map<std::string, double> constants = {{"pi", PI},
                                                   {"e", E},
                                                   {"eps0", EPSILON0},
                                                   {"epsilon0", EPSILON0},
                                                   {"q", Q},
                                                   {"kb", KB}};
} // namespace Ddhdg::Constants
