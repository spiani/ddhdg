#pragma once

namespace Ddhdg
{
  DeclExceptionMsg(UnknownEinsteinDiffusionModel,
                   "Invalid Einstein diffusion model");

  enum EinsteinDiffusionModel
  {
    M0, // D_n = 0
    M1  // D_n = Kb * T / q * mu_n
  };
} // namespace Ddhdg
