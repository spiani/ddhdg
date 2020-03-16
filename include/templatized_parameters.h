#pragma once

#include "einstein_diffusion_model.h"

namespace Ddhdg
{
  template <EinsteinDiffusionModel m>
  struct TemplatizedParameters
  {
    static const EinsteinDiffusionModel einstein_diffusion_model = m;
  };

} // namespace Ddhdg
