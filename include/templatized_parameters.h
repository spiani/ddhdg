#pragma once

#include "einstein_diffusion_model.h"

namespace Ddhdg
{
  template <bool                   V_enabled,
            bool                   n_enabled,
            bool                   p_enabled,
            EinsteinDiffusionModel m>
  struct TemplatizedParameters
  {
    static const bool is_V_enabled = V_enabled;
    static const bool is_n_enabled = n_enabled;
    static const bool is_p_enabled = p_enabled;
    static const EinsteinDiffusionModel einstein_diffusion_model = m;
  };

} // namespace Ddhdg
