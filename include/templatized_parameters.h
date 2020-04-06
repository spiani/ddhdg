#pragma once

namespace Ddhdg
{
  template <bool V_enabled, bool n_enabled, bool p_enabled>
  struct TemplatizedParameters
  {
    static const bool is_V_enabled = V_enabled;
    static const bool is_n_enabled = n_enabled;
    static const bool is_p_enabled = p_enabled;
  };

} // namespace Ddhdg
