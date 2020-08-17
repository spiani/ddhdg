#include "assembling_methods.template.h"

namespace Ddhdg
{
  template class NPSolver<2, HomogeneousPermittivity<2>>;
  template void
  NPSolver<2, HomogeneousPermittivity<2>>::assemble_system<false>(
    bool reconstruct_trace,
    bool compute_thermodynamic_equilibrium);
  template void
  NPSolver<2, HomogeneousPermittivity<2>>::assemble_system<true>(
    bool reconstruct_trace,
    bool compute_thermodynamic_equilibrium);
} // namespace Ddhdg
