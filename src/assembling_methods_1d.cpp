#include "assembling_methods.template.h"

namespace Ddhdg
{
  template class NPSolver<1, HomogeneousProblem<1>>;
  template void
  NPSolver<1, HomogeneousProblem<1>>::assemble_system<false>(
    bool reconstruct_trace,
    bool compute_thermodynamic_equilibrium);
  template void
  NPSolver<1, HomogeneousProblem<1>>::assemble_system<true>(
    bool reconstruct_trace,
    bool compute_thermodynamic_equilibrium);
} // namespace Ddhdg
