#include "nonlinear_iteration_results.h"

namespace Ddhdg
{
  NonlinearIterationResults::NonlinearIterationResults(
    const bool         converged,
    const unsigned int iterations,
    const double       last_update_norm)
    : converged(converged)
    , iterations(iterations)
    , last_update_norm(last_update_norm)
  {}
} // namespace Ddhdg
