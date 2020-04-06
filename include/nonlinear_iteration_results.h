#pragma once

namespace Ddhdg
{
  struct NonlinearIterationResults
  {
    const bool         converged;
    const unsigned int iterations;
    const double       last_update_norm;

    NonlinearIterationResults(bool         converged,
                              unsigned int iterations,
                              double       last_update_norm);
  };
} // namespace Ddhdg
