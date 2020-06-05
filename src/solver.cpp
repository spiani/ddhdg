#include "solver.h"

#include <adimensionalizer.h>

namespace Ddhdg
{
  template <int dim, class Permittivity>
  Solver<dim, Permittivity>::Solver(
    std::shared_ptr<const Problem<dim, Permittivity>> problem,
    std::shared_ptr<const Adimensionalizer>           adimensionalizer)
    : problem(problem)
    , adimensionalizer(adimensionalizer)
  {}



  template <int dim, class Permittivity>
  void
  Solver<dim, Permittivity>::refine_grid_once()
  {
    return this->refine_grid(1);
  }



  template <int dim, class Permittivity>
  void
  Solver<dim, Permittivity>::set_component(
    Component                                    c,
    std::shared_ptr<const dealii::Function<dim>> c_function,
    bool                                         use_projection)
  {
    if (use_projection)
      return this->project_component(c, c_function);
    else
      return this->interpolate_component(c, c_function);
  }



  template <int dim, class Permittivity>
  NonlinearIterationResults
  Solver<dim, Permittivity>::compute_thermodynamic_equilibrium(
    double absolute_tol,
    double relative_tol,
    int    max_number_of_iterations)
  {
    return this->compute_thermodynamic_equilibrium(absolute_tol,
                                                   relative_tol,
                                                   max_number_of_iterations,
                                                   true);
  }



  template <int dim, class Permittivity>
  NonlinearIterationResults
  Solver<dim, Permittivity>::compute_thermodynamic_equilibrium()
  {
    return this->compute_thermodynamic_equilibrium(true);
  }



  template <int dim, class Permittivity>
  void
  Solver<dim, Permittivity>::output_results(
    const std::string &solution_filename) const
  {
    return this->output_results(solution_filename, false);
  }



  template <int dim, class Permittivity>
  void
  Solver<dim, Permittivity>::output_results(
    const std::string &solution_filename,
    const std::string &trace_filename) const
  {
    return this->output_results(solution_filename, trace_filename, false);
  }



  template class Solver<1, HomogeneousPermittivity<1>>;
  template class Solver<2, HomogeneousPermittivity<2>>;
  template class Solver<3, HomogeneousPermittivity<3>>;
} // namespace Ddhdg
