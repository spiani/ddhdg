#include "solver.h"

#include <adimensionalizer.h>

namespace Ddhdg
{
  template <int dim, typename ProblemType>
  Solver<dim, ProblemType>::Solver(
    std::shared_ptr<const ProblemType>      problem,
    std::shared_ptr<const Adimensionalizer> adimensionalizer)
    : problem(problem)
    , adimensionalizer(adimensionalizer)
  {}



  template <int dim, typename ProblemType>
  void
  Solver<dim, ProblemType>::refine_grid_once(const bool preserve_solution)
  {
    return this->refine_grid(1, preserve_solution);
  }



  template <int dim, typename ProblemType>
  void
  Solver<dim, ProblemType>::set_component(
    Component                                    c,
    std::shared_ptr<const dealii::Function<dim>> c_function,
    bool                                         use_projection)
  {
    if (use_projection)
      return this->project_component(c, c_function);
    else
      return this->interpolate_component(c, c_function);
  }



  template <int dim, typename ProblemType>
  NonlinearIterationResults
  Solver<dim, ProblemType>::compute_thermodynamic_equilibrium(
    double absolute_tol,
    double relative_tol,
    int    max_number_of_iterations)
  {
    return this->compute_thermodynamic_equilibrium(absolute_tol,
                                                   relative_tol,
                                                   max_number_of_iterations,
                                                   true);
  }



  template <int dim, typename ProblemType>
  NonlinearIterationResults
  Solver<dim, ProblemType>::compute_thermodynamic_equilibrium()
  {
    return this->compute_thermodynamic_equilibrium(true);
  }



  template <int dim, typename ProblemType>
  void
  Solver<dim, ProblemType>::output_results(
    const std::string &solution_filename) const
  {
    return this->output_results(solution_filename, false);
  }



  template <int dim, typename ProblemType>
  void
  Solver<dim, ProblemType>::output_results(
    const std::string &solution_filename,
    const std::string &trace_filename) const
  {
    return this->output_results(solution_filename, trace_filename, false);
  }



  template class Solver<1, HomogeneousProblem<1>>;
  template class Solver<2, HomogeneousProblem<2>>;
  template class Solver<3, HomogeneousProblem<3>>;
} // namespace Ddhdg
