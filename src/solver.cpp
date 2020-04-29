#include "solver.h"

#include <adimensionalizer.h>

namespace Ddhdg
{
  template <int dim>
  Solver<dim>::Solver(std::shared_ptr<const Problem<dim>>     problem,
                      std::shared_ptr<const Adimensionalizer> adimensionalizer)
    : problem(problem)
    , adimensionalizer(adimensionalizer)
  {}



  template <int dim>
  void
  Solver<dim>::refine_grid_once()
  {
    return this->refine_grid(1);
  }



  template <int dim>
  void
  Solver<dim>::set_component(
    Component                                    c,
    std::shared_ptr<const dealii::Function<dim>> c_function,
    bool                                         use_projection)
  {
    if (use_projection)
      return this->project_component(c, c_function);
    else
      return this->interpolate_component(c, c_function);
  }



  template <int dim>
  void
  Solver<dim>::output_results(const std::string &solution_filename) const
  {
    return this->output_results(solution_filename, false);
  }



  template <int dim>
  void
  Solver<dim>::output_results(const std::string &solution_filename,
                              const std::string &trace_filename) const
  {
    return this->output_results(solution_filename, trace_filename, false);
  }



  template class Solver<1>;
  template class Solver<2>;
  template class Solver<3>;
} // namespace Ddhdg
