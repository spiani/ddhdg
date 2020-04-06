#include "problem.h"

namespace Ddhdg
{
  template <int dim>
  Problem<dim>::Problem(
    const std::shared_ptr<const dealii::Triangulation<dim>> triangulation,
    const std::shared_ptr<const Permittivity<dim>>          permittivity,
    const std::shared_ptr<const ElectronMobility<dim>>      n_electron_mobility,
    const std::shared_ptr<const RecombinationTerm<dim>> n_recombination_term,
    const std::shared_ptr<const ElectronMobility<dim>>  p_electron_mobility,
    const std::shared_ptr<const RecombinationTerm<dim>> p_recombination_term,
    const std::shared_ptr<const dealii::Function<dim>>  temperature,
    const std::shared_ptr<const dealii::Function<dim>>  doping,
    const std::shared_ptr<const BoundaryConditionHandler<dim>> boundary_handler)
    : triangulation(triangulation)
    , permittivity(permittivity)
    , n_electron_mobility(n_electron_mobility)
    , n_recombination_term(n_recombination_term)
    , p_electron_mobility(p_electron_mobility)
    , p_recombination_term(p_recombination_term)
    , temperature(temperature)
    , doping(doping)
    , boundary_handler(boundary_handler)
  {}

  template struct Problem<1>;
  template struct Problem<2>;
  template struct Problem<3>;
} // namespace Ddhdg
