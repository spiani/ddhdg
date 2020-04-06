#pragma once
#include <deal.II/base/function.h>

#include <deal.II/grid/tria.h>

#include "boundary_conditions.h"
#include "electron_mobility.h"
#include "permittivity.h"
#include "recombination_term.h"

namespace Ddhdg
{
  template <int dim>
  struct Problem
  {
    Problem(
      std::shared_ptr<const dealii::Triangulation<dim>>    triangulation,
      std::shared_ptr<const Permittivity<dim>>             permittivity,
      std::shared_ptr<const ElectronMobility<dim>>         n_electron_mobility,
      std::shared_ptr<const RecombinationTerm<dim>>        n_recombination_term,
      std::shared_ptr<const ElectronMobility<dim>>         p_electron_mobility,
      std::shared_ptr<const RecombinationTerm<dim>>        p_recombination_term,
      std::shared_ptr<const dealii::Function<dim>>         temperature,
      std::shared_ptr<const dealii::Function<dim>>         doping,
      std::shared_ptr<const BoundaryConditionHandler<dim>> boundary_handler);


    const std::shared_ptr<const dealii::Triangulation<dim>> triangulation;
    const std::shared_ptr<const Permittivity<dim>>          permittivity;
    const std::shared_ptr<const ElectronMobility<dim>>      n_electron_mobility;
    const std::shared_ptr<const RecombinationTerm<dim>> n_recombination_term;
    const std::shared_ptr<const ElectronMobility<dim>>  p_electron_mobility;
    const std::shared_ptr<const RecombinationTerm<dim>> p_recombination_term;
    const std::shared_ptr<const dealii::Function<dim>>  temperature;
    const std::shared_ptr<const dealii::Function<dim>>  doping;
    const std::shared_ptr<const BoundaryConditionHandler<dim>> boundary_handler;
  };

} // namespace Ddhdg
