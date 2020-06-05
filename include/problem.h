#pragma once
#include <deal.II/base/function.h>

#include <deal.II/grid/tria.h>

#include "boundary_conditions.h"
#include "electron_mobility.h"
#include "permittivity.h"
#include "recombination_term.h"

namespace Ddhdg
{
  template <int dim, class Permittivity>
  struct Problem
  {
    Problem(
      std::shared_ptr<const dealii::Triangulation<dim>>    triangulation,
      std::shared_ptr<const Permittivity>                  permittivity,
      std::shared_ptr<const ElectronMobility<dim>>         n_electron_mobility,
      std::shared_ptr<const ElectronMobility<dim>>         p_electron_mobility,
      std::shared_ptr<const RecombinationTerm<dim>>        recombination_term,
      std::shared_ptr<const dealii::Function<dim>>         temperature,
      std::shared_ptr<const dealii::Function<dim>>         doping,
      std::shared_ptr<const BoundaryConditionHandler<dim>> boundary_handler,
      double conduction_band_density,
      double valence_band_density,
      double conduction_band_edge_energy = 0.,
      double valence_band_edge_energy    = 0.);


    const std::shared_ptr<const dealii::Triangulation<dim>> triangulation;
    const std::shared_ptr<const Permittivity>               permittivity;
    const std::shared_ptr<const ElectronMobility<dim>>      n_electron_mobility;
    const std::shared_ptr<const ElectronMobility<dim>>      p_electron_mobility;
    const std::shared_ptr<const RecombinationTerm<dim>>     recombination_term;
    const std::shared_ptr<const dealii::Function<dim>>      temperature;
    const std::shared_ptr<const dealii::Function<dim>>      doping;
    const std::shared_ptr<const BoundaryConditionHandler<dim>> boundary_handler;

    const std::map<Component, double> band_density;
    const std::map<Component, double> band_edge_energy;
  };

} // namespace Ddhdg
