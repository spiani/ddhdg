#include "problem.h"

namespace Ddhdg
{
  template <int dim>
  Problem<dim>::Problem(
    const std::shared_ptr<const dealii::Triangulation<dim>> triangulation,
    const std::shared_ptr<const Permittivity<dim>>          permittivity,
    const std::shared_ptr<const ElectronMobility<dim>>      n_electron_mobility,
    const std::shared_ptr<const ElectronMobility<dim>>      p_electron_mobility,
    const std::shared_ptr<const RecombinationTerm<dim>>     recombination_term,
    const std::shared_ptr<const dealii::Function<dim>>      temperature,
    const std::shared_ptr<const dealii::Function<dim>>      doping,
    const std::shared_ptr<const BoundaryConditionHandler<dim>> boundary_handler,
    const double conduction_band_density,
    const double valence_band_density,
    const double conduction_band_edge_energy,
    const double valence_band_edge_energy)
    : triangulation(triangulation)
    , permittivity(permittivity)
    , n_electron_mobility(n_electron_mobility)
    , p_electron_mobility(p_electron_mobility)
    , recombination_term(recombination_term)
    , temperature(temperature)
    , doping(doping)
    , boundary_handler(boundary_handler)
    , band_density{{Component::n, conduction_band_density},
                   {Component::p, valence_band_density}}
    , band_edge_energy{{Component::n, conduction_band_edge_energy},
                       {Component::p, valence_band_edge_energy}}
  {}

  template struct Problem<1>;
  template struct Problem<2>;
  template struct Problem<3>;
} // namespace Ddhdg
