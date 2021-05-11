#include "problem.h"

namespace Ddhdg
{
  template <int dim, class Permittivity, class NMobility, class PMobility>
  Problem<dim, Permittivity, NMobility, PMobility>::Problem(
    const std::shared_ptr<const dealii::Triangulation<dim>> triangulation,
    const std::shared_ptr<const Permittivity>               permittivity,
    const std::shared_ptr<const NMobility>                  n_mobility,
    const std::shared_ptr<const PMobility>                  p_mobility,
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
    , n_mobility(n_mobility)
    , p_mobility(p_mobility)
    , recombination_term(recombination_term)
    , temperature(temperature)
    , doping(doping)
    , boundary_handler(boundary_handler)
    , band_density{{Component::n, conduction_band_density},
                   {Component::p, valence_band_density}}
    , band_edge_energy{{Component::n, conduction_band_edge_energy},
                       {Component::p, valence_band_edge_energy}}
  {}

  template struct Ddhdg::Problem<1,
                                 Ddhdg::HomogeneousPermittivity<1>,
                                 Ddhdg::HomogeneousMobility<1>,
                                 Ddhdg::HomogeneousMobility<1>>;
  template struct Ddhdg::Problem<2,
                                 Ddhdg::HomogeneousPermittivity<2>,
                                 Ddhdg::HomogeneousMobility<2>,
                                 Ddhdg::HomogeneousMobility<2>>;
  template struct Ddhdg::Problem<3,
                                 Ddhdg::HomogeneousPermittivity<3>,
                                 Ddhdg::HomogeneousMobility<3>,
                                 Ddhdg::HomogeneousMobility<3>>;
} // namespace Ddhdg
