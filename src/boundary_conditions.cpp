#include "boundary_conditions.h"

template<int dim>
Ddhdg::boundary_maps<dim>
Ddhdg::BoundaryConditionHandler<dim>::get_dirichlet_boundary_maps () const
{
  std::set<Ddhdg::Component> components;
  for (auto const &x: dbc_map)
    {
      components.insert (x.second.get_component());
    }

  std::map<Ddhdg::Component, boundary_function_map<dim>> boundary_maps;
  for (auto const &component: components)
    {
      boundary_function_map<dim> bfm;
      boundary_maps.insert({component, bfm});
    }
  for (auto const &x: dbc_map)
    {
      dealii::types::boundary_id current_id = x.first;
      const dealii::Function<dim> *current_f = &(*(x.second.get_function ()));
      auto & component_boundary_map = boundary_maps[x.second.get_component()];
      component_boundary_map.insert ({current_id, current_f});
    }

  return boundary_maps;
}

template class Ddhdg::BoundaryConditionHandler<1>;
template class Ddhdg::BoundaryConditionHandler<2>;
template class Ddhdg::BoundaryConditionHandler<3>;
