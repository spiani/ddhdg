#include "boundary_conditions.h"

#include <string>

template <int dim>
void
Ddhdg::BoundaryConditionHandler<dim>::add_boundary_condition(
  dealii::types::boundary_id                    id,
  const Ddhdg::DirichletBoundaryCondition<dim> &db)
{
  if (dbc_map[id].find(db.get_component()) != dbc_map[id].end())
    {
      std::string error =
        "On boundary id " + std::to_string(id) + ", two " +
        "Dirichlet boundary conditions have been specified for the " +
        "same component";
      AssertThrow(false, RepeatedBoundaryCondition(error));
    }

  dbc_map[id].insert({db.get_component(), db});
}

template <int dim>
void
Ddhdg::BoundaryConditionHandler<dim>::add_boundary_condition(
  dealii::types::boundary_id                  id,
  const Ddhdg::NeumannBoundaryCondition<dim> &db)
{
  if (nbc_map[id].find(db.get_component()) != nbc_map[id].end())
    {
      std::string error =
        "On boundary id " + std::to_string(id) + ", two " +
        "Neumann boundary conditions have been specified for the " +
        "same component";
      AssertThrow(false, RepeatedBoundaryCondition(error));
    }

  nbc_map[id].insert({db.get_component(), db});
}

template <int dim>
void
Ddhdg::BoundaryConditionHandler<dim>::add_boundary_condition(
  dealii::types::boundary_id                   id,
  BoundaryConditionType                        bc_type,
  Component                                    c,
  std::shared_ptr<const dealii::Function<dim>> f)
{
  if (bc_type == Ddhdg::dirichlet)
    {
      this->add_boundary_condition(id, DirichletBoundaryCondition(f, c));
    }
  else if (bc_type == Ddhdg::neumann)
    {
      this->add_boundary_condition(id, NeumannBoundaryCondition(f, c));
    }
}

template <int dim>
bool
Ddhdg::BoundaryConditionHandler<dim>::has_dirichlet_boundary_conditions() const
{
  return !dbc_map.empty();
}


template <int dim>
bool
Ddhdg::BoundaryConditionHandler<dim>::has_dirichlet_boundary_conditions(
  dealii::types::boundary_id id) const
{
  return dbc_map.find(id) != dbc_map.end();
}

template <int dim>
bool
Ddhdg::BoundaryConditionHandler<dim>::has_dirichlet_boundary_conditions(
  dealii::types::boundary_id id,
  Component                  c) const
{
  if (dbc_map.find(id) == dbc_map.end())
    {
      return false;
    }
  dirichlet_boundary_map<dim> boundary_map = this->dbc_map.at(id);
  return boundary_map.find(c) != boundary_map.end();
}

template <int dim>
bool
Ddhdg::BoundaryConditionHandler<dim>::has_neumann_boundary_conditions() const
{
  return !nbc_map.empty();
}

template <int dim>
bool
Ddhdg::BoundaryConditionHandler<dim>::has_neumann_boundary_conditions(
  dealii::types::boundary_id id) const
{
  return nbc_map.find(id) != nbc_map.end();
}

template <int dim>
bool
Ddhdg::BoundaryConditionHandler<dim>::has_neumann_boundary_conditions(
  dealii::types::boundary_id id,
  Component                  c) const
{
  if (nbc_map.find(id) == nbc_map.end())
    {
      return false;
    }
  neumann_boundary_map<dim> boundary_map = this->nbc_map.at(id);
  return boundary_map.find(c) != boundary_map.end();
}

template <int dim>
Ddhdg::dirichlet_boundary_map<dim>
Ddhdg::BoundaryConditionHandler<dim>::get_dirichlet_conditions_for_id(
  dealii::types::boundary_id id) const
{
  return dbc_map.at(id);
}

template <int dim>
Ddhdg::DirichletBoundaryCondition<dim>
Ddhdg::BoundaryConditionHandler<dim>::get_dirichlet_conditions_for_id(
  dealii::types::boundary_id id, Ddhdg::Component c) const
{
  auto boundary_map = dbc_map.at(id);
  return boundary_map.find(c)->second;

}

template <int dim>
Ddhdg::neumann_boundary_map<dim>
Ddhdg::BoundaryConditionHandler<dim>::get_neumann_conditions_for_id(
  dealii::types::boundary_id id) const
{
  return nbc_map.at(id);
}

template <int dim>
Ddhdg::NeumannBoundaryCondition<dim>
Ddhdg::BoundaryConditionHandler<dim>::get_neumann_conditions_for_id(
  dealii::types::boundary_id id, Ddhdg::Component c) const
{
  auto boundary_map = nbc_map.at(id);
  return boundary_map.find(c)->second;

}

template class Ddhdg::BoundaryConditionHandler<1>;

template class Ddhdg::BoundaryConditionHandler<2>;

template class Ddhdg::BoundaryConditionHandler<3>;
