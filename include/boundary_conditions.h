#pragma once

#include <deal.II/base/function.h>

#include "components.h"

namespace Ddhdg
{
  enum BoundaryConditionType
  {
    dirichlet,
    neumann,
    robin
  };

  template <int dim>
  class BoundaryCondition
  {
  public:
    BoundaryCondition(BoundaryConditionType bc_type_, Component c)
      : bc_type(bc_type_)
      , component(c)
    {}

    BoundaryCondition(const BoundaryCondition &other) = default;

    BoundaryConditionType
    get_type() const
    {
      return bc_type;
    }

    Component
    get_component() const
    {
      return component;
    }

  protected:
    const BoundaryConditionType bc_type;
    const Component             component;
  };

  template <int dim>
  class DirichletBoundaryCondition : public BoundaryCondition<dim>
  {
  public:
    DirichletBoundaryCondition(std::shared_ptr<const dealii::Function<dim>> f,
                               Component                                    c)
      : BoundaryCondition<dim>(dirichlet, c)
      , function(f)
    {}

    DirichletBoundaryCondition(const DirichletBoundaryCondition &dbc)
      : BoundaryCondition<dim>(dirichlet, dbc.get_component())
      , function(dbc.get_function())
    {}

    std::shared_ptr<const dealii::Function<dim>>
    get_function() const
    {
      return function;
    }

    double
    evaluate(const dealii::Point<dim> p) const
    {
      return function->value(p);
    }

  protected:
    std::shared_ptr<const dealii::Function<dim>> function;
  };

  template <int dim>
  class NeumannBoundaryCondition : public BoundaryCondition<dim>
  {
  public:
    NeumannBoundaryCondition(std::shared_ptr<const dealii::Function<dim>> f,
                             Component                                    c)
      : BoundaryCondition<dim>(neumann, c)
      , function(f)
    {}

    NeumannBoundaryCondition(const NeumannBoundaryCondition &dbc)
      : BoundaryCondition<dim>(neumann, dbc.get_component())
      , function(dbc.get_function())
    {}

    std::shared_ptr<const dealii::Function<dim>>
    get_function() const
    {
      return function;
    }

    double
    evaluate(const dealii::Point<dim> p) const
    {
      return function->value(p);
    }

  protected:
    std::shared_ptr<const dealii::Function<dim>> function;
  };

  template <int dim>
  using dirichlet_boundary_map =
    std::map<Component, const DirichletBoundaryCondition<dim>>;

  template <int dim>
  using dirichlet_boundary_id_map =
    std::map<dealii::types::boundary_id, dirichlet_boundary_map<dim>>;

  template <int dim>
  using neumann_boundary_map =
    std::map<Component, const NeumannBoundaryCondition<dim>>;

  template <int dim>
  using neumann_boundary_id_map =
    std::map<dealii::types::boundary_id, neumann_boundary_map<dim>>;

  DeclExceptionMsg(RepeatedBoundaryCondition,
                   "Trying to overwrite a boundary condition");

  template <int dim>
  class BoundaryConditionHandler
  {
  public:
    void
    add_boundary_condition(dealii::types::boundary_id             id,
                           const DirichletBoundaryCondition<dim> &db);

    bool
    replace_boundary_condition(dealii::types::boundary_id             id,
                               const DirichletBoundaryCondition<dim> &db);

    void
    add_boundary_condition(dealii::types::boundary_id           id,
                           const NeumannBoundaryCondition<dim> &db);

    bool
    replace_boundary_condition(dealii::types::boundary_id           id,
                               const NeumannBoundaryCondition<dim> &db);

    void
    add_boundary_condition(dealii::types::boundary_id                   id,
                           BoundaryConditionType                        bc_type,
                           Component                                    c,
                           std::shared_ptr<const dealii::Function<dim>> f);

    bool
    replace_boundary_condition(dealii::types::boundary_id id,
                               BoundaryConditionType      bc_type,
                               Component                  c,
                               std::shared_ptr<const dealii::Function<dim>> f);

    [[nodiscard]] bool
    has_dirichlet_boundary_conditions() const;

    [[nodiscard]] bool
    has_dirichlet_boundary_conditions(dealii::types::boundary_id id) const;

    [[nodiscard]] bool
    has_dirichlet_boundary_conditions(dealii::types::boundary_id id,
                                      Component                  c) const;

    [[nodiscard]] bool
    has_neumann_boundary_conditions() const;

    [[nodiscard]] bool
    has_neumann_boundary_conditions(dealii::types::boundary_id id) const;

    [[nodiscard]] bool
    has_neumann_boundary_conditions(dealii::types::boundary_id id,
                                    Component                  c) const;

    dirichlet_boundary_map<dim>
    get_dirichlet_conditions_for_id(dealii::types::boundary_id id) const;

    DirichletBoundaryCondition<dim>
    get_dirichlet_conditions_for_id(dealii::types::boundary_id id,
                                    Component                  c) const;

    neumann_boundary_map<dim>
    get_neumann_conditions_for_id(dealii::types::boundary_id id) const;

    NeumannBoundaryCondition<dim>
    get_neumann_conditions_for_id(dealii::types::boundary_id id,
                                  Component                  c) const;

    [[nodiscard]] bool
    has_boundary_conditions() const;

    [[nodiscard]] bool
    has_boundary_conditions(dealii::types::boundary_id id) const;

  protected:
    dirichlet_boundary_id_map<dim> dbc_map;
    neumann_boundary_id_map<dim>   nbc_map;
  };

} // end of namespace Ddhdg
