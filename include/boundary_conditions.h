#pragma once

#include <deal.II/base/function.h>

#include "components.h"

namespace Ddhdg
{
    template<int dim>
    using boundary_function_map = std::map<dealii::types::boundary_id, const dealii::Function<dim> *>;

    template<int dim>
    using boundary_maps = std::map<Ddhdg::Component, boundary_function_map<dim>>;

    enum BoundaryConditionType {
        dirichlet, neumann, robin
    };

    template<int dim>
    class BoundaryCondition {
     public:
      explicit BoundaryCondition (BoundaryConditionType bc_type_, Component c)
          : bc_type (bc_type_), component (c)
      {}

      BoundaryConditionType get_type () const
      { return bc_type; }

      Component get_component () const
      { return component; }

     protected:
      const BoundaryConditionType bc_type;
      const Component component;
    };

    template<int dim>
    class DirichletBoundaryCondition : public BoundaryCondition<dim> {
     public:
      explicit DirichletBoundaryCondition (std::shared_ptr<const dealii::Function<dim>> f, Component c)
          : BoundaryCondition<dim> (dirichlet, c), function (f)
      {}

      DirichletBoundaryCondition (const DirichletBoundaryCondition &dbc)
          : BoundaryCondition<dim> (dirichlet, dbc.get_component ()), function (dbc.get_function ())
      {}

      std::shared_ptr<const dealii::Function<dim>> get_function () const
      { return function; }

     protected:
      const std::shared_ptr<const dealii::Function<dim>> function;
    };

    template<int dim>
    class NeumannBoundaryCondition : public BoundaryCondition<dim> {
     public:
      explicit NeumannBoundaryCondition (std::shared_ptr<const dealii::Function<dim>> f, Component c)
          : BoundaryCondition<dim> (neumann, c), function (f)
      {}

      NeumannBoundaryCondition (const NeumannBoundaryCondition &dbc)
          : BoundaryCondition<dim> (neumann, dbc.get_component ()), function (dbc.get_function ())
      {}

      std::shared_ptr<const dealii::Function<dim>> get_function () const
      { return function; }

     protected:
      const std::shared_ptr<const dealii::Function<dim>> function;
    };

    template<int dim>
    class BoundaryConditionHandler {
     public:
      void add_boundary_condition (dealii::types::boundary_id id, const DirichletBoundaryCondition<dim> &db)
      { dbc_map.insert ({id, db}); }

      void add_boundary_condition (dealii::types::boundary_id id, const NeumannBoundaryCondition<dim> &db)
      { nbc_map.insert ({id, db}); }

      [[nodiscard]] bool has_dirichlet_boundary_conditions () const
      { return !dbc_map.empty (); }

      [[nodiscard]] bool has_neumann_boundary_conditions () const
      { return !nbc_map.empty (); }

      boundary_maps<dim> get_dirichlet_boundary_maps () const;

     protected:
      std::map<dealii::types::boundary_id, const DirichletBoundaryCondition<dim>> dbc_map;
      std::map<dealii::types::boundary_id, const NeumannBoundaryCondition<dim>> nbc_map;
    };

} // end of namespace Ddhdg
