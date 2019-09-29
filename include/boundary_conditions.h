#pragma once

#include <deal.II/base/function.h>

namespace Ddhdg
{

    enum BoundaryConditionType {
        dirichlet, neumann, robin
    };

    template<int dim>
    class BoundaryCondition {
     public:
      explicit BoundaryCondition (BoundaryConditionType bc_type_)
          : bc_type (bc_type_)
      {}

      BoundaryConditionType get_type () const
      { return bc_type; }

     protected:
      const BoundaryConditionType bc_type;
    };

    template<int dim>
    class DirichletBoundaryCondition : public BoundaryCondition<dim> {
     public:
      explicit DirichletBoundaryCondition (std::shared_ptr<const dealii::Function<dim>> f)
          : BoundaryCondition<dim> (dirichlet), function (f)
      {}

      DirichletBoundaryCondition (const DirichletBoundaryCondition &dbc)
          : BoundaryCondition<dim> (dirichlet), function (dbc.get_function ())
      {}

      std::shared_ptr<const dealii::Function<dim>> get_function () const
      { return function; }

     protected:
      const std::shared_ptr<const dealii::Function<dim>> function;
    };

    template<int dim>
    class NeumannBoundaryCondition : public BoundaryCondition<dim> {
     public:
      explicit NeumannBoundaryCondition (std::shared_ptr<const dealii::Function<dim>> f)
          : BoundaryCondition<dim> (neumann), function (f)
      {}

      NeumannBoundaryCondition (const NeumannBoundaryCondition &dbc)
          : BoundaryCondition<dim> (neumann), function (dbc.get_function ())
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

      std::map<dealii::types::boundary_id, const dealii::Function<dim> *> get_dirichlet_boundary_map () const
      {
        std::map<dealii::types::boundary_id, const dealii::Function<dim> *> boundary_map;
        for (auto const &x : dbc_map)
          {
            dealii::types::boundary_id current_id = x.first;
            const dealii::Function<dim> *current_f = &(*(x.second.get_function ()));
            boundary_map.insert ({current_id, current_f});
          }

        return boundary_map;
      }

     protected:
      std::map<dealii::types::boundary_id, const DirichletBoundaryCondition<dim>> dbc_map;
      std::map<dealii::types::boundary_id, const NeumannBoundaryCondition<dim>> nbc_map;
    };

} // end of namespace Ddhdg
