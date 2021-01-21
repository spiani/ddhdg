#include "np_solver.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>

#include <cmath>
#include <fstream>

#include "function_tools.h"

namespace Ddhdg
{
  NPSolverParameters::NPSolverParameters(
    const unsigned int V_degree,
    const unsigned int n_degree,
    const unsigned int p_degree,
    const double       nonlinear_solver_absolute_tolerance,
    const double       nonlinear_solver_relative_tolerance,
    const int          nonlinear_solver_max_number_of_iterations,
    const double       V_tau,
    const double       n_tau,
    const double       p_tau,
    const bool         iterative_linear_solver,
    const bool         multithreading,
    const DDFluxType   dd_flux_type,
    const bool         phi_linearize)
    : degree{{Component::V, V_degree},
             {Component::n, n_degree},
             {Component::p, p_degree}}
    , nonlinear_solver_absolute_tolerance(nonlinear_solver_absolute_tolerance)
    , nonlinear_solver_relative_tolerance(nonlinear_solver_relative_tolerance)
    , nonlinear_solver_max_number_of_iterations(
        nonlinear_solver_max_number_of_iterations)
    , tau{{Component::V, V_tau}, {Component::n, n_tau}, {Component::p, p_tau}}
    , iterative_linear_solver(iterative_linear_solver)
    , multithreading(multithreading)
    , dd_flux_type(dd_flux_type)
    , phi_linearize(phi_linearize)
  {}



  template <int dim, typename ProblemType>
  std::unique_ptr<dealii::Triangulation<dim>>
  NPSolver<dim, ProblemType>::copy_triangulation(
    const std::shared_ptr<const dealii::Triangulation<dim>> triangulation)
  {
    std::unique_ptr<dealii::Triangulation<dim>> new_triangulation =
      std::make_unique<dealii::Triangulation<dim>>();
    new_triangulation->copy_triangulation(*triangulation);
    return new_triangulation;
  }



  template <int dim, typename ProblemType>
  std::map<Component, std::vector<double>>
  NPSolver<dim, ProblemType>::ScratchData::initialize_double_map_on_components(
    const unsigned int n)
  {
    std::map<Component, std::vector<double>> m;
    for (const auto c : Ddhdg::all_primary_components())
      {
        m.insert({c, std::vector<double>(n)});
      }
    return m;
  }



  template <int dim, typename ProblemType>
  std::map<Component, std::vector<Tensor<1, dim>>>
  NPSolver<dim, ProblemType>::ScratchData::initialize_tensor_map_on_components(
    const unsigned int n)
  {
    std::map<Component, std::vector<Tensor<1, dim>>> m;
    for (const auto c : Ddhdg::all_primary_components())
      {
        m.insert({c, std::vector<Tensor<1, dim>>(n)});
      }
    return m;
  }



  template <int dim, typename ProblemType>
  std::map<Component, std::vector<double>>
  NPSolver<dim, ProblemType>::ScratchData::initialize_double_map_on_n_and_p(
    const unsigned int k)
  {
    std::map<Component, std::vector<double>> m;
    m.insert({Component::n, std::vector<double>(k)});
    m.insert({Component::p, std::vector<double>(k)});
    return m;
  }



  template <int dim, typename ProblemType>
  NPSolver<dim, ProblemType>::ScratchData::ScratchData(
    const FiniteElement<dim> & fe_trace_restricted,
    const FiniteElement<dim> & fe_trace,
    const FiniteElement<dim> & fe_cell,
    const QGauss<dim> &        quadrature_formula,
    const QGauss<dim - 1> &    face_quadrature_formula,
    const UpdateFlags          cell_flags,
    const UpdateFlags          cell_face_flags,
    const UpdateFlags          trace_flags,
    const UpdateFlags          trace_restricted_flags,
    const Permittivity &       permittivity,
    const NMobility &          n_mobility,
    const PMobility &          p_mobility,
    const std::set<Component> &enabled_components,
    const std::map<Component, const dealii::FiniteElement<dim> &> &fe_map)
    : fe_values_cell(fe_cell, quadrature_formula, cell_flags)
    , fe_face_values_cell(fe_cell, face_quadrature_formula, cell_face_flags)
    , fe_face_values_trace(fe_trace, face_quadrature_formula, trace_flags)
    , fe_face_values_trace_restricted(fe_trace_restricted,
                                      face_quadrature_formula,
                                      trace_restricted_flags)
    , enabled_component_indices(
        check_dofs_on_enabled_components(fe_cell, enabled_components))
    , fe_cell_support_on_face(
        check_dofs_on_faces_for_cells(fe_cell, enabled_component_indices))
    , fe_trace_support_on_face(
        check_dofs_on_faces_for_trace(fe_trace_restricted))
    , dofs_on_enabled_components(enabled_component_indices.size())
    , cc_matrix(dofs_on_enabled_components, dofs_on_enabled_components)
    , ct_matrix(dofs_on_enabled_components, fe_trace_restricted.dofs_per_cell)
    , tc_matrix(fe_trace_restricted.dofs_per_cell, dofs_on_enabled_components)
    , tmp_matrix(fe_trace_restricted.dofs_per_cell, dofs_on_enabled_components)
    , cc_rhs(dofs_on_enabled_components)
    , tmp_rhs(fe_cell.dofs_per_cell)
    , restricted_tmp_rhs(dofs_on_enabled_components)
    , permittivity(permittivity)
    , n_mobility(n_mobility)
    , p_mobility(p_mobility)
    , cell_quadrature_points(quadrature_formula.size())
    , face_quadrature_points(face_quadrature_formula.size())
    , T_cell(quadrature_formula.size())
    , T_face(face_quadrature_formula.size())
    , U_T_cell(quadrature_formula.size())
    , U_T_face(face_quadrature_formula.size())
    , doping_cell(quadrature_formula.size())
    , r_cell(quadrature_formula.size())
    , dr_cell(initialize_double_map_on_n_and_p(quadrature_formula.size()))
    , phi(initialize_double_map_on_n_and_p(quadrature_formula.size()))
    , previous_c_cell(
        initialize_double_map_on_components(quadrature_formula.size()))
    , previous_c_face(
        initialize_double_map_on_components(face_quadrature_formula.size()))
    , previous_d_cell(
        initialize_tensor_map_on_components(quadrature_formula.size()))
    , previous_d_face(
        initialize_tensor_map_on_components(face_quadrature_formula.size()))
    , previous_tr_c_face(
        initialize_double_map_on_components(face_quadrature_formula.size()))
    , d(initialize_tensor_map_on_components(fe_cell.dofs_per_cell))
    , d_div(initialize_double_map_on_components(fe_cell.dofs_per_cell))
    , c(initialize_double_map_on_components(fe_cell.dofs_per_cell))
    , c_grad(initialize_tensor_map_on_components(fe_cell.dofs_per_cell))
    , tr_c(
        initialize_double_map_on_components(fe_trace_restricted.dofs_per_cell))
    , tr_c_solution_values(
        initialize_double_map_on_components(face_quadrature_formula.size()))
    , local_condenser(fe_map)
  {}



  template <int dim, typename ProblemType>
  std::vector<unsigned int>
  NPSolver<dim, ProblemType>::ScratchData::check_dofs_on_enabled_components(
    const FiniteElement<dim> & fe_cell,
    const std::set<Component> &enabled_components)
  {
    const unsigned int dofs_per_cell = fe_cell.dofs_per_cell;

    std::vector<unsigned int> enabled_component_indices;

    std::set<unsigned int> component_indices;
    for (const Component c : enabled_components)
      component_indices.insert(get_component_index(c));

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const unsigned int current_index =
          fe_cell.system_to_block_index(i).first;
        if (component_indices.find(current_index) != component_indices.end())
          enabled_component_indices.push_back(i);
      }
    return enabled_component_indices;
  }



  template <int dim, typename ProblemType>
  std::vector<std::vector<unsigned int>>
  NPSolver<dim, ProblemType>::ScratchData::check_dofs_on_faces_for_cells(
    const FiniteElement<dim> &       fe_cell,
    const std::vector<unsigned int> &enabled_component_indices)
  {
    const unsigned int faces_per_cell     = GeometryInfo<dim>::faces_per_cell;
    const unsigned int dofs_per_component = enabled_component_indices.size();

    std::vector<std::vector<unsigned int>> fe_cell_support_on_face(
      faces_per_cell);

    for (unsigned int face = 0; face < faces_per_cell; ++face)
      for (unsigned int i = 0; i < dofs_per_component; ++i)
        {
          const unsigned int ii = enabled_component_indices[i];
          if (fe_cell.has_support_on_face(ii, face))
            fe_cell_support_on_face[face].push_back(i);
        }
    return fe_cell_support_on_face;
  }



  template <int dim, typename ProblemType>
  std::vector<std::vector<unsigned int>>
  NPSolver<dim, ProblemType>::ScratchData::check_dofs_on_faces_for_trace(
    const FiniteElement<dim> &fe_trace_restricted)
  {
    const unsigned int faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    const unsigned int dofs_per_cell  = fe_trace_restricted.dofs_per_cell;

    std::vector<std::vector<unsigned int>> fe_trace_support_on_face(
      faces_per_cell);

    for (unsigned int face = 0; face < faces_per_cell; ++face)
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          if (fe_trace_restricted.has_support_on_face(i, face))
            fe_trace_support_on_face[face].push_back(i);
        }

    return fe_trace_support_on_face;
  }



  template <int dim, typename ProblemType>
  NPSolver<dim, ProblemType>::ScratchData::ScratchData(const ScratchData &sd)
    : fe_values_cell(sd.fe_values_cell.get_fe(),
                     sd.fe_values_cell.get_quadrature(),
                     sd.fe_values_cell.get_update_flags())
    , fe_face_values_cell(sd.fe_face_values_cell.get_fe(),
                          sd.fe_face_values_cell.get_quadrature(),
                          sd.fe_face_values_cell.get_update_flags())
    , fe_face_values_trace(sd.fe_face_values_trace.get_fe(),
                           sd.fe_face_values_trace.get_quadrature(),
                           sd.fe_face_values_trace.get_update_flags())
    , fe_face_values_trace_restricted(
        sd.fe_face_values_trace_restricted.get_fe(),
        sd.fe_face_values_trace_restricted.get_quadrature(),
        sd.fe_face_values_trace_restricted.get_update_flags())
    , enabled_component_indices(sd.enabled_component_indices)
    , fe_cell_support_on_face(sd.fe_cell_support_on_face)
    , fe_trace_support_on_face(sd.fe_trace_support_on_face)
    , dofs_on_enabled_components(sd.dofs_on_enabled_components)
    , cc_matrix(sd.cc_matrix)
    , ct_matrix(sd.ct_matrix)
    , tc_matrix(sd.tc_matrix)
    , tmp_matrix(sd.tmp_matrix)
    , cc_rhs(sd.cc_rhs)
    , tmp_rhs(sd.tmp_rhs)
    , restricted_tmp_rhs(sd.restricted_tmp_rhs)
    , permittivity(sd.permittivity)
    , n_mobility(sd.n_mobility)
    , p_mobility(sd.p_mobility)
    , cell_quadrature_points(sd.cell_quadrature_points)
    , face_quadrature_points(sd.face_quadrature_points)
    , T_cell(sd.T_cell)
    , T_face(sd.T_face)
    , U_T_cell(sd.U_T_cell)
    , U_T_face(sd.U_T_face)
    , doping_cell(sd.doping_cell)
    , r_cell(sd.r_cell)
    , dr_cell(sd.dr_cell)
    , phi(sd.phi)
    , previous_c_cell(sd.previous_c_cell)
    , previous_c_face(sd.previous_c_face)
    , previous_d_cell(sd.previous_d_cell)
    , previous_d_face(sd.previous_d_face)
    , previous_tr_c_face(sd.previous_tr_c_face)
    , d(sd.d)
    , d_div(sd.d_div)
    , c(sd.c)
    , c_grad(sd.c_grad)
    , tr_c(sd.tr_c)
    , tr_c_solution_values(sd.tr_c_solution_values)
    , local_condenser(sd.local_condenser)
  {}



  template <int dim, typename ProblemType>
  FESystem<dim> const *
  NPSolver<dim, ProblemType>::generate_fe_system(
    const std::map<Component, unsigned int> &degree,
    const bool                               on_trace)
  {
    std::vector<const FiniteElement<dim> *> fe_systems;
    std::vector<unsigned int>               multiplicities;

    if (on_trace)
      for (std::pair<Component, unsigned int> c_degree : degree)
        {
          fe_systems.push_back(new FE_FaceQ<dim>(c_degree.second));
          multiplicities.push_back(1);
        }
    else
      for (std::pair<Component, unsigned int> c_degree : degree)
        {
          const unsigned int deg = c_degree.second;
          fe_systems.push_back(new FESystem<dim>(
            FESystem(FE_DGQ<dim>(deg), dim), 1, FE_DGQ<dim>(deg), 1));
          multiplicities.push_back(1);
        }

    Assert(fe_systems.size() > 0,
           dealii::ExcMessage(
             "Trying to build an empty fe_system. No component enabled!"))
      const auto output = new FESystem<dim>(fe_systems, multiplicities);

    for (const FiniteElement<dim> *fe : fe_systems)
      delete (fe);

    return output;
  }



  template <int dim, typename ProblemType>
  NPSolver<dim, ProblemType>::NPSolver(
    const std::shared_ptr<const ProblemType>        problem,
    const std::shared_ptr<const NPSolverParameters> parameters,
    const std::shared_ptr<const Adimensionalizer>   adimensionalizer,
    const bool                                      verbose)
    : Solver<dim, ProblemType>(problem, adimensionalizer)
    , triangulation(copy_triangulation(problem->triangulation))
    , parameters(std::make_unique<NPSolverParameters>(*parameters))
    , rescaled_doping(
        this->adimensionalizer->template adimensionalize_doping_function<dim>(
          this->problem->doping))
    , recombination_term(problem->recombination_term->copy())
    , fe_cell(std::unique_ptr<const dealii::FESystem<dim>>(
        generate_fe_system(parameters->degree, false)))
    , dof_handler_cell(*triangulation)
    , fe_trace(std::unique_ptr<const dealii::FESystem<dim>>(
        generate_fe_system(parameters->degree, true)))
    , fe_trace_restricted(std::unique_ptr<const dealii::FESystem<dim>>(
        generate_fe_system(parameters->degree, true)))
    , dof_handler_trace(*triangulation)
    , dof_handler_trace_restricted(*triangulation)
    , n_dirichlet_constraints(0)
  {
    if (!verbose)
      this->log_standard_level = Logging::severity_level::debug;
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::refine_grid(const unsigned int i,
                                          const bool         preserve_solution)
  {
    Assert(this->initialized || !preserve_solution, ExcNotInitialized());

    if (!preserve_solution)
      {
        this->triangulation->refine_global(i);
        this->initialized = false;
        return;
      }

    for (unsigned int k = 0; k < i; k++)
      {
        dealii::SolutionTransfer<dim> solution_transfer_cell(
          this->dof_handler_cell);

        this->triangulation->set_all_refine_flags();

        solution_transfer_cell.prepare_for_coarsening_and_refinement(
          this->current_solution_cell);

        this->triangulation->execute_coarsening_and_refinement();

        this->dof_handler_cell.distribute_dofs(*(this->fe_cell));
        this->dof_handler_trace.distribute_dofs(*(this->fe_trace));

        Vector<double> tmp_cell(this->dof_handler_cell.n_dofs());

        solution_transfer_cell.interpolate(this->current_solution_cell,
                                           tmp_cell);

        this->current_solution_cell = tmp_cell;
      }
    this->current_solution_trace.reinit(this->dof_handler_trace.n_dofs());

    this->global_constraints.clear();
    DoFTools::make_hanging_node_constraints(this->dof_handler_trace,
                                            this->global_constraints);
    this->global_constraints.close();

    this->project_cell_function_on_trace(
      all_primary_components(),
      TraceProjectionStrategy::reconstruct_problem_solution);

    this->update_cell.reinit(this->dof_handler_cell.n_dofs());
    this->update_trace.reinit(this->dof_handler_trace.n_dofs());
    this->setup_restricted_trace_system();
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::refine_and_coarsen_fixed_fraction(
    const Vector<float> &criteria,
    const double         top_fraction,
    const double         bottom_fraction,
    const unsigned int   max_n_cells)
  {
    if (!this->initialized)
      this->setup_overall_system();

    dealii::SolutionTransfer<dim> solution_transfer_cell(
      this->dof_handler_cell);

    dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
      *(this->triangulation),
      criteria,
      top_fraction,
      bottom_fraction,
      max_n_cells);

    this->triangulation->prepare_coarsening_and_refinement();

    solution_transfer_cell.prepare_for_coarsening_and_refinement(
      this->current_solution_cell);

    this->triangulation->execute_coarsening_and_refinement();

    this->dof_handler_cell.distribute_dofs(*(this->fe_cell));
    this->dof_handler_trace.distribute_dofs(*(this->fe_trace));

    Vector<double> tmp_cell(this->dof_handler_cell.n_dofs());

    solution_transfer_cell.interpolate(this->current_solution_cell, tmp_cell);

    this->current_solution_cell = tmp_cell;
    this->current_solution_trace.reinit(this->dof_handler_trace.n_dofs());

    this->global_constraints.clear();
    DoFTools::make_hanging_node_constraints(this->dof_handler_trace,
                                            this->global_constraints);
    this->global_constraints.close();

    this->project_cell_function_on_trace(
      all_primary_components(),
      TraceProjectionStrategy::reconstruct_problem_solution);

    this->update_cell.reinit(this->dof_handler_cell.n_dofs());
    this->update_trace.reinit(this->dof_handler_trace.n_dofs());
    this->setup_restricted_trace_system();
  }



  template <int dim, typename ProblemType>
  std::map<Component, unsigned int>
  NPSolver<dim, ProblemType>::restrict_degrees_on_enabled_component() const
  {
    std::map<Component, unsigned int> restricted_map;
    for (const auto k : this->parameters->degree)
      {
        if (this->is_enabled(k.first))
          restricted_map.insert(k);
      }
    return restricted_map;
  }



  template <int dim, typename ProblemType>
  unsigned int
  NPSolver<dim, ProblemType>::get_dofs_constrained_by_dirichlet_conditions(
    std::vector<bool> &lines) const
  {
    Assert(lines.size() == this->dof_handler_trace_restricted.n_dofs(),
           ExcMessage("Wrong size of lines vector"));

    const unsigned int dofs_per_cell = this->fe_trace_restricted->dofs_per_cell;
    const unsigned int faces_per_cell =
      dealii::GeometryInfo<dim>::faces_per_cell;

    std::map<Component, std::vector<std::vector<unsigned int>>> dofs_on_face;

    for (Component c : this->enabled_components)
      {
        const unsigned int c_index =
          get_component_index(c, this->enabled_components);
        std::vector<std::vector<unsigned int>> c_dofs_vector(faces_per_cell);
        for (unsigned int face = 0; face < faces_per_cell; ++face)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int current_index =
                  this->fe_trace_restricted->system_to_block_index(i).first;
                if (current_index == c_index &&
                    this->fe_trace_restricted->has_support_on_face(i, face))
                  c_dofs_vector[face].push_back(i);
              }
          }
        dofs_on_face.insert({c, c_dofs_vector});
      }

    bool                                 copied_global_dofs;
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
    unsigned int                         n_constrained_dofs = 0;

    for (const auto &cell :
         this->dof_handler_trace_restricted.active_cell_iterators())
      {
        copied_global_dofs = false;
        for (Component c : this->enabled_components)
          for (unsigned int face = 0; face < faces_per_cell; ++face)
            {
              const types::boundary_id face_boundary_id =
                cell->face(face)->boundary_id();
              const bool has_dirichlet_boundary_conditions =
                this->problem->boundary_handler
                  ->has_dirichlet_boundary_conditions(face_boundary_id, c);

              if (has_dirichlet_boundary_conditions)
                {
                  if (!copied_global_dofs)
                    {
                      cell->get_dof_indices(dof_indices);
                      copied_global_dofs = true;
                    }
                  const auto &c_face_dofs_on_face = dofs_on_face.at(c)[face];
                  for (unsigned int i : c_face_dofs_on_face)
                    lines[dof_indices[i]] = true;
                  n_constrained_dofs += c_face_dofs_on_face.size();
                }
            }
      }
    return n_constrained_dofs;
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::setup_overall_system()
  {
    this->write_log(
      "Building (or rebuilding) the vectors for storing the solution on the "
      "cells and on the skeleton (for all the components)");
    this->dof_handler_cell.distribute_dofs(*(this->fe_cell));
    this->dof_handler_trace.distribute_dofs(*(this->fe_trace));

    const unsigned int cell_dofs  = this->dof_handler_cell.n_dofs();
    const unsigned int trace_dofs = this->dof_handler_trace.n_dofs();

    this->write_log(
      "Number of degrees of freedom for the solution on the cells (all "
      "components): %s",
      cell_dofs);
    this->write_log(
      "Number of degrees of freedom for the solution on the skeleton (all "
      "components): %s",
      trace_dofs);

    this->current_solution_trace.reinit(trace_dofs);
    this->update_trace.reinit(trace_dofs);

    this->current_solution_cell.reinit(cell_dofs);
    this->update_cell.reinit(cell_dofs);

    this->global_constraints.clear();
    DoFTools::make_hanging_node_constraints(this->dof_handler_trace,
                                            this->global_constraints);
    this->global_constraints.close();

    this->setup_restricted_trace_system();

    this->initialized = true;
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::setup_restricted_trace_system()
  {
    this->write_log(
      "Building (or rebuilding) the vectors for storing the "
      "solution on the skeleton restricted only on the active components");
    this->fe_trace_restricted.reset(
      generate_fe_system(restrict_degrees_on_enabled_component(), true));

    this->dof_handler_trace_restricted.distribute_dofs(
      *(this->fe_trace_restricted));

    const unsigned int trace_restricted_dofs =
      this->dof_handler_trace_restricted.n_dofs();

    this->write_log(
      "Number of degrees of freedom for the solution on the skeleton for "
      "the active components: %s",
      trace_restricted_dofs);

    std::vector<bool> dirichlet_constraints(trace_restricted_dofs, false);
    this->n_dirichlet_constraints =
      this->get_dofs_constrained_by_dirichlet_conditions(dirichlet_constraints);

    this->n_dirichlet_constraints = n_dirichlet_constraints;
    this->constrained_dof_indices.resize(this->n_dirichlet_constraints);
    this->constrained_dof_values.resize(this->n_dirichlet_constraints);

    // This is tricky; we need as way to recognize if we have computed the
    // the values of the previous vectors (or if they are still just allocated)
    // For this reason, we put a "invalid flag" at the beginning of the vector
    // to make clear that they do not contain (yet) reasonable values. If there
    // are no dirichlet constraints this is useless because they contain
    // already the right values (which is no one)
    if (this->n_dirichlet_constraints > 0)
      this->constrained_dof_indices[0] = dealii::numbers::invalid_dof_index;

    this->write_log(
      "%s degree of freedom (on the skeleton) are constrained by Dirichlet "
      "boundary conditions",
      this->n_dirichlet_constraints);

    this->system_rhs.reinit(trace_restricted_dofs);
    this->system_solution.reinit(trace_restricted_dofs);

    this->constraints.clear();
    DoFTools::make_hanging_node_constraints(this->dof_handler_trace_restricted,
                                            this->constraints);
    this->constraints.add_lines(dirichlet_constraints);
    this->constraints.close();

    {
      DynamicSparsityPattern dsp(trace_restricted_dofs);
      DoFTools::make_sparsity_pattern(this->dof_handler_trace_restricted,
                                      dsp,
                                      this->constraints,
                                      false);
      this->sparsity_pattern.copy_from(dsp);
    }
    this->system_matrix.reinit(this->sparsity_pattern);

    this->build_restricted_to_trace_dof_map();
  }



  template <int dim, typename ProblemType>
  dealii::ComponentMask
  NPSolver<dim, ProblemType>::get_component_mask(
    const Component component) const
  {
    dealii::ComponentMask mask(3 * (dim + 1), false);
    switch (component)
      {
        case Component::V:
          mask.set(dim, true);
          break;
        case Component::n:
          mask.set(2 * dim + 1, true);
          break;
        case Component::p:
          mask.set(3 * dim + 2, true);
          break;
        default:
          Assert(false, InvalidComponent());
          break;
      }
    return mask;
  }



  template <int dim, typename ProblemType>
  dealii::ComponentMask
  NPSolver<dim, ProblemType>::get_component_mask(
    const Displacement displacement) const
  {
    dealii::ComponentMask mask(3 * (dim + 1), false);
    switch (displacement)
      {
        case Displacement::E:
          for (unsigned int i = 0; i < dim; i++)
            mask.set(i, true);
          break;
        case Displacement::Wn:
          for (unsigned int i = dim + 1; i < 2 * dim + 1; i++)
            mask.set(i, true);
          break;
        case Displacement::Wp:
          for (unsigned int i = 2 * dim + 2; i < 3 * dim + 2; i++)
            mask.set(i, true);
          break;
        default:
          Assert(false, InvalidDisplacement());
          break;
      }
    return mask;
  }



  template <int dim, typename ProblemType>
  dealii::ComponentMask
  NPSolver<dim, ProblemType>::get_trace_component_mask(
    const Component component,
    const bool      restricted) const
  {
    std::set<Component> components;
    if (restricted)
      components = this->enabled_components;
    else
      components = all_primary_components();

    const unsigned int    c_index = get_component_index(component, components);
    dealii::ComponentMask mask(components.size(), false);
    mask.set(c_index, true);
    return mask;
  }



  template <int dim, typename ProblemType>
  std::shared_ptr<dealii::Function<dim>>
  NPSolver<dim, ProblemType>::extend_function_on_all_components(
    const std::shared_ptr<const dealii::Function<dim>> f,
    const Component                                    c) const
  {
    Assert(f->n_components == 1, FunctionMustBeScalar());
    component_map<dim> f_components;
    switch (c)
      {
          case V: {
            f_components.insert({dim, f});
            break;
          }
          case n: {
            f_components.insert({2 * dim + 1, f});
            break;
          }
          case p: {
            f_components.insert({3 * dim + 2, f});
            break;
          }
        default:
          Assert(false, InvalidComponent());
          break;
      }
    std::shared_ptr<dealii::Function<dim>> function_extended =
      std::make_shared<FunctionByComponents<dim>>(3 * (dim + 1), f_components);
    return function_extended;
  }



  template <int dim, typename ProblemType>
  std::shared_ptr<dealii::Function<dim>>
  NPSolver<dim, ProblemType>::extend_function_on_all_components(
    const std::shared_ptr<const dealii::Function<dim>> f,
    const Displacement                                 d) const
  {
    Assert(f->n_components == dim,
           dealii::ExcDimensionMismatch(f->n_components, dim));
    component_map<dim> f_components;
    switch (d)
      {
          case E: {
            for (unsigned int i = 0; i < dim; i++)
              {
                f_components.insert(
                  {i, std::make_shared<const ComponentFunction<dim>>(f, i)});
              }
            break;
          }
          case Wn: {
            for (unsigned int i = 0; i < dim; i++)
              {
                f_components.insert(
                  {dim + 1 + i,
                   std::make_shared<const ComponentFunction<dim>>(f, i)});
              }
            break;
          }
          case Wp: {
            for (unsigned int i = 0; i < dim; i++)
              {
                f_components.insert(
                  {2 * dim + 2 + i,
                   std::make_shared<const ComponentFunction<dim>>(f, i)});
              }
            break;
          }
        default:
          Assert(false, InvalidDisplacement());
          break;
      }
    std::shared_ptr<dealii::Function<dim>> function_extended =
      std::make_shared<FunctionByComponents<dim>>(3 * (dim + 1), f_components);
    return function_extended;
  }



  template <int dim, typename ProblemType>
  std::shared_ptr<dealii::Function<dim>>
  NPSolver<dim, ProblemType>::extend_function_on_all_trace_components(
    const std::shared_ptr<const dealii::Function<dim>> f,
    const Component                                    c) const
  {
    Assert(f->n_components == 1, FunctionMustBeScalar());
    component_map<dim> f_components;
    switch (c)
      {
          case V: {
            f_components.insert({0, f});
            break;
          }
          case n: {
            f_components.insert({1, f});
            break;
          }
          case p: {
            f_components.insert({2, f});
            break;
          }
        default:
          Assert(false, InvalidComponent());
          break;
      }
    std::shared_ptr<dealii::Function<dim>> function_extended =
      std::make_shared<FunctionByComponents<dim>>(3, f_components);
    return function_extended;
  }



  template <int dim, typename ProblemType>
  unsigned int
  NPSolver<dim, ProblemType>::get_number_of_quadrature_points() const
  {
    Assert(this->initialized, dealii::ExcNotInitialized());
    const unsigned int cell_degree  = this->fe_cell->degree;
    const unsigned int trace_degree = this->fe_trace->degree;
    const unsigned int max_degree =
      (cell_degree > trace_degree) ? cell_degree : trace_degree;
    constexpr unsigned int K = 1;
    return max_degree + K;
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::interpolate_component(
    const Component                                    c,
    const std::shared_ptr<const dealii::Function<dim>> c_function)
  {
    Assert(c == Component::V || c == Component::n || c == Component::p,
           InvalidComponent());

    if (!this->initialized)
      this->setup_overall_system();

    const Displacement d = component2displacement(c);

    std::shared_ptr<dealii::Function<dim>> c_function_rescaled =
      this->adimensionalizer->template adimensionalize_component_function<dim>(
        c_function, c);

    auto c_function_extended =
      this->extend_function_on_all_components(c_function_rescaled, c);
    auto c_grad     = std::make_shared<Gradient<dim>>(c_function_rescaled);
    auto d_function = std::make_shared<Opposite<dim>>(c_grad);
    auto d_function_extended =
      this->extend_function_on_all_components(d_function, d);

    dealii::VectorTools::interpolate(this->dof_handler_cell,
                                     *c_function_extended,
                                     this->current_solution_cell,
                                     this->get_component_mask(c));

    dealii::VectorTools::interpolate(this->dof_handler_cell,
                                     *d_function_extended,
                                     this->current_solution_cell,
                                     this->get_component_mask(d));

    if (dim == 2 || dim == 3)
      {
        auto c_function_trace_extended =
          this->extend_function_on_all_trace_components(c_function_rescaled, c);
        dealii::VectorTools::interpolate(this->dof_handler_trace,
                                         *c_function_trace_extended,
                                         this->current_solution_trace,
                                         this->get_trace_component_mask(c));
        return;
      }

    const QGauss<dim - 1> face_quadrature_formula(
      this->get_number_of_quadrature_points());
    const UpdateFlags flags(update_values | update_quadrature_points);

    FEFaceValues<dim>  fe_face_trace_values(*(this->fe_trace),
                                           face_quadrature_formula,
                                           flags);
    const unsigned int n_face_q_points =
      fe_face_trace_values.get_quadrature().size();
    Assert(n_face_q_points == 1, ExcDimensionMismatch(n_face_q_points, 1));

    const unsigned int dofs_per_cell =
      fe_face_trace_values.get_fe().dofs_per_cell;
    Assert(dofs_per_cell == 6, ExcDimensionMismatch(dofs_per_cell, 6));

    Vector<double> local_values(dofs_per_cell);

    std::vector<Point<dim>>            face_quadrature_points(n_face_q_points);
    dealii::FEValuesExtractors::Scalar extractor =
      this->get_trace_component_extractor(c);
    unsigned int nonzero_shape_functions = 0;
    double       current_value;
    double       function_value;

    for (const auto &cell : this->dof_handler_trace.active_cell_iterators())
      {
        cell->get_dof_values(this->current_solution_trace, local_values);
        for (unsigned int face_number = 0;
             face_number < GeometryInfo<dim>::faces_per_cell;
             ++face_number)
          {
            fe_face_trace_values.reinit(cell, face_number);

            for (unsigned int q = 0; q < n_face_q_points; ++q)
              face_quadrature_points[q] =
                fe_face_trace_values.quadrature_point(q);

            function_value =
              c_function_rescaled->value(face_quadrature_points[0]);

            nonzero_shape_functions = 0;
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                current_value = fe_face_trace_values[extractor].value(k, 0);
                if (abs(current_value) > 1e-9)
                  {
                    ++nonzero_shape_functions;
                    local_values[k] = function_value / current_value;
                  }
              }
            Assert(nonzero_shape_functions > 0,
                   ExcMessage(
                     "No shape function found that is different from zero on "
                     "the current node"));
            Assert(
              nonzero_shape_functions == 1,
              ExcMessage(
                "More than one shape function found that is different from "
                "zero on the current node"));
          }
        cell->set_dof_values(local_values, this->current_solution_trace);
      }
    this->global_constraints.distribute(this->current_solution_trace);
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::set_multithreading(const bool multithreading)
  {
    this->parameters->multithreading = multithreading;
  }



  template <int dim, typename ProblemType>
  bool
  NPSolver<dim, ProblemType>::is_enabled(const Component c) const
  {
    return this->enabled_components.find(c) != this->enabled_components.end();
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::enable_component(const Component c)
  {
    if (this->enabled_components.find(c) == this->enabled_components.end())
      {
        this->enabled_components.insert(c);
        if (this->initialized)
          this->setup_restricted_trace_system();
      }
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::disable_component(const Component c)
  {
    if (this->enabled_components.find(c) != this->enabled_components.end())
      {
        this->enabled_components.erase(c);
        if (this->initialized)
          this->setup_restricted_trace_system();
      }
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::enable_components(const std::set<Component> &cmps)
  {
    bool changed_something = false;

    for (const Component c : cmps)
      if (this->enabled_components.find(c) == this->enabled_components.end())
        {
          this->enabled_components.insert(c);
          changed_something = true;
        }

    if (changed_something && this->initialized)
      this->setup_restricted_trace_system();
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::disable_components(
    const std::set<Component> &cmps)
  {
    bool changed_something = false;

    for (const Component c : cmps)
      if (this->enabled_components.find(c) != this->enabled_components.end())
        {
          this->enabled_components.erase(c);
          changed_something = true;
        }

    if (changed_something && this->initialized)
      this->setup_restricted_trace_system();
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::set_enabled_components(const bool V_enabled,
                                                     const bool n_enabled,
                                                     const bool p_enabled)
  {
    bool changed_something = false;

    std::map<Component, bool> target_state = {{Component::V, V_enabled},
                                              {Component::n, n_enabled},
                                              {Component::p, p_enabled}};

    for (const Component c : all_primary_components())
      {
        const bool c_enabled = target_state.at(c);
        if (c_enabled && !this->is_enabled(c))
          {
            this->enabled_components.insert(c);
            changed_something = true;
          }
        if (!c_enabled && this->is_enabled(c))
          {
            this->enabled_components.erase(c);
            changed_something = true;
          }
      }

    if (changed_something && this->initialized)
      this->setup_restricted_trace_system();
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::set_current_solution(
    const std::shared_ptr<const dealii::Function<dim>> V_function,
    const std::shared_ptr<const dealii::Function<dim>> n_function,
    const std::shared_ptr<const dealii::Function<dim>> p_function,
    const bool                                         use_projection)
  {
    if (!this->initialized)
      this->setup_overall_system();

    this->set_component(Component::V, V_function, use_projection);
    this->set_component(Component::n, n_function, use_projection);
    this->set_component(Component::p, p_function, use_projection);
  }


  template <int dim, typename ProblemType>
  dealii::FEValuesExtractors::Scalar
  NPSolver<dim, ProblemType>::get_component_extractor(
    const Component component) const
  {
    dealii::FEValuesExtractors::Scalar extractor;
    switch (component)
      {
        case Component::V:
          extractor = dealii::FEValuesExtractors::Scalar(dim);
          break;
        case Component::n:
          extractor = dealii::FEValuesExtractors::Scalar(2 * dim + 1);
          break;
        case Component::p:
          extractor = dealii::FEValuesExtractors::Scalar(3 * dim + 2);
          break;
        default:
          Assert(false, InvalidComponent());
          break;
      }
    return extractor;
  }



  template <int dim, typename ProblemType>
  dealii::FEValuesExtractors::Vector
  NPSolver<dim, ProblemType>::get_displacement_extractor(
    const Displacement displacement) const
  {
    dealii::FEValuesExtractors::Vector extractor;
    switch (displacement)
      {
        case Displacement::E:
          extractor = dealii::FEValuesExtractors::Vector(0);
          break;
        case Displacement::Wn:
          extractor = dealii::FEValuesExtractors::Vector(dim + 1);
          break;
        case Displacement::Wp:
          extractor = dealii::FEValuesExtractors::Vector(2 * (dim + 1));
          break;
        default:
          Assert(false, InvalidDisplacement());
          break;
      }
    return extractor;
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::generate_dof_to_component_map(
    std::vector<Component> &dof_to_component,
    std::vector<DofType> &  dof_to_dof_type,
    const bool              for_trace,
    const bool              restricted) const
  {
    Assert(this->initialized, dealii::ExcNotInitialized());

    const auto &fe =
      (for_trace) ?
        ((restricted) ? *(this->fe_trace_restricted) : *(this->fe_trace)) :
        *(this->fe_cell);
    const auto &dof_handler =
      (for_trace) ? ((restricted) ? this->dof_handler_trace_restricted :
                                    this->dof_handler_trace) :
                    this->dof_handler_cell;

    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    AssertDimension(dof_to_component.size(), dof_handler.n_dofs());
    AssertDimension(dof_to_dof_type.size(), dof_handler.n_dofs());

    std::vector<Component> cell_dof_to_component(dofs_per_cell);
    std::vector<DofType>   cell_dof_to_dof_type(dofs_per_cell);

    std::vector<dealii::types::global_dof_index> global_indices(dofs_per_cell);

    std::map<unsigned int, Component> component_from_index;
    std::set<Component>               cmps;
    if (for_trace && restricted)
      cmps = this->enabled_components;
    else
      cmps = Ddhdg::all_primary_components();
    for (const Component c : cmps)
      component_from_index.insert({get_component_index(c, cmps), c});

    // Create a map that associates to each component its FESystem (this is used
    // only when we are on a cell and we have a complex FESystem for each
    // component
    std::map<const Component, const dealii::FiniteElement<dim> &>
      component_to_fe_system;
    if (!for_trace)
      for (const auto c : all_primary_components())
        {
          const Displacement          d          = component2displacement(c);
          const dealii::ComponentMask c_mask     = this->get_component_mask(c);
          const dealii::ComponentMask d_mask     = this->get_component_mask(d);
          const dealii::ComponentMask total_mask = c_mask | d_mask;
          const dealii::FiniteElement<dim> &fe_system =
            this->fe_cell->get_sub_fe(total_mask.first_selected_component(),
                                      total_mask.n_selected_components());
          component_to_fe_system.insert({c, fe_system});
        }

    // Fill the cell_dof_to_component vector
    for (unsigned int i = 0; i < dofs_per_cell; i++)
      {
        const unsigned int current_block = fe.system_to_block_index(i).first;
        const Component    current_component =
          component_from_index.at(current_block);
        cell_dof_to_component[i] = current_component;
      }

    // Fill the cell_dof_to_dof_type vector
    if (for_trace)
      for (unsigned int i = 0; i < dofs_per_cell; i++)
        cell_dof_to_dof_type[i] = DofType::TRACE;
    else
      for (unsigned int i = 0; i < dofs_per_cell; i++)
        {
          const auto block_position = fe.system_to_block_index(i);

          const unsigned int current_block       = block_position.first;
          const unsigned int current_block_index = block_position.second;

          const Component current_component =
            component_from_index.at(current_block);
          const dealii::FiniteElement<dim> &sub_fe =
            component_to_fe_system.at(current_component);

          const unsigned int component_or_displacement =
            sub_fe.system_to_block_index(current_block_index).first;

          switch (component_or_displacement)
            {
              case 0:
                cell_dof_to_dof_type[i] = DofType::DISPLACEMENT;
                break;
              case 1:
                cell_dof_to_dof_type[i] = DofType::COMPONENT;
                break;
              default:
                Assert(false, ExcInternalError("Unexpected index value"));
                break;
            }
        }

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell->get_dof_indices(global_indices);
        for (unsigned int i = 0; i < dofs_per_cell; i++)
          {
            const dealii::types::global_dof_index global_i = global_indices[i];
            dof_to_component[global_i] = cell_dof_to_component[i];
            dof_to_dof_type[global_i]  = cell_dof_to_dof_type[i];
          }
      }
  }



  template <int dim, typename ProblemType>
  dealii::FEValuesExtractors::Scalar
  NPSolver<dim, ProblemType>::get_trace_component_extractor(
    const Component component,
    const bool      restricted) const
  {
    unsigned int c_index;
    if (restricted)
      c_index = get_component_index(component, this->enabled_components);
    else
      c_index = get_component_index(component);

    return dealii::FEValuesExtractors::Scalar(c_index);
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::copy_local_to_global(const PerTaskData &data)
  {
    if (!data.trace_reconstruct)
      {
        this->constraints.distribute_local_to_global(data.tt_matrix,
                                                     data.tt_vector,
                                                     data.dof_indices,
                                                     this->system_matrix,
                                                     this->system_rhs);

        // We copy also the values for the node with the dirichlet boundary
        // conditions
        const unsigned int constrained_dofs = data.n_dirichlet_constrained_dofs;
        for (unsigned int i = 0; i < constrained_dofs; ++i)
          {
            Logging::write_log<Logging::severity_level::trace>(
              "Saving that dof %s must assume value %s because of the "
              "Dirichlet boundary conditions",
              data.dirichlet_trace_dof_indices[i],
              data.dirichlet_trace_dof_values[i]);
            this->constrained_dof_indices[this->n_dirichlet_constraints + i] =
              data.dirichlet_trace_dof_indices[i];
            this->constrained_dof_values[this->n_dirichlet_constraints + i] =
              data.dirichlet_trace_dof_values[i];
          }
        this->n_dirichlet_constraints += constrained_dofs;
      }
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::solve_linear_problem()
  {
    this->write_log("    RHS l2 norm       : %s", this->system_rhs.l2_norm());
    this->write_log("    Matrix linfty norm: %s",
                    this->system_matrix.linfty_norm());

    SolverControl solver_control(this->system_matrix.m() * 10,
                                 1e-10 * this->system_rhs.l2_norm());

    if (parameters->iterative_linear_solver)
      {
        SolverGMRES<> linear_solver(solver_control);
        linear_solver.solve(this->system_matrix,
                            this->system_solution,
                            this->system_rhs,
                            PreconditionIdentity());
        this->write_log("    Number of GMRES iterations: %s",
                        solver_control.last_step());
      }
    else
      {
        SparseDirectUMFPACK Ainv;
        Ainv.initialize(this->system_matrix);
        Ainv.vmult(this->system_solution, this->system_rhs);
      }
    this->constraints.distribute(this->system_solution);
  }


  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::build_restricted_to_trace_dof_map()
  {
    const unsigned int trace_dofs = this->fe_trace->dofs_per_cell;
    const unsigned int trace_restricted_dofs =
      this->fe_trace_restricted->dofs_per_cell;

    this->restricted_to_trace_dof_map.clear();
    this->restricted_to_trace_dof_map.resize(
      this->dof_handler_trace_restricted.n_dofs());

    std::map<unsigned int, Component> component_index;
    for (const Component c : Ddhdg::all_primary_components())
      component_index.insert({get_component_index(c), c});

    std::map<unsigned int, Component> restricted_component_index;
    for (const Component c : this->enabled_components)
      restricted_component_index.insert(
        {get_component_index(c, this->enabled_components), c});

    std::map<Component, std::map<unsigned int, unsigned int>> index_position;

    for (unsigned int i = 0; i < trace_dofs; i++)
      {
        const auto         k = this->fe_trace->system_to_block_index(i);
        const unsigned int current_block    = k.first;
        const unsigned int current_position = k.second;
        const Component current_component   = component_index.at(current_block);
        auto &block_index_to_global_index   = index_position[current_component];
        block_index_to_global_index[current_position] = i;
      }

    std::vector<unsigned int> index_map(trace_restricted_dofs);
    for (unsigned int i = 0; i < trace_restricted_dofs; i++)
      {
        const auto k = this->fe_trace_restricted->system_to_block_index(i);
        const unsigned int current_block    = k.first;
        const unsigned int current_position = k.second;
        const Component    current_component =
          restricted_component_index.at(current_block);
        index_map[i] =
          index_position.at(current_component).at(current_position);
      }

    std::vector<unsigned int> trace_dof_indices(trace_dofs);
    std::vector<unsigned int> trace_restricted_dof_indices(
      trace_restricted_dofs);
    for (const auto &cell : this->dof_handler_trace.active_cell_iterators())
      {
        typename DoFHandler<dim>::active_cell_iterator restricted_cell(
          &(*triangulation),
          cell->level(),
          cell->index(),
          &dof_handler_trace_restricted);

        restricted_cell->get_dof_indices(trace_restricted_dof_indices);
        cell->get_dof_indices(trace_dof_indices);

        for (unsigned int i = 0; i < trace_restricted_dofs; i++)
          {
            const unsigned int j = index_map[i];
            const unsigned int restricted_global_dof =
              trace_restricted_dof_indices[i];
            const unsigned int global_dof = trace_dof_indices[j];
            Assert(
              this->restricted_to_trace_dof_map[restricted_global_dof] == 0 ||
                this->restricted_to_trace_dof_map[restricted_global_dof] ==
                  global_dof,
              ExcInternalError(
                "Error in mapping the dofs from restricted trace to global trace"));
            this->restricted_to_trace_dof_map[restricted_global_dof] =
              global_dof;
          }
      }
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::copy_restricted_to_trace()
  {
    const unsigned int restricted_global_dofs =
      this->dof_handler_trace_restricted.n_dofs();

    for (unsigned int i = 0; i < restricted_global_dofs; i++)
      {
        const unsigned int j  = this->restricted_to_trace_dof_map[i];
        this->update_trace[j] = this->system_solution[i];
      }
  }



  template <int dim, typename ProblemType>
  NonlinearIterationResults
  NPSolver<dim, ProblemType>::private_run(
    const double absolute_tol,
    const double relative_tol,
    const int    max_number_of_iterations,
    const bool   compute_thermodynamic_equilibrium)
  {
    if (!this->initialized)
      this->setup_overall_system();

    bool   convergence_reached = false;
    int    step                = 0;
    double update_cell_norm    = 0.;
    double current_solution_cell_norm;
    double update_trace_norm = 0.;
    double current_solution_trace_norm;
    double update_norm = 0.;
    double current_solution_norm;

    unsigned int dirichlet_constrains_check = this->n_dirichlet_constraints;
    // This variable is used only inside Assert expressions; this prevents the
    // "unused variable" warnings
    (void)dirichlet_constrains_check;

    for (step = 1;
         step <= max_number_of_iterations || max_number_of_iterations < 0;
         step++)
      {
        this->write_log("Computing step number %s", step);

        this->update_trace            = 0;
        this->update_cell             = 0;
        this->system_matrix           = 0;
        this->system_rhs              = 0;
        this->system_solution         = 0;
        this->n_dirichlet_constraints = 0;

        if (parameters->multithreading)
          assemble_system<true>(false, compute_thermodynamic_equilibrium);
        else
          assemble_system<false>(false, compute_thermodynamic_equilibrium);

        Assert(
          this->n_dirichlet_constraints == dirichlet_constrains_check,
          ExcMessage(
            "Wrong number of dofs constrained by the Dirichlet boundary conditions"));

        solve_linear_problem();

        // Copy Dirichlet boundary conditions
        for (unsigned int i = 0; i < this->n_dirichlet_constraints; ++i)
          {
            const dealii::types::global_dof_index dof_index =
              this->constrained_dof_indices[i];
            Logging::write_log<Logging::severity_level::trace>(
              "Setting dof index %s of system_solution to value %s to satisfy "
              "the Dirichlet boundary conditions",
              dof_index,
              this->constrained_dof_values[i]);
            this->system_solution[dof_index] = this->constrained_dof_values[i];
          }

        this->copy_restricted_to_trace();

        if (parameters->multithreading)
          assemble_system<true>(true, compute_thermodynamic_equilibrium);
        else
          assemble_system<false>(true, compute_thermodynamic_equilibrium);

        update_cell_norm           = this->update_cell.linfty_norm();
        current_solution_cell_norm = this->current_solution_cell.linfty_norm();
        update_trace_norm          = this->update_trace.linfty_norm();
        current_solution_trace_norm =
          this->current_solution_trace.linfty_norm();
        update_norm = std::max(update_cell_norm, update_trace_norm);
        current_solution_norm =
          std::max(current_solution_cell_norm, current_solution_trace_norm);

        this->write_log(
          "Difference in linfty norm compared to the previous step: %s",
          update_norm);

        this->update_trace *= 1.;
        this->update_cell *= 1.;

        this->current_solution_trace += this->update_trace;
        this->current_solution_cell += this->update_cell;

        this->write_log("Current solution norm: %s", current_solution_norm);

        if (update_norm < absolute_tol)
          {
            this->write_log(
              "Update is smaller than absolute tolerance. CONVERGENCE REACHED");
            convergence_reached = true;
            break;
          }
        if (update_norm < relative_tol * current_solution_norm)
          {
            this->write_log(
              "Update is smaller than relative tolerance. CONVERGENCE REACHED");
            convergence_reached = true;
            break;
          }
      }

    return NonlinearIterationResults(convergence_reached,
                                     step,
                                     update_cell_norm);
  }



  template <int dim, typename ProblemType>
  NonlinearIterationResults
  NPSolver<dim, ProblemType>::run(const double absolute_tol,
                                  const double relative_tol,
                                  const int    max_number_of_iterations)
  {
    return this->private_run(absolute_tol,
                             relative_tol,
                             max_number_of_iterations,
                             false);
  }



  template <int dim, typename ProblemType>
  NonlinearIterationResults
  NPSolver<dim, ProblemType>::run()
  {
    return this->run(
      this->parameters->nonlinear_solver_absolute_tolerance,
      this->parameters->nonlinear_solver_relative_tolerance,
      this->parameters->nonlinear_solver_max_number_of_iterations);
  }



  template <int dim, typename ProblemType>
  unsigned int
  NPSolver<dim, ProblemType>::get_n_dofs(bool for_trace) const
  {
    if (for_trace)
      return this->dof_handler_trace.n_dofs();
    return this->dof_handler_cell.n_dofs();
  }



  template <int dim, typename ProblemType>
  unsigned int
  NPSolver<dim, ProblemType>::get_n_active_cells() const
  {
    return this->triangulation->n_active_cells();
  }



  template <int dim, typename ProblemType>
  std::shared_ptr<dealii::Function<dim>>
  NPSolver<dim, ProblemType>::get_solution() const
  {
    return std::make_shared<dealii::Functions::FEFieldFunction<dim>>(
      this->dof_handler_cell, this->current_solution_cell);
  }



  template <int dim, typename ProblemType>
  std::shared_ptr<dealii::Function<dim>>
  NPSolver<dim, ProblemType>::get_solution(Component c) const
  {
    const unsigned int c_index = get_component_index(c);
    const unsigned int i       = c_index * (dim + 1) + dim;
    return std::make_shared<ComponentFunction<dim>>(this->get_solution(), i);
  }



  template <int dim, typename ProblemType>
  double
  NPSolver<dim, ProblemType>::get_solution_on_a_point(
    const dealii::Point<dim> &p,
    const Component           c) const
  {
    Assert(this->initialized, dealii::ExcNotInitialized());

    dealii::Functions::FEFieldFunction<dim> fe_field_function(
      this->dof_handler_cell, this->current_solution_cell);

    if (c == Component::V || c == Component::n || c == Component::p)
      {
        const unsigned int c_index          = get_component_index(c);
        const unsigned int dealii_component = (dim + 1) * c_index + dim;
        const double       rescaling_factor =
          this->adimensionalizer->get_component_rescaling_factor(c);

        return fe_field_function.value(p, dealii_component) * rescaling_factor;
      }

    if (c == Component::phi_n || c == Component::phi_p)
      {
        const Component primal_c =
          (c == Component::phi_n) ? Component::n : Component::p;
        const unsigned int primal_c_index = get_component_index(primal_c);
        const unsigned int primal_dealii_component =
          (dim + 1) * primal_c_index + dim;
        const double primal_rescaling_factor =
          this->adimensionalizer->get_component_rescaling_factor(primal_c);

        const unsigned int v_index = get_component_index(Component::V);
        const unsigned int v_dealii_component = (dim + 1) * v_index + dim;
        const double       v_rescaling_factor =
          this->adimensionalizer->get_component_rescaling_factor(Component::V);

        const unsigned int n_of_components = all_primary_components().size();

        dealii::Vector<double> f_values((dim + 1) * n_of_components);

        fe_field_function.vector_value(p, f_values);

        const double primal_value =
          f_values[primal_dealii_component] * primal_rescaling_factor;
        const double v_value =
          f_values[v_dealii_component] * v_rescaling_factor;

        const double temperature = this->problem->temperature->value(p);

        switch (primal_c)
          {
            case Component::n:
              return this->template compute_quasi_fermi_potential<Component::n>(
                primal_value, v_value, temperature);
            case Component::p:
              return this->template compute_quasi_fermi_potential<Component::p>(
                primal_value, v_value, temperature);
            default:
              Assert(false, InvalidComponent());
          }
      }

    AssertThrow(false, InvalidComponent());
  }



  template <int dim, typename ProblemType>
  dealii::Vector<double>
  NPSolver<dim, ProblemType>::get_solution_on_a_point(
    const dealii::Point<dim> &p,
    const Displacement        d) const
  {
    Assert(this->initialized, dealii::ExcNotInitialized());

    dealii::Functions::FEFieldFunction<dim> fe_field_function(
      this->dof_handler_cell, this->current_solution_cell);

    const Component    c                = displacement2component(d);
    const unsigned int c_index          = get_component_index(c);
    const unsigned int dealii_component = (dim + 1) * c_index;
    const unsigned int n_of_components  = all_primary_components().size();
    const double       rescaling_factor =
      this->adimensionalizer->get_component_rescaling_factor(c);

    dealii::Vector<double> all_values((dim + 1) * n_of_components);
    dealii::Vector<double> component_values(dim);

    fe_field_function.vector_value(p, all_values);

    for (unsigned int i = 0; i < dim; i++)
      component_values[i] = all_values[dealii_component + i] *
                            rescaling_factor /
                            this->adimensionalizer->scale_length;

    return component_values;
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::output_results(
    const std::string &solution_filename,
    const bool         save_update,
    const bool         redimensionalize_quantities) const
  {
    std::ofstream output(solution_filename);
    DataOut<dim>  data_out;

    if (dim > 1)
      {
        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;
        data_out.set_flags(flags);
      }

    const std::set<Component> components      = all_primary_components();
    const unsigned int        n_of_components = components.size();

    std::vector<std::string> names;
    for (const Component c : components)
      {
        const Displacement d      = component2displacement(c);
        const std::string  c_name = get_component_name(c);
        const std::string  d_name = get_displacement_name(d);
        for (unsigned int i = 0; i < dim; i++)
          {
            names.emplace_back(d_name);
          }
        names.emplace_back(c_name);
      }

    std::vector<std::string> update_names;
    for (const auto &n : names)
      update_names.push_back(n + "_updates");

    std::vector<Component> dof_to_component_map;
    std::vector<DofType>   dof_to_dof_type_map;
    Vector<double>         rescaled_solution;
    Vector<double>         rescaled_update;

    if (redimensionalize_quantities)
      {
        dof_to_component_map.resize(this->dof_handler_cell.n_dofs());
        dof_to_dof_type_map.resize(this->dof_handler_cell.n_dofs());
        rescaled_solution.reinit(this->current_solution_cell.size());
        this->generate_dof_to_component_map(dof_to_component_map,
                                            dof_to_dof_type_map,
                                            false);

        this->adimensionalizer->redimensionalize_dof_vector(
          this->current_solution_cell,
          dof_to_component_map,
          dof_to_dof_type_map,
          rescaled_solution);
      }

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      component_interpretation(
        n_of_components * (dim + 1),
        DataComponentInterpretation::component_is_part_of_vector);
    for (unsigned int i = 0; i < n_of_components; i++)
      component_interpretation[(dim + 1) * i + dim] =
        DataComponentInterpretation::component_is_scalar;

    data_out.add_data_vector(this->dof_handler_cell,
                             (redimensionalize_quantities) ?
                               rescaled_solution :
                               this->current_solution_cell,
                             names,
                             component_interpretation);

    if (save_update)
      {
        if (redimensionalize_quantities)
          {
            rescaled_update.reinit(this->dof_handler_cell.n_dofs());
            this->adimensionalizer->redimensionalize_dof_vector(
              this->update_cell,
              dof_to_component_map,
              dof_to_dof_type_map,
              rescaled_update);
          }
        data_out.add_data_vector(this->dof_handler_cell,
                                 (redimensionalize_quantities) ?
                                   rescaled_update :
                                   this->update_cell,
                                 update_names,
                                 component_interpretation);
      }

    dealii::DoFHandler<dim> n_current_dofs(*(this->triangulation));
    dealii::DoFHandler<dim> p_current_dofs(*(this->triangulation));

    dealii::DoFHandler<dim> phi_n_dofs(*(this->triangulation));
    dealii::DoFHandler<dim> phi_p_dofs(*(this->triangulation));

    dealii::DoFHandler<dim> problem_const_dofs(*(this->triangulation));

    const unsigned int V_degree = this->parameters->degree.at(Component::V);
    const unsigned int n_degree = this->parameters->degree.at(Component::n);
    const unsigned int p_degree = this->parameters->degree.at(Component::p);

    const unsigned int Jn_degree = (V_degree > n_degree) ? V_degree : n_degree;
    const unsigned int Jp_degree = (V_degree > p_degree) ? V_degree : p_degree;

    const unsigned int max_degree =
      (Jn_degree > Jp_degree) ? Jn_degree : Jp_degree;

    const unsigned int phi_n_degree =
      (V_degree > n_degree) ? V_degree : n_degree;
    const unsigned int phi_p_degree =
      (V_degree > p_degree) ? V_degree : p_degree;

    n_current_dofs.distribute_dofs(FESystem(FE_DGQ<dim>(Jn_degree), dim));
    p_current_dofs.distribute_dofs(FESystem(FE_DGQ<dim>(Jp_degree), dim));

    phi_n_dofs.distribute_dofs(FE_DGQ<dim>(phi_n_degree));
    phi_p_dofs.distribute_dofs(FE_DGQ<dim>(phi_p_degree));

    problem_const_dofs.distribute_dofs(FE_DGQ<dim>(max_degree));

    dealii::Vector<double> Jn_data;
    dealii::Vector<double> Jp_data;

    dealii::Vector<double> phi_n_data;
    dealii::Vector<double> phi_p_data;

    dealii::Vector<double> recombination_data;

    dealii::Vector<double> doping_data(problem_const_dofs.n_dofs());
    dealii::Vector<double> temperature_data(problem_const_dofs.n_dofs());

    dealii::VectorTools::interpolate(problem_const_dofs,
                                     *(this->problem->doping),
                                     doping_data);
    dealii::VectorTools::interpolate(problem_const_dofs,
                                     *(this->problem->temperature),
                                     temperature_data);

    if (!redimensionalize_quantities)
      doping_data /= this->adimensionalizer->doping_magnitude;
    if (!redimensionalize_quantities)
      temperature_data /= this->adimensionalizer->temperature_magnitude;

    this->compute_current<Component::n>(n_current_dofs,
                                        Jn_data,
                                        redimensionalize_quantities);
    this->compute_current<Component::p>(p_current_dofs,
                                        Jp_data,
                                        redimensionalize_quantities);

    this->compute_qf_potential<Component::n>(phi_n_dofs, phi_n_data);
    this->compute_qf_potential<Component::p>(phi_p_dofs, phi_p_data);

    this->compute_recombination_term(problem_const_dofs,
                                     recombination_data,
                                     redimensionalize_quantities);

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      J_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);

    data_out.add_data_vector(problem_const_dofs, doping_data, "doping");
    data_out.add_data_vector(problem_const_dofs,
                             temperature_data,
                             "temperature");

    data_out.add_data_vector(n_current_dofs,
                             Jn_data,
                             std::vector<std::string>(dim, "electron_current"),
                             J_component_interpretation);
    data_out.add_data_vector(p_current_dofs,
                             Jp_data,
                             std::vector<std::string>(dim, "hole_current"),
                             J_component_interpretation);

    data_out.add_data_vector(problem_const_dofs,
                             recombination_data,
                             "recombination_generation_term");

    data_out.add_data_vector(phi_n_dofs,
                             phi_n_data,
                             "electron_quasi_fermi_potential");
    data_out.add_data_vector(phi_p_dofs,
                             phi_p_data,
                             "hole_quasi_fermi_potential");

    data_out.build_patches(StaticMappingQ1<dim>::mapping,
                           this->fe_cell->degree,
                           DataOut<dim>::curved_inner_cells);
    data_out.write_vtk(output);
  }



  template <>
  void
  NPSolver<1,
           Problem<1,
                   HomogeneousPermittivity<1>,
                   HomogeneousElectronMobility<1>,
                   HomogeneousElectronMobility<1>>>::
    output_results(const std::string &solution_filename,
                   const std::string &trace_filename,
                   const bool         save_update,
                   const bool         redimensionalize_quantities) const
  {
    (void)solution_filename;
    (void)trace_filename;
    (void)save_update;
    (void)redimensionalize_quantities;
    AssertThrow(false, NoTraceIn1D());
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::output_results(
    const std::string &solution_filename,
    const std::string &trace_filename,
    const bool         save_update,
    const bool         redimensionalize_quantities) const
  {
    this->output_results(solution_filename,
                         save_update,
                         redimensionalize_quantities);

    std::ofstream     face_output(trace_filename);
    DataOutFaces<dim> data_out_face(false);

    const std::set<Component> components      = all_primary_components();
    const unsigned int        n_of_components = components.size();

    std::vector<std::string> face_names;
    for (const Component c : components)
      face_names.push_back(get_component_name(c));

    std::vector<std::string> update_face_names;
    for (const auto &n : face_names)
      update_face_names.push_back(n + "_updates");

    std::vector<Component> dof_to_component_map;
    std::vector<DofType>   dof_to_dof_type_map;
    Vector<double>         rescaled_solution;
    Vector<double>         rescaled_update;

    if (redimensionalize_quantities)
      {
        dof_to_component_map.resize(this->dof_handler_trace.n_dofs());
        dof_to_dof_type_map.resize(this->dof_handler_trace.n_dofs());
        rescaled_solution.reinit(this->dof_handler_trace.n_dofs());
        this->generate_dof_to_component_map(dof_to_component_map,
                                            dof_to_dof_type_map,
                                            true);

        this->adimensionalizer->redimensionalize_dof_vector(
          this->current_solution_trace,
          dof_to_component_map,
          dof_to_dof_type_map,
          rescaled_solution);
      }

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      face_component_type(n_of_components,
                          DataComponentInterpretation::component_is_scalar);
    data_out_face.add_data_vector(this->dof_handler_trace,
                                  (redimensionalize_quantities) ?
                                    rescaled_solution :
                                    this->current_solution_trace,
                                  face_names,
                                  face_component_type);

    if (save_update)
      {
        if (redimensionalize_quantities)
          {
            rescaled_update.reinit(this->dof_handler_trace.n_dofs());
            this->adimensionalizer->redimensionalize_dof_vector(
              this->update_trace,
              dof_to_component_map,
              dof_to_dof_type_map,
              rescaled_update);
          }
        data_out_face.add_data_vector(this->dof_handler_trace,
                                      (redimensionalize_quantities) ?
                                        rescaled_update :
                                        this->update_trace,
                                      update_face_names,
                                      face_component_type);
      }

    data_out_face.build_patches(fe_trace_restricted->degree);
    data_out_face.write_vtk(face_output);
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::print_convergence_table(
    std::shared_ptr<dealii::ParsedConvergenceTable> error_table,
    std::shared_ptr<const dealii::Function<dim>>    expected_V_solution,
    std::shared_ptr<const dealii::Function<dim>>    expected_n_solution,
    std::shared_ptr<const dealii::Function<dim>>    expected_p_solution,
    unsigned int                                    n_cycles,
    unsigned int                                    initial_refinements,
    std::ostream &                                  out)
  {
    const std::shared_ptr<const dealii::Function<dim>> initial_V_function =
      std::make_shared<const dealii::Functions::ZeroFunction<dim>>();
    const std::shared_ptr<const dealii::Function<dim>> initial_n_function =
      std::make_shared<const dealii::Functions::ZeroFunction<dim>>();
    const std::shared_ptr<const dealii::Function<dim>> initial_p_function =
      std::make_shared<const dealii::Functions::ZeroFunction<dim>>();

    this->print_convergence_table(error_table,
                                  expected_V_solution,
                                  expected_n_solution,
                                  expected_p_solution,
                                  initial_V_function,
                                  initial_n_function,
                                  initial_p_function,
                                  n_cycles,
                                  initial_refinements,
                                  out);
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::print_convergence_table(
    std::shared_ptr<dealii::ParsedConvergenceTable> error_table,
    std::shared_ptr<const dealii::Function<dim>>    expected_V_solution,
    std::shared_ptr<const dealii::Function<dim>>    expected_n_solution,
    std::shared_ptr<const dealii::Function<dim>>    expected_p_solution,
    std::shared_ptr<const dealii::Function<dim>>    initial_V_function,
    std::shared_ptr<const dealii::Function<dim>>    initial_n_function,
    std::shared_ptr<const dealii::Function<dim>>    initial_p_function,
    unsigned int                                    n_cycles,
    unsigned int                                    initial_refinements,
    std::ostream &                                  out)
  {
    this->refine_grid(initial_refinements, false);

    std::shared_ptr<dealii::Function<dim>> expected_V_solution_rescaled =
      this->adimensionalizer->template adimensionalize_component_function<dim>(
        expected_V_solution, Component::V);
    std::shared_ptr<dealii::Function<dim>> expected_n_solution_rescaled =
      this->adimensionalizer->template adimensionalize_component_function<dim>(
        expected_n_solution, Component::n);
    std::shared_ptr<dealii::Function<dim>> expected_p_solution_rescaled =
      this->adimensionalizer->template adimensionalize_component_function<dim>(
        expected_p_solution, Component::p);

    std::map<unsigned int, const std::shared_ptr<const dealii::Function<dim>>>
      components;
    components.insert({dim, expected_V_solution_rescaled});
    components.insert({2 * dim + 1, expected_n_solution_rescaled});
    components.insert({3 * dim + 2, expected_p_solution_rescaled});

    for (unsigned int i = 0; i < dim; i++)
      {
        components.insert(
          {i,
           std::make_shared<Opposite<dim>>(
             get_partial_derivative<dim>(expected_V_solution_rescaled, i))});
        components.insert(
          {dim + 1 + i,
           std::make_shared<Opposite<dim>>(
             get_partial_derivative<dim>(expected_n_solution_rescaled, i))});
        components.insert(
          {2 * dim + 2 + i,
           std::make_shared<Opposite<dim>>(
             get_partial_derivative<dim>(expected_p_solution_rescaled, i))});
      }

    FunctionByComponents expected_solution(3 * (dim + 1), components);

    bool         converged;
    unsigned int iterations;
    double       last_step_difference;

    auto converged_function            = [&]() { return converged; };
    auto iterations_function           = [&]() { return iterations; };
    auto last_step_difference_function = [&]() { return last_step_difference; };

    error_table->add_extra_column("converged", converged_function, false);
    error_table->add_extra_column("iterations", iterations_function, false);
    error_table->add_extra_column("last step",
                                  last_step_difference_function,
                                  false);


    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        this->set_current_solution(initial_V_function,
                                   initial_n_function,
                                   initial_p_function,
                                   true);
        const NonlinearIterationResults iter_status = this->run();

        converged            = iter_status.converged;
        iterations           = iter_status.iterations;
        last_step_difference = iter_status.last_update_norm;

        error_table->error_from_exact(dof_handler_cell,
                                      current_solution_cell,
                                      expected_solution);

        // this->output_results("solution_" + std::to_string(cycle) + ".vtk",
        //                      "trace_" + std::to_string(cycle) + ".vtk");

        if (cycle != n_cycles - 1)
          this->refine_grid_once(false);
      }
    error_table->output_table(out);
  }


  template class NPSolver<1, HomogeneousProblem<1>>;
  template class NPSolver<2, HomogeneousProblem<2>>;
  template class NPSolver<3, HomogeneousProblem<3>>;

} // namespace Ddhdg
