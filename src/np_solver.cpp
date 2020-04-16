#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

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
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include "constants.h"
#include "function_tools.h"
#include "templatized_parameters.h"

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
    const bool         multithreading)
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
  {}

  template <int dim>
  std::unique_ptr<dealii::Triangulation<dim>>
  NPSolver<dim>::copy_triangulation(
    const std::shared_ptr<const dealii::Triangulation<dim>> triangulation)
  {
    std::unique_ptr<dealii::Triangulation<dim>> new_triangulation =
      std::make_unique<dealii::Triangulation<dim>>();
    new_triangulation->copy_triangulation(*triangulation);
    return new_triangulation;
  }



  template <int dim>
  std::map<Component, std::vector<double>>
  NPSolver<dim>::ScratchData::initialize_double_map_on_components(
    const unsigned int n)
  {
    std::map<Component, std::vector<double>> m;
    for (const auto c : Ddhdg::all_components())
      {
        m.insert({c, std::vector<double>(n)});
      }
    return m;
  }



  template <int dim>
  std::map<Component, std::vector<Tensor<1, dim>>>
  NPSolver<dim>::ScratchData::initialize_tensor_map_on_components(
    const unsigned int n)
  {
    std::map<Component, std::vector<Tensor<1, dim>>> m;
    for (const auto c : Ddhdg::all_components())
      {
        m.insert({c, std::vector<Tensor<1, dim>>(n)});
      }
    return m;
  }



  template <int dim>
  std::map<Component, std::vector<double>>
  NPSolver<dim>::ScratchData::initialize_double_map_on_n_and_p(
    const unsigned int k)
  {
    std::map<Component, std::vector<double>> m;
    m.insert({Component::n, std::vector<double>(k)});
    m.insert({Component::p, std::vector<double>(k)});
    return m;
  }



  template <int dim>
  NPSolver<dim>::ScratchData::ScratchData(
    const FiniteElement<dim> & fe_trace_restricted,
    const FiniteElement<dim> & fe_trace,
    const FiniteElement<dim> & fe_cell,
    const QGauss<dim> &        quadrature_formula,
    const QGauss<dim - 1> &    face_quadrature_formula,
    const UpdateFlags          cell_flags,
    const UpdateFlags          cell_face_flags,
    const UpdateFlags          trace_flags,
    const UpdateFlags          trace_restricted_flags,
    const std::set<Component> &enabled_components)
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
    , cell_quadrature_points(quadrature_formula.size())
    , face_quadrature_points(face_quadrature_formula.size())
    , epsilon_cell(quadrature_formula.size())
    , epsilon_face(face_quadrature_formula.size())
    , mu_n_cell(quadrature_formula.size())
    , mu_p_cell(quadrature_formula.size())
    , mu_n_face(face_quadrature_formula.size())
    , mu_p_face(face_quadrature_formula.size())
    , T_cell(quadrature_formula.size())
    , T_face(face_quadrature_formula.size())
    , doping_cell(quadrature_formula.size())
    , r_n_cell(quadrature_formula.size())
    , r_p_cell(quadrature_formula.size())
    , dr_n_cell(initialize_double_map_on_n_and_p(quadrature_formula.size()))
    , dr_p_cell(initialize_double_map_on_n_and_p(quadrature_formula.size()))
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
  {}



  template <int dim>
  std::vector<unsigned int>
  NPSolver<dim>::ScratchData::check_dofs_on_enabled_components(
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



  template <int dim>
  std::vector<std::vector<unsigned int>>
  NPSolver<dim>::ScratchData::check_dofs_on_faces_for_cells(
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



  template <int dim>
  std::vector<std::vector<unsigned int>>
  NPSolver<dim>::ScratchData::check_dofs_on_faces_for_trace(
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



  template <int dim>
  NPSolver<dim>::ScratchData::ScratchData(const ScratchData &sd)
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
    , cell_quadrature_points(sd.cell_quadrature_points)
    , face_quadrature_points(sd.face_quadrature_points)
    , epsilon_cell(sd.epsilon_cell)
    , epsilon_face(sd.epsilon_face)
    , mu_n_cell(sd.mu_n_cell)
    , mu_p_cell(sd.mu_p_cell)
    , mu_n_face(sd.mu_n_face)
    , mu_p_face(sd.mu_p_face)
    , T_cell(sd.T_cell)
    , T_face(sd.T_face)
    , doping_cell(sd.doping_cell)
    , r_n_cell(sd.r_n_cell)
    , r_p_cell(sd.r_p_cell)
    , dr_n_cell(sd.dr_n_cell)
    , dr_p_cell(sd.dr_p_cell)
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
  {}



  template <int dim>
  FESystem<dim>
  NPSolver<dim>::generate_fe_system(
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
          fe_systems.push_back(
            new FESystem<dim>(FE_DGQ<dim>(deg), dim, FE_DGQ<dim>(deg), 1));
          multiplicities.push_back(1);
        }


    const FESystem<dim> output(fe_systems, multiplicities);

    for (const FiniteElement<dim> *fe : fe_systems)
      delete (fe);

    return output;
  }



  template <int dim>
  NPSolver<dim>::NPSolver(
    const std::shared_ptr<const Problem<dim>>       problem,
    const std::shared_ptr<const NPSolverParameters> parameters)
    : triangulation(copy_triangulation(problem->triangulation))
    , permittivity(problem->permittivity)
    , n_electron_mobility(problem->n_electron_mobility)
    , n_recombination_term(problem->n_recombination_term)
    , p_electron_mobility(problem->p_electron_mobility)
    , p_recombination_term(problem->p_recombination_term)
    , temperature(problem->temperature)
    , doping(problem->doping)
    , boundary_handler(problem->boundary_handler)
    , band_density(problem->band_density)
    , band_edge_energy(problem->band_edge_energy)
    , parameters(std::make_unique<NPSolverParameters>(*parameters))
    , fe_cell(std::make_unique<dealii::FESystem<dim>>(
        generate_fe_system(parameters->degree, false)))
    , dof_handler_cell(*triangulation)
    , fe_trace(std::make_unique<dealii::FESystem<dim>>(
        generate_fe_system(parameters->degree, true)))
    , fe_trace_restricted(std::make_unique<dealii::FESystem<dim>>(
        generate_fe_system(parameters->degree, true)))
    , dof_handler_trace(*triangulation)
    , dof_handler_trace_restricted(*triangulation)
  {}



  template <int dim>
  void
  NPSolver<dim>::refine_grid(unsigned int i)
  {
    triangulation->refine_global(i);
    this->initialized = false;
  }



  template <int dim>
  std::map<Component, unsigned int>
  NPSolver<dim>::restrict_degrees_on_enabled_component() const
  {
    std::map<Component, unsigned int> restricted_map;
    for (const auto k : this->parameters->degree)
      {
        if (this->is_enabled(k.first))
          restricted_map.insert(k);
      }
    return restricted_map;
  }



  template <int dim>
  void
  NPSolver<dim>::setup_overall_system()
  {
    this->dof_handler_cell.distribute_dofs(*(this->fe_cell));
    this->dof_handler_trace.distribute_dofs(*(this->fe_trace));

    const unsigned int cell_dofs  = this->dof_handler_cell.n_dofs();
    const unsigned int trace_dofs = this->dof_handler_trace.n_dofs();

    this->current_solution_trace.reinit(trace_dofs);
    this->update_trace.reinit(trace_dofs);

    this->current_solution_cell.reinit(cell_dofs);
    this->update_cell.reinit(cell_dofs);

    this->setup_restricted_trace_system();

    this->initialized = true;
  }



  template <int dim>
  void
  NPSolver<dim>::setup_restricted_trace_system()
  {
    this->fe_trace_restricted.reset(new FESystem<dim>(
      generate_fe_system(restrict_degrees_on_enabled_component(), true)));

    this->dof_handler_trace_restricted.distribute_dofs(
      *(this->fe_trace_restricted));

    const unsigned int trace_restricted_dofs =
      this->dof_handler_trace_restricted.n_dofs();

    std::cout << "   Number of degrees of freedom: " << trace_restricted_dofs
              << std::endl;

    this->system_rhs.reinit(trace_restricted_dofs);
    this->system_solution.reinit(trace_restricted_dofs);

    this->constraints.clear();
    DoFTools::make_hanging_node_constraints(this->dof_handler_trace_restricted,
                                            this->constraints);
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



  template <int dim>
  dealii::ComponentMask
  NPSolver<dim>::get_component_mask(const Component component) const
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



  template <int dim>
  dealii::ComponentMask
  NPSolver<dim>::get_component_mask(const Displacement displacement) const
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



  template <int dim>
  dealii::ComponentMask
  NPSolver<dim>::get_trace_component_mask(const Component component) const
  {
    dealii::ComponentMask mask(3, false);
    switch (component)
      {
        case Component::V:
          mask.set(0, true);
          break;
        case Component::n:
          mask.set(1, true);
          break;
        case Component::p:
          mask.set(2, true);
          break;
        default:
          Assert(false, InvalidComponent());
          break;
      }
    return mask;
  }



  template <int dim>
  std::shared_ptr<dealii::Function<dim>>
  NPSolver<dim>::extend_function_on_all_components(
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



  template <int dim>
  std::shared_ptr<dealii::Function<dim>>
  NPSolver<dim>::extend_function_on_all_components(
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



  template <int dim>
  std::shared_ptr<dealii::Function<dim>>
  NPSolver<dim>::extend_function_on_all_trace_components(
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



  template <int dim>
  unsigned int
  NPSolver<dim>::get_number_of_quadrature_points() const
  {
    Assert(this->initialized, dealii::ExcNotInitialized());
    if (this->fe_cell->degree > this->fe_trace->degree)
      return this->fe_cell->degree + 1;
    return this->fe_trace->degree + 1;
  }



  template <int dim>
  void
  NPSolver<dim>::interpolate_component(
    const Component                                    c,
    const std::shared_ptr<const dealii::Function<dim>> c_function)
  {
    if (!this->initialized)
      this->setup_overall_system();

    const Displacement d = component2displacement(c);

    auto c_function_extended =
      this->extend_function_on_all_components(c_function, c);
    auto c_grad     = std::make_shared<Gradient<dim>>(c_function);
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
          this->extend_function_on_all_trace_components(c_function, c);
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

            function_value = c_function->value(face_quadrature_points[0]);

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
  }



  template <int dim>
  void
  NPSolver<dim>::project_component(
    const Component                                    c,
    const std::shared_ptr<const dealii::Function<dim>> c_function)
  {
    if (!this->initialized)
      this->setup_overall_system();

    const Displacement d = component2displacement(c);

    const QGauss<dim> quadrature_formula(
      this->get_number_of_quadrature_points());
    const QGauss<dim - 1> face_quadrature_formula(
      this->get_number_of_quadrature_points());

    const unsigned int c_index = get_component_index(c);

    // In this scope, the code takes care of project the right solution in the
    // FEM space of the cells. After that, we will project the solution on the
    // trace. I split the code in two parts because, in this way, it is
    // possible to handle the situations in which the space of the cell is
    // different from the space of the trace (for example, for HDG(A))
    {
      const UpdateFlags flags_cell(update_values | update_gradients |
                                   update_JxW_values |
                                   update_quadrature_points);
      const UpdateFlags flags_trace(update_values | update_normal_vectors |
                                    update_quadrature_points |
                                    update_JxW_values);

      FEValues<dim>     fe_values_cell(*(this->fe_cell),
                                   quadrature_formula,
                                   flags_cell);
      FEFaceValues<dim> fe_face_values(*(this->fe_cell),
                                       face_quadrature_formula,
                                       flags_trace);

      const unsigned int n_q_points = fe_values_cell.get_quadrature().size();
      const unsigned int n_face_q_points =
        fe_face_values.get_quadrature().size();

      // Now we need to map the dofs that are related to the current component
      std::vector<unsigned int> on_current_component;
      for (unsigned int i = 0; i < this->fe_cell->dofs_per_cell; ++i)
        {
          const unsigned int current_index =
            this->fe_cell->system_to_block_index(i).first;
          if (current_index == c_index)
            on_current_component.push_back(i);
        }
      const unsigned int dofs_per_component = on_current_component.size();

      std::vector<std::vector<unsigned int>> component_support_on_face(
        GeometryInfo<dim>::faces_per_cell);
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        for (unsigned int i = 0; i < dofs_per_component; ++i)
          {
            const unsigned int dof_index = on_current_component[i];
            if (this->fe_cell->has_support_on_face(dof_index, face))
              component_support_on_face[face].push_back(i);
          }
      const unsigned int dofs_per_face_on_component =
        component_support_on_face[0].size();

      const FEValuesExtractors::Vector d_extractor =
        this->get_displacement_extractor(d);
      const FEValuesExtractors::Scalar c_extractor =
        this->get_component_extractor(c);

      LAPACKFullMatrix<double> local_matrix(dofs_per_component,
                                            dofs_per_component);
      Vector<double>           local_residual(dofs_per_component);
      Vector<double>           local_values(fe_values_cell.dofs_per_cell);

      // Temporary buffer for the values of the local base function on a
      // quadrature point
      std::vector<double>         c_bf(dofs_per_component);
      std::vector<Tensor<1, dim>> d_bf(dofs_per_component);
      std::vector<double>         d_div_bf(dofs_per_component);

      std::vector<Point<dim>> cell_quadrature_points(n_q_points);
      std::vector<Point<dim>> face_quadrature_points(n_face_q_points);

      std::vector<double> evaluated_c(n_q_points);
      std::vector<double> evaluated_c_face(n_face_q_points);

      for (const auto &cell : this->dof_handler_cell.active_cell_iterators())
        {
          local_matrix   = 0.;
          local_residual = 0.;

          fe_values_cell.reinit(cell);

          // Get the position of the quadrature points
          for (unsigned int q = 0; q < n_q_points; ++q)
            cell_quadrature_points[q] = fe_values_cell.quadrature_point(q);

          // Evaluated the analytic functions over the quadrature points
          c_function->value_list(cell_quadrature_points, evaluated_c);

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              // Copy data of the shape function
              for (unsigned int k = 0; k < dofs_per_component; ++k)
                {
                  const unsigned int i = on_current_component[k];
                  c_bf[k]     = fe_values_cell[c_extractor].value(i, q);
                  d_bf[k]     = fe_values_cell[d_extractor].value(i, q);
                  d_div_bf[k] = fe_values_cell[d_extractor].divergence(i, q);
                }

              const double JxW = fe_values_cell.JxW(q);

              for (unsigned int i = 0; i < dofs_per_component; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_component; ++j)
                    {
                      local_matrix(i, j) +=
                        (c_bf[j] * c_bf[i] + d_bf[i] * d_bf[j]) * JxW;
                    }
                  local_residual[i] +=
                    (evaluated_c[q] * (c_bf[i] + d_div_bf[i])) * JxW;
                }
            }
          for (unsigned int face_number = 0;
               face_number < GeometryInfo<dim>::faces_per_cell;
               ++face_number)
            {
              fe_face_values.reinit(cell, face_number);

              for (unsigned int q = 0; q < n_face_q_points; ++q)
                face_quadrature_points[q] = fe_face_values.quadrature_point(q);

              c_function->value_list(face_quadrature_points, evaluated_c_face);

              for (unsigned int q = 0; q < n_face_q_points; ++q)
                {
                  const double JxW    = fe_face_values.JxW(q);
                  const auto   normal = fe_face_values.normal_vector(q);

                  for (unsigned int k = 0; k < dofs_per_face_on_component; ++k)
                    {
                      const auto kk = component_support_on_face[face_number][k];
                      const auto kkk = on_current_component[kk];
                      const auto f_bf_face =
                        fe_face_values[d_extractor].value(kkk, q);
                      local_residual[kk] +=
                        (-evaluated_c_face[q] * (f_bf_face * normal)) * JxW;
                    }
                }
            }
          local_matrix.compute_lu_factorization();
          local_matrix.solve(local_residual);

          cell->get_dof_values(this->current_solution_cell, local_values);
          for (unsigned int i = 0; i < dofs_per_component; i++)
            local_values[on_current_component[i]] = local_residual[i];
          cell->set_dof_values(local_values, this->current_solution_cell);
        }
    }

    // This is the part for the trace
    {
      const UpdateFlags flags(update_values | update_normal_vectors |
                              update_quadrature_points | update_JxW_values);

      FEFaceValues<dim>  fe_face_trace_values(*(this->fe_trace),
                                             face_quadrature_formula,
                                             flags);
      const unsigned int n_face_q_points =
        fe_face_trace_values.get_quadrature().size();

      const unsigned int dofs_per_cell = this->fe_trace->dofs_per_cell;

      const FEValuesExtractors::Scalar c_extractor =
        this->get_trace_component_extractor(c);

      // Again, we map the dofs that are related to the current component
      std::vector<unsigned int> on_current_component;
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const unsigned int current_index =
            this->fe_trace->system_to_block_index(i).first;
          if (current_index == c_index)
            on_current_component.push_back(i);
        }
      const unsigned int dofs_per_component = on_current_component.size();

      std::vector<std::vector<unsigned int>> component_support_on_face(
        GeometryInfo<dim>::faces_per_cell);
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        for (unsigned int i = 0; i < dofs_per_component; ++i)
          {
            const unsigned int dof_index = on_current_component[i];
            if (this->fe_trace->has_support_on_face(dof_index, face))
              component_support_on_face[face].push_back(i);
          }
      const unsigned int dofs_per_face_on_component =
        component_support_on_face[0].size();

      std::vector<double> c_bf(dofs_per_face_on_component);

      std::vector<Point<dim>> face_quadrature_points(n_face_q_points);

      std::vector<double> evaluated_c_face(n_face_q_points);

      LAPACKFullMatrix<double> local_trace_matrix(dofs_per_component,
                                                  dofs_per_component);
      Vector<double>           local_trace_residual(dofs_per_component);
      Vector<double>           local_trace_values(dofs_per_cell);

      for (const auto &cell : dof_handler_trace.active_cell_iterators())
        {
          local_trace_matrix   = 0;
          local_trace_residual = 0;
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              fe_face_trace_values.reinit(cell, face);
              for (unsigned int q = 0; q < n_face_q_points; ++q)
                face_quadrature_points[q] =
                  fe_face_trace_values.quadrature_point(q);

              c_function->value_list(face_quadrature_points, evaluated_c_face);

              for (unsigned int q = 0; q < n_face_q_points; ++q)
                {
                  // Copy data of the shape function
                  for (unsigned int k = 0; k < dofs_per_face_on_component; ++k)
                    {
                      const unsigned int component_index =
                        component_support_on_face[face][k];
                      const unsigned int local_index =
                        on_current_component[component_index];
                      c_bf[k] =
                        fe_face_trace_values[c_extractor].value(local_index, q);
                    }

                  const double JxW = fe_face_trace_values.JxW(q);

                  for (unsigned int i = 0; i < dofs_per_face_on_component; ++i)
                    {
                      const unsigned int ii =
                        component_support_on_face[face][i];
                      for (unsigned int j = 0; j < dofs_per_face_on_component;
                           ++j)
                        {
                          const unsigned int jj =
                            component_support_on_face[face][j];
                          local_trace_matrix(ii, jj) +=
                            (c_bf[j] * c_bf[i]) * JxW;
                        }
                      local_trace_residual[ii] +=
                        (evaluated_c_face[q] * c_bf[i]) * JxW;
                    }
                }
            }
          local_trace_matrix.compute_lu_factorization();
          local_trace_matrix.solve(local_trace_residual);

          cell->get_dof_values(this->current_solution_trace,
                               local_trace_values);
          for (unsigned int i = 0; i < dofs_per_component; i++)
            local_trace_values[on_current_component[i]] =
              local_trace_residual[i];
          cell->set_dof_values(local_trace_values,
                               this->current_solution_trace);
        }
    }
  }



  template <int dim>
  void
  NPSolver<dim>::set_multithreading(const bool multithreading)
  {
    this->parameters->multithreading = multithreading;
  }



  template <int dim>
  bool
  NPSolver<dim>::is_enabled(const Component c) const
  {
    return this->enabled_components.find(c) != this->enabled_components.end();
  }



  template <int dim>
  void
  NPSolver<dim>::enable_component(const Component c)
  {
    if (this->enabled_components.find(c) == this->enabled_components.end())
      {
        this->enabled_components.insert(c);
        if (this->initialized)
          this->setup_restricted_trace_system();
      }
  }



  template <int dim>
  void
  NPSolver<dim>::disable_component(const Component c)
  {
    if (this->enabled_components.find(c) != this->enabled_components.end())
      {
        this->enabled_components.erase(c);
        if (this->initialized)
          this->setup_restricted_trace_system();
      }
  }



  template <int dim>
  void
  NPSolver<dim>::enable_components(const std::set<Component> &cmps)
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



  template <int dim>
  void
  NPSolver<dim>::disable_components(const std::set<Component> &cmps)
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



  template <int dim>
  void
  NPSolver<dim>::set_enabled_components(const bool V_enabled,
                                        const bool n_enabled,
                                        const bool p_enabled)
  {
    bool changed_something = false;

    std::map<Component, bool> target_state = {{Component::V, V_enabled},
                                              {Component::n, n_enabled},
                                              {Component::p, p_enabled}};

    for (const Component c : all_components())
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



  template <int dim>
  void
  NPSolver<dim>::set_current_solution(
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


  template <int dim>
  dealii::FEValuesExtractors::Scalar
  NPSolver<dim>::get_component_extractor(const Component component) const
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



  template <int dim>
  dealii::FEValuesExtractors::Vector
  NPSolver<dim>::get_displacement_extractor(
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



  template <int dim>
  dealii::FEValuesExtractors::Scalar
  NPSolver<dim>::get_trace_component_extractor(const Component component,
                                               const bool      restricted) const
  {
    unsigned int c_index;
    if (restricted)
      c_index = get_component_index(component, this->enabled_components);
    else
      c_index = get_component_index(component);

    return dealii::FEValuesExtractors::Scalar(c_index);
  }



  template <int dim>
  void
  NPSolver<dim>::assemble_system_multithreaded(
    const bool trace_reconstruct,
    const bool compute_thermodynamic_equilibrium)
  {
    Assert(this->initialized, dealii::ExcNotInitialized());

    const QGauss<dim> quadrature_formula(
      this->get_number_of_quadrature_points());
    const QGauss<dim - 1> face_quadrature_formula(
      this->get_number_of_quadrature_points());

    const UpdateFlags flags_cell(update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points);
    const UpdateFlags face_flags_cell(update_values);
    const UpdateFlags flags_trace(update_values);
    const UpdateFlags flags_trace_restricted(
      update_values | update_normal_vectors | update_quadrature_points |
      update_JxW_values);
    PerTaskData task_data(this->fe_trace_restricted->dofs_per_cell,
                          trace_reconstruct);
    ScratchData scratch(*(this->fe_trace_restricted),
                        *(this->fe_trace),
                        *(this->fe_cell),
                        quadrature_formula,
                        face_quadrature_formula,
                        flags_cell,
                        face_flags_cell,
                        flags_trace,
                        flags_trace_restricted,
                        this->enabled_components);
    WorkStream::run(dof_handler_trace_restricted.begin_active(),
                    dof_handler_trace_restricted.end(),
                    *this,
                    this->get_assemble_system_one_cell_function(
                      compute_thermodynamic_equilibrium),
                    &NPSolver<dim>::copy_local_to_global,
                    scratch,
                    task_data);
  }

  template <int dim>
  void
  NPSolver<dim>::assemble_system(const bool trace_reconstruct,
                                 const bool compute_thermodynamic_equilibrium)
  {
    Assert(this->initialized, dealii::ExcNotInitialized());

    const QGauss<dim> quadrature_formula(
      this->get_number_of_quadrature_points());
    const QGauss<dim - 1> face_quadrature_formula(
      this->get_number_of_quadrature_points());

    const UpdateFlags flags_cell(update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points);
    const UpdateFlags face_flags_cell(update_values);
    const UpdateFlags flags_trace(update_values);
    const UpdateFlags flags_trace_restricted(
      update_values | update_normal_vectors | update_quadrature_points |
      update_JxW_values);

    PerTaskData task_data(this->fe_trace_restricted->dofs_per_cell,
                          trace_reconstruct);
    ScratchData scratch(*(this->fe_trace_restricted),
                        *(this->fe_trace),
                        *(this->fe_cell),
                        quadrature_formula,
                        face_quadrature_formula,
                        flags_cell,
                        face_flags_cell,
                        flags_trace,
                        flags_trace_restricted,
                        this->enabled_components);

    for (const auto &cell :
         this->dof_handler_trace_restricted.active_cell_iterators())
      {
        (this->*get_assemble_system_one_cell_function(
                  compute_thermodynamic_equilibrium))(cell, scratch, task_data);
        copy_local_to_global(task_data);
      }
  }



  template <int dim>
  typename NPSolver<dim>::assemble_system_one_cell_pointer
  NPSolver<dim>::get_assemble_system_one_cell_function(
    const bool compute_thermodynamic_equilibrium)
  {
    // The last three bits of parameter_mask are the values of the flags
    // this->is_enabled(Component::V), this->is_enabled(Component::n) and
    // this->is_enabled(Component::p); as usual 1 is for TRUE, 0 is FALSE.
    unsigned int parameter_mask = 0;
    if (compute_thermodynamic_equilibrium)
      parameter_mask += 8;
    if (this->is_enabled(Component::V))
      parameter_mask += 4;
    if (this->is_enabled(Component::n))
      parameter_mask += 2;
    if (this->is_enabled(Component::p))
      parameter_mask += 1;

    TemplatizedParametersInterface<dim> *p1;
    TemplatizedParametersInterface<dim> *p2;

    p1 = new TemplatizedParameters<dim, 15>();
    while (p1->get_parameter_mask() != parameter_mask)
      {
        p2 = p1->get_previous();
        delete p1;
        p1 = p2;
      }

    typename NPSolver<dim>::assemble_system_one_cell_pointer f =
      p1->get_assemble_system_one_cell_function();
    delete p1;
    return f;
  }



  template <int dim>
  void
  NPSolver<dim>::prepare_data_on_cell_quadrature_points(
    Ddhdg::NPSolver<dim>::ScratchData &scratch)
  {
    const unsigned int n_q_points =
      scratch.fe_values_cell.get_quadrature().size();

    // Copy the values of the previous solution regarding the previous cell in
    // the scratch. This must be done for every component because, for
    // example, the equation in n requires the data from p and so on
    for (const auto c : Ddhdg::all_components())
      {
        const Displacement d = component2displacement(c);

        const FEValuesExtractors::Scalar c_extractor =
          this->get_component_extractor(c);
        const FEValuesExtractors::Vector d_extractor =
          this->get_displacement_extractor(d);

        scratch.fe_values_cell[c_extractor].get_function_values(
          current_solution_cell, scratch.previous_c_cell.at(c));
        scratch.fe_values_cell[d_extractor].get_function_values(
          current_solution_cell, scratch.previous_d_cell.at(c));
      }

    // Get the position of the quadrature points
    for (unsigned int q = 0; q < n_q_points; ++q)
      scratch.cell_quadrature_points[q] =
        scratch.fe_values_cell.quadrature_point(q);

    // Compute the value of epsilon
    this->permittivity->compute_absolute_permittivity(
      scratch.cell_quadrature_points, scratch.epsilon_cell);

    // Compute the value of mu
    if (this->is_enabled(Component::n))
      this->n_electron_mobility->compute_electron_mobility(
        scratch.cell_quadrature_points, scratch.mu_n_cell);

    if (this->is_enabled(Component::p))
      this->p_electron_mobility->compute_electron_mobility(
        scratch.cell_quadrature_points, scratch.mu_p_cell);

    // Compute the value of T
    if (this->is_enabled(Component::n) || this->is_enabled(Component::p))
      this->temperature->value_list(scratch.cell_quadrature_points,
                                    scratch.T_cell);

    // Compute the value of the doping
    if (this->is_enabled(Component::V))
      this->doping->value_list(scratch.cell_quadrature_points,
                               scratch.doping_cell);

    // Compute the value of the recombination term and its derivative respect
    // to n and p
    if (this->is_enabled(Component::n))
      {
        this->n_recombination_term->compute_multiple_recombination_terms(
          scratch.previous_c_cell.at(Component::n),
          scratch.previous_c_cell.at(Component::p),
          scratch.cell_quadrature_points,
          scratch.r_n_cell);
        this->n_recombination_term
          ->compute_multiple_derivatives_of_recombination_terms(
            scratch.previous_c_cell.at(Component::n),
            scratch.previous_c_cell.at(Component::p),
            scratch.cell_quadrature_points,
            Component::n,
            scratch.dr_n_cell.at(Component::n));
        this->n_recombination_term
          ->compute_multiple_derivatives_of_recombination_terms(
            scratch.previous_c_cell.at(Component::n),
            scratch.previous_c_cell.at(Component::p),
            scratch.cell_quadrature_points,
            Component::p,
            scratch.dr_n_cell.at(Component::p));
      }

    if (this->is_enabled(Component::p))
      {
        this->p_recombination_term->compute_multiple_recombination_terms(
          scratch.previous_c_cell.at(Component::n),
          scratch.previous_c_cell.at(Component::p),
          scratch.cell_quadrature_points,
          scratch.r_p_cell);
        this->p_recombination_term
          ->compute_multiple_derivatives_of_recombination_terms(
            scratch.previous_c_cell.at(Component::n),
            scratch.previous_c_cell.at(Component::p),
            scratch.cell_quadrature_points,
            Component::n,
            scratch.dr_p_cell.at(Component::n));
        this->p_recombination_term
          ->compute_multiple_derivatives_of_recombination_terms(
            scratch.previous_c_cell.at(Component::n),
            scratch.previous_c_cell.at(Component::p),
            scratch.cell_quadrature_points,
            Component::p,
            scratch.dr_p_cell.at(Component::p));
      }
  }



  template <int dim>
  template <typename prm>
  void
  NPSolver<dim>::add_cell_products_to_cc_matrix(
    Ddhdg::NPSolver<dim>::ScratchData &scratch)
  {
    const unsigned int n_q_points =
      scratch.fe_values_cell.get_quadrature().size();
    const unsigned int loc_dofs_per_cell =
      scratch.fe_values_cell.get_fe().dofs_per_cell;

    const FEValuesExtractors::Vector electric_field =
      this->get_displacement_extractor(Displacement::E);
    const FEValuesExtractors::Scalar electric_potential =
      this->get_component_extractor(Component::V);
    const FEValuesExtractors::Vector electron_displacement =
      this->get_displacement_extractor(Displacement::Wn);
    const FEValuesExtractors::Scalar electron_density =
      this->get_component_extractor(Component::n);
    const FEValuesExtractors::Vector hole_displacement =
      this->get_displacement_extractor(Displacement::Wp);
    const FEValuesExtractors::Scalar hole_density =
      this->get_component_extractor(Component::p);

    // The following are just aliases (and some of them refer to the same
    // vector). These may seem useless, but the make the code that assemble
    // the matrix a lot more understandable
    auto &E = scratch.d.at(Component::V);
    auto &V = scratch.c.at(Component::V);

    auto &Wn = scratch.d.at(Component::n);
    auto &n  = scratch.c.at(Component::n);

    auto &Wp = scratch.d.at(Component::p);
    auto &p  = scratch.c.at(Component::p);

    auto &q1      = scratch.d.at(Component::V);
    auto &q1_div  = scratch.d_div.at(Component::V);
    auto &z1      = scratch.c.at(Component::V);
    auto &z1_grad = scratch.c_grad.at(Component::V);

    auto &q2      = scratch.d.at(Component::n);
    auto &q2_div  = scratch.d_div.at(Component::n);
    auto &z2      = scratch.c.at(Component::n);
    auto &z2_grad = scratch.c_grad.at(Component::n);

    auto &q3      = scratch.d.at(Component::p);
    auto &q3_div  = scratch.d_div.at(Component::p);
    auto &z3      = scratch.c.at(Component::p);
    auto &z3_grad = scratch.c_grad.at(Component::p);

    const auto &V0 = scratch.previous_c_cell.at(Component::V);
    const auto &n0 = scratch.previous_c_cell.at(Component::n);
    const auto &p0 = scratch.previous_c_cell.at(Component::p);
    const auto &E0 = scratch.previous_d_cell.at(Component::V);

    const unsigned int dofs_per_component =
      scratch.enabled_component_indices.size();

    dealii::Tensor<1, dim> mu_n_times_previous_E;
    dealii::Tensor<1, dim> mu_p_times_previous_E;

    dealii::Tensor<2, dim> n_einstein_diffusion_coefficient;
    dealii::Tensor<2, dim> p_einstein_diffusion_coefficient;

    const std::vector<double> &dr_n_n = scratch.dr_n_cell.at(Component::n);
    const std::vector<double> &dr_n_p = scratch.dr_n_cell.at(Component::p);

    const std::vector<double> &dr_p_n = scratch.dr_p_cell.at(Component::n);
    const std::vector<double> &dr_p_p = scratch.dr_p_cell.at(Component::p);

    const double nv = this->band_density.at(Component::n);
    const double nc = this->band_density.at(Component::p);
    const double ev = this->band_edge_energy.at(Component::n);
    const double ec = this->band_edge_energy.at(Component::p);

    double thermodynamic_equilibrium_der = 0.;

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double JxW = scratch.fe_values_cell.JxW(q);
        for (unsigned int k = 0; k < loc_dofs_per_cell; ++k)
          {
            if (prm::is_V_enabled)
              {
                q1[k] = scratch.fe_values_cell[electric_field].value(k, q);
                q1_div[k] =
                  scratch.fe_values_cell[electric_field].divergence(k, q);
                z1[k] = scratch.fe_values_cell[electric_potential].value(k, q);
                z1_grad[k] =
                  scratch.fe_values_cell[electric_potential].gradient(k, q);
              }

            if (prm::is_n_enabled)
              {
                q2[k] =
                  scratch.fe_values_cell[electron_displacement].value(k, q);
                q2_div[k] =
                  scratch.fe_values_cell[electron_displacement].divergence(k,
                                                                           q);
                z2[k] = scratch.fe_values_cell[electron_density].value(k, q);
                z2_grad[k] =
                  scratch.fe_values_cell[electron_density].gradient(k, q);
              }

            if (prm::is_p_enabled)
              {
                q3[k] = scratch.fe_values_cell[hole_displacement].value(k, q);
                q3_div[k] =
                  scratch.fe_values_cell[hole_displacement].divergence(k, q);
                z3[k] = scratch.fe_values_cell[hole_density].value(k, q);
                z3_grad[k] =
                  scratch.fe_values_cell[hole_density].gradient(k, q);
              }
          }

        if (prm::is_n_enabled)
          mu_n_times_previous_E = scratch.mu_n_cell[q] * E0[q];

        if (prm::is_p_enabled)
          mu_p_times_previous_E = scratch.mu_p_cell[q] * E0[q];

        if (prm::is_n_enabled)
          n_einstein_diffusion_coefficient =
            scratch
              .template compute_einstein_diffusion_coefficient<Component::n,
                                                               false>(q);
        if (prm::is_p_enabled)
          p_einstein_diffusion_coefficient =
            scratch
              .template compute_einstein_diffusion_coefficient<Component::p,
                                                               false>(q);

        if (prm::thermodyn_eq)
          {
            const double KbT        = Constants::KB * scratch.T_cell[q];
            const double q_over_KbT = Constants::Q / KbT;
            thermodynamic_equilibrium_der =
              -Constants::Q *
              (nv * q_over_KbT * exp((ev - Constants::Q * V0[q]) / KbT) +
               nc * q_over_KbT * exp((Constants::Q * V0[q] - ec) / KbT));
          }

        for (unsigned int i = 0; i < dofs_per_component; ++i)
          {
            const unsigned int ii = scratch.enabled_component_indices[i];
            for (unsigned int j = 0; j < dofs_per_component; ++j)
              {
                const unsigned int jj = scratch.enabled_component_indices[j];
                if (prm::is_V_enabled)
                  {
                    scratch.cc_matrix(i, j) +=
                      (-V[jj] * q1_div[ii] + E[jj] * q1[ii] -
                       (scratch.epsilon_cell[q] * E[jj]) * z1_grad[ii]) *
                      JxW;

                    if (prm::thermodyn_eq)
                      scratch.cc_matrix(i, j) +=
                        -thermodynamic_equilibrium_der * V[jj] * z1[ii] * JxW;
                    else
                      scratch.cc_matrix(i, j) +=
                        Constants::Q * (n[jj] - p[jj]) * z1[ii] * JxW;
                  }
                if (prm::is_n_enabled)
                  scratch.cc_matrix(i, j) +=
                    (-n[jj] * q2_div[ii] + Wn[jj] * q2[ii] -
                     n[jj] * (mu_n_times_previous_E * z2_grad[ii]) +
                     n0[q] * ((scratch.mu_n_cell[q] * E[jj]) * z2_grad[ii]) +
                     (n_einstein_diffusion_coefficient * Wn[jj]) * z2_grad[ii] -
                     (dr_n_n[q] * n[jj] + dr_n_p[q] * p[jj]) * z2[ii] /
                       Constants::Q) *
                    JxW;
                if (prm::is_p_enabled)
                  scratch.cc_matrix(i, j) +=
                    (-p[jj] * q3_div[ii] + Wp[jj] * q3[ii] +
                     p[jj] * (mu_p_times_previous_E * z3_grad[ii]) -
                     p0[q] * ((scratch.mu_p_cell[q] * E[jj]) * z3_grad[ii]) +
                     (p_einstein_diffusion_coefficient * Wp[jj]) * z3_grad[ii] -
                     (dr_p_n[q] * n[jj] + dr_p_p[q] * p[jj]) * z3[ii] /
                       Constants::Q) *
                    JxW;
              }
          }
      }
  }



  template <int dim>
  template <typename prm>
  void
  NPSolver<dim>::add_cell_products_to_cc_rhs(
    Ddhdg::NPSolver<dim>::ScratchData &scratch)
  {
    const unsigned int n_q_points =
      scratch.fe_values_cell.get_quadrature().size();
    const unsigned int dofs_per_component =
      scratch.enabled_component_indices.size();

    const FEValuesExtractors::Vector electric_field =
      this->get_displacement_extractor(Displacement::E);
    const FEValuesExtractors::Scalar electric_potential =
      this->get_component_extractor(Component::V);
    const FEValuesExtractors::Vector electron_displacement =
      this->get_displacement_extractor(Displacement::Wn);
    const FEValuesExtractors::Scalar electron_density =
      this->get_component_extractor(Component::n);
    const FEValuesExtractors::Vector hole_displacement =
      this->get_displacement_extractor(Displacement::Wp);
    const FEValuesExtractors::Scalar hole_density =
      this->get_component_extractor(Component::p);

    const auto &V0  = scratch.previous_c_cell.at(Component::V);
    const auto &E0  = scratch.previous_d_cell.at(Component::V);
    const auto &n0  = scratch.previous_c_cell.at(Component::n);
    const auto &Wn0 = scratch.previous_d_cell.at(Component::n);
    const auto &p0  = scratch.previous_c_cell.at(Component::p);
    const auto &Wp0 = scratch.previous_d_cell.at(Component::p);

    const auto &c0 = scratch.doping_cell;

    dealii::Tensor<2, dim> n_einstein_diffusion_coefficient;
    dealii::Tensor<2, dim> p_einstein_diffusion_coefficient;

    dealii::Tensor<1, dim> Jn;
    dealii::Tensor<1, dim> Jp;

    const double nv = this->band_density.at(Component::n);
    const double nc = this->band_density.at(Component::p);
    const double ev = this->band_edge_energy.at(Component::n);
    const double ec = this->band_edge_energy.at(Component::p);

    double thermodynamic_equilibrium_rhs = 0.;

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double JxW = scratch.fe_values_cell.JxW(q);

        if (prm::is_n_enabled)
          n_einstein_diffusion_coefficient =
            scratch
              .template compute_einstein_diffusion_coefficient<Component::n,
                                                               false>(q);
        if (prm::is_p_enabled)
          p_einstein_diffusion_coefficient =
            scratch
              .template compute_einstein_diffusion_coefficient<Component::p,
                                                               false>(q);

        if (prm::is_n_enabled)
          Jn = n0[q] * (scratch.mu_n_cell[q] * E0[q]) -
               (n_einstein_diffusion_coefficient * Wn0[q]);
        if (prm::is_p_enabled)
          Jp = -p0[q] * (scratch.mu_p_cell[q] * E0[q]) -
               (p_einstein_diffusion_coefficient * Wp0[q]);

        if (prm::thermodyn_eq)
          {
            const double KbT = Constants::KB * scratch.T_cell[q];
            thermodynamic_equilibrium_rhs =
              Constants::Q * (nv * exp((ev - Constants::Q * V0[q]) / KbT) -
                              nc * exp((Constants::Q * V0[q] - ec) / KbT));
          }

        for (unsigned int i = 0; i < dofs_per_component; ++i)
          {
            const unsigned int ii = scratch.enabled_component_indices[i];
            if (prm::is_V_enabled)
              {
                const dealii::Tensor<1, dim> q1 =
                  scratch.fe_values_cell[electric_field].value(ii, q);
                const double q1_div =
                  scratch.fe_values_cell[electric_field].divergence(ii, q);
                const double z1 =
                  scratch.fe_values_cell[electric_potential].value(ii, q);
                const dealii::Tensor<1, dim> z1_grad =
                  scratch.fe_values_cell[electric_potential].gradient(ii, q);

                scratch.cc_rhs[i] +=
                  (V0[q] * q1_div - E0[q] * q1 +
                   (scratch.epsilon_cell[q] * E0[q]) * z1_grad +
                   Constants::Q * c0[q] * z1) *
                  JxW;

                if (prm::thermodyn_eq)
                  scratch.cc_rhs[i] += thermodynamic_equilibrium_rhs * z1;
                else
                  scratch.cc_rhs[i] +=
                    Constants::Q * (-n0[q] + p0[q]) * z1 * JxW;
              }

            if (prm::is_n_enabled)
              {
                const dealii::Tensor<1, dim> q2 =
                  scratch.fe_values_cell[electron_displacement].value(ii, q);
                const double q2_div =
                  scratch.fe_values_cell[electron_displacement].divergence(ii,
                                                                           q);
                const double z2 =
                  scratch.fe_values_cell[electron_density].value(ii, q);
                const dealii::Tensor<1, dim> z2_grad =
                  scratch.fe_values_cell[electron_density].gradient(ii, q);

                scratch.cc_rhs[i] +=
                  (n0[q] * q2_div - Wn0[q] * q2 +
                   scratch.r_n_cell[q] / Constants::Q * z2 + Jn * z2_grad) *
                  JxW;
              }

            if (prm::is_p_enabled)
              {
                const dealii::Tensor<1, dim> q3 =
                  scratch.fe_values_cell[hole_displacement].value(ii, q);
                const double q3_div =
                  scratch.fe_values_cell[hole_displacement].divergence(ii, q);
                const double z3 =
                  scratch.fe_values_cell[hole_density].value(ii, q);
                const dealii::Tensor<1, dim> z3_grad =
                  scratch.fe_values_cell[hole_density].gradient(ii, q);

                scratch.cc_rhs[i] +=
                  (p0[q] * q3_div - Wp0[q] * q3 +
                   +scratch.r_p_cell[q] / Constants::Q * z3 + Jp * z3_grad) *
                  JxW;
              }
          }
      }
  }



  template <int dim>
  void
  NPSolver<dim>::prepare_data_on_face_quadrature_points(
    Ddhdg::NPSolver<dim>::ScratchData &scratch)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      scratch.face_quadrature_points[q] =
        scratch.fe_face_values_trace_restricted.quadrature_point(q);

    for (const auto c : Ddhdg::all_components())
      {
        const Displacement d = component2displacement(c);

        const FEValuesExtractors::Scalar c_extractor =
          this->get_component_extractor(c);
        const FEValuesExtractors::Vector d_extractor =
          this->get_displacement_extractor(d);
        const FEValuesExtractors::Scalar tr_c_extractor =
          this->get_trace_component_extractor(c);

        scratch.fe_face_values_cell[c_extractor].get_function_values(
          current_solution_cell, scratch.previous_c_face[c]);
        scratch.fe_face_values_cell[d_extractor].get_function_values(
          current_solution_cell, scratch.previous_d_face[c]);
        scratch.fe_face_values_trace[tr_c_extractor].get_function_values(
          current_solution_trace, scratch.previous_tr_c_face[c]);
      }

    if (this->is_enabled(Component::V))
      this->permittivity->compute_absolute_permittivity(
        scratch.face_quadrature_points, scratch.epsilon_face);

    if (this->is_enabled(Component::n))
      this->n_electron_mobility->compute_electron_mobility(
        scratch.face_quadrature_points, scratch.mu_n_face);

    if (this->is_enabled(Component::p))
      this->p_electron_mobility->compute_electron_mobility(
        scratch.face_quadrature_points, scratch.mu_p_face);

    if (this->is_enabled(Component::n) || this->is_enabled(Component::p))
      this->temperature->value_list(scratch.face_quadrature_points,
                                    scratch.T_face);
  }



  template <int dim>
  inline void
  NPSolver<dim>::copy_fe_values_on_scratch(
    Ddhdg::NPSolver<dim>::ScratchData &scratch,
    const unsigned int                 face,
    const unsigned int                 q)
  {
    for (const auto c : this->enabled_components)
      {
        const Displacement d = component2displacement(c);

        const FEValuesExtractors::Scalar c_extractor =
          this->get_component_extractor(c);
        const FEValuesExtractors::Vector d_extractor =
          this->get_displacement_extractor(d);

        auto &f  = scratch.d.at(c);
        auto &c_ = scratch.c.at(c);

        const unsigned int cell_dofs_on_face =
          scratch.fe_cell_support_on_face[face].size();
        for (unsigned int k = 0; k < cell_dofs_on_face; ++k)
          {
            const unsigned int component_dof_index =
              scratch.fe_cell_support_on_face[face][k];
            const unsigned int kk =
              scratch.enabled_component_indices[component_dof_index];
            f[k]  = scratch.fe_face_values_cell[d_extractor].value(kk, q);
            c_[k] = scratch.fe_face_values_cell[c_extractor].value(kk, q);
          }
      }
  }



  template <int dim>
  inline void
  NPSolver<dim>::copy_fe_values_for_trace(
    Ddhdg::NPSolver<dim>::ScratchData &scratch,
    const unsigned int                 face,
    const unsigned int                 q)
  {
    for (const auto c : this->enabled_components)
      {
        const FEValuesExtractors::Scalar extractor =
          this->get_trace_component_extractor(c, true);

        auto &tr_c = scratch.tr_c.at(c);

        const unsigned int dofs_on_face =
          scratch.fe_trace_support_on_face[face].size();
        for (unsigned int k = 0; k < dofs_on_face; ++k)
          {
            tr_c[k] = scratch.fe_face_values_trace_restricted[extractor].value(
              scratch.fe_trace_support_on_face[face][k], q);
          }
      }
  }



  template <int dim>
  template <typename prm>
  inline void
  NPSolver<dim>::assemble_ct_matrix(Ddhdg::NPSolver<dim>::ScratchData &scratch,
                                    const unsigned int                 face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    const unsigned int cell_dofs_on_face =
      scratch.fe_cell_support_on_face[face].size();
    const unsigned int trace_dofs_on_face =
      scratch.fe_trace_support_on_face[face].size();

    auto &q1   = scratch.d.at(Component::V);
    auto &z1   = scratch.c.at(Component::V);
    auto &tr_V = scratch.tr_c.at(Component::V);

    auto &q2   = scratch.d.at(Component::n);
    auto &z2   = scratch.c.at(Component::n);
    auto &tr_n = scratch.tr_c.at(Component::n);

    auto &q3   = scratch.d.at(Component::p);
    auto &z3   = scratch.c.at(Component::p);
    auto &tr_p = scratch.tr_c.at(Component::p);

    const double V_tau = this->parameters->tau.at(Component::V);
    const double n_tau = this->parameters->tau.at(Component::n);
    const double p_tau = this->parameters->tau.at(Component::p);

    double V_tau_stabilized = 0.;
    double n_tau_stabilized = 0.;
    double p_tau_stabilized = 0.;

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);
        copy_fe_values_for_trace(scratch, face, q);

        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        if (prm::is_V_enabled)
          V_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::V>(V_tau,
                                                                  normal,
                                                                  q);
        if (prm::is_n_enabled)
          n_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::n>(n_tau,
                                                                  normal,
                                                                  q);
        if (prm::is_p_enabled)
          p_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::p>(p_tau,
                                                                  normal,
                                                                  q);

        for (unsigned int i = 0; i < cell_dofs_on_face; ++i)
          {
            const unsigned int ii = scratch.fe_cell_support_on_face[face][i];
            for (unsigned int j = 0; j < trace_dofs_on_face; ++j)
              {
                const unsigned int jj =
                  scratch.fe_trace_support_on_face[face][j];

                // Integrals of trace functions using as test function
                // the restriction of cell test function on the border
                // i is the index of the test function
                if (prm::is_V_enabled)
                  scratch.ct_matrix(ii, jj) +=
                    (tr_V[j] * (q1[i] * normal) -
                     V_tau_stabilized * tr_V[j] * z1[i]) *
                    JxW;
                if (prm::is_n_enabled)
                  scratch.ct_matrix(ii, jj) +=
                    (tr_n[j] * (q2[i] * normal) +
                     n_tau_stabilized * (tr_n[j] * z2[i])) *
                    JxW;
                if (prm::is_p_enabled)
                  scratch.ct_matrix(ii, jj) +=
                    (tr_p[j] * (q3[i] * normal) +
                     p_tau_stabilized * (tr_p[j] * z3[i])) *
                    JxW;
              }
          }
      }
  }



  template <int dim>
  template <typename prm>
  inline void
  NPSolver<dim>::add_ct_matrix_terms_to_cc_rhs(
    Ddhdg::NPSolver<dim>::ScratchData &scratch,
    const unsigned int                 face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    const FEValuesExtractors::Vector electric_field =
      this->get_displacement_extractor(Displacement::E);
    const FEValuesExtractors::Scalar electric_potential =
      this->get_component_extractor(Component::V);
    const FEValuesExtractors::Vector electron_displacement =
      this->get_displacement_extractor(Displacement::Wn);
    const FEValuesExtractors::Scalar electron_density =
      this->get_component_extractor(Component::n);
    const FEValuesExtractors::Vector hole_displacement =
      this->get_displacement_extractor(Displacement::Wp);
    const FEValuesExtractors::Scalar hole_density =
      this->get_component_extractor(Component::p);

    const auto &tr_V0 = scratch.previous_tr_c_face.at(Component::V);
    const auto &tr_n0 = scratch.previous_tr_c_face.at(Component::n);
    const auto &tr_p0 = scratch.previous_tr_c_face.at(Component::p);

    const double V_tau = this->parameters->tau.at(Component::V);
    const double n_tau = this->parameters->tau.at(Component::n);
    const double p_tau = this->parameters->tau.at(Component::p);

    double V_tau_stabilized = 0.;
    double n_tau_stabilized = 0.;
    double p_tau_stabilized = 0.;

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        if (prm::is_V_enabled)
          V_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::V>(V_tau,
                                                                  normal,
                                                                  q);
        if (prm::is_n_enabled)
          n_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::n>(n_tau,
                                                                  normal,
                                                                  q);
        if (prm::is_p_enabled)
          p_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::p>(p_tau,
                                                                  normal,
                                                                  q);

        for (unsigned int i = 0;
             i < scratch.fe_cell_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii  = scratch.fe_cell_support_on_face[face][i];
            const unsigned int iii = scratch.enabled_component_indices[ii];

            if (prm::is_V_enabled)
              {
                const dealii::Tensor<1, dim> q1 =
                  scratch.fe_face_values_cell[electric_field].value(iii, q);
                const double z1 =
                  scratch.fe_face_values_cell[electric_potential].value(iii, q);

                scratch.cc_rhs(ii) += (-tr_V0[q] * (q1 * normal) +
                                       V_tau_stabilized * tr_V0[q] * z1) *
                                      JxW;
              }

            if (prm::is_n_enabled)
              {
                const dealii::Tensor<1, dim> q2 =
                  scratch.fe_face_values_cell[electron_displacement].value(iii,
                                                                           q);
                const double z2 =
                  scratch.fe_face_values_cell[electron_density].value(iii, q);

                scratch.cc_rhs(ii) += (-tr_n0[q] * (q2 * normal) -
                                       n_tau_stabilized * tr_n0[q] * z2) *
                                      JxW;
              }

            if (prm::is_p_enabled)
              {
                const dealii::Tensor<1, dim> q3 =
                  scratch.fe_face_values_cell[hole_displacement].value(iii, q);
                const double z3 =
                  scratch.fe_face_values_cell[hole_density].value(iii, q);

                scratch.cc_rhs(ii) += (-tr_p0[q] * (q3 * normal) -
                                       p_tau_stabilized * tr_p0[q] * z3) *
                                      JxW;
              }
          }
      }
  }



  template <int dim>
  template <typename prm, Component c>
  inline void
  NPSolver<dim>::assemble_tc_matrix(Ddhdg::NPSolver<dim>::ScratchData &scratch,
                                    const unsigned int                 face)
  {
    if (c != V and c != n and c != p)
      AssertThrow(false, InvalidComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    auto &c_ = scratch.c.at(c);
    auto &f  = scratch.d.at(c);
    auto &xi = scratch.tr_c.at(c);

    const auto &E = scratch.previous_d_face.at(Component::V);

    dealii::Tensor<1, dim> mu_n_times_previous_E;
    dealii::Tensor<1, dim> mu_p_times_previous_E;

    dealii::Tensor<2, dim> n_einstein_diffusion_coefficient;
    dealii::Tensor<2, dim> p_einstein_diffusion_coefficient;

    const std::vector<double> n0 = scratch.previous_c_face.at(Component::n);
    const std::vector<double> p0 = scratch.previous_c_face.at(Component::p);

    const double tau = this->parameters->tau.at(c);

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);
        copy_fe_values_for_trace(scratch, face, q);

        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        if (c == Component::n)
          {
            mu_n_times_previous_E = scratch.mu_n_face[q] * E[q];
            n_einstein_diffusion_coefficient =
              scratch
                .template compute_einstein_diffusion_coefficient<Component::n>(
                  q);
          }
        if (c == Component::p)
          {
            mu_p_times_previous_E = scratch.mu_p_face[q] * E[q];
            p_einstein_diffusion_coefficient =
              scratch
                .template compute_einstein_diffusion_coefficient<Component::p>(
                  q);
          }

        const double tau_stabilized =
          scratch.template compute_stabilized_tau<c>(tau, normal, q);

        const unsigned int trace_dofs_per_face =
          scratch.fe_trace_support_on_face[face].size();
        const unsigned int cell_dofs_per_face =
          scratch.fe_cell_support_on_face[face].size();

        for (unsigned int i = 0; i < trace_dofs_per_face; ++i)
          {
            const unsigned int ii = scratch.fe_trace_support_on_face[face][i];
            for (unsigned int j = 0; j < cell_dofs_per_face; ++j)
              {
                const unsigned int jj =
                  scratch.fe_cell_support_on_face[face][j];

                // Integrals of the cell functions restricted on the
                // border. The sign is reversed to be used in the
                // Schur complement (and therefore this is the
                // opposite of the right matrix that describes the
                // problem on this cell)
                if (c == V)
                  {
                    scratch.tc_matrix(ii, jj) -=
                      ((scratch.epsilon_face[q] * f[j]) * normal +
                       tau_stabilized * c_[j]) *
                      xi[i] * JxW;
                  }
                if (c == n)
                  {
                    const dealii::Tensor<1, dim> mu_n_times_E =
                      scratch.mu_n_face[q] * scratch.d[Component::V][j];

                    scratch.tc_matrix(ii, jj) -=
                      ((c_[j] * mu_n_times_previous_E + n0[q] * mu_n_times_E) *
                         normal -
                       (n_einstein_diffusion_coefficient * f[j]) * normal -
                       tau_stabilized * c_[j]) *
                      xi[i] * JxW;
                  }
                if (c == p)
                  {
                    const dealii::Tensor<1, dim> mu_p_times_E =
                      scratch.mu_p_face[q] * scratch.d[Component::V][j];

                    scratch.tc_matrix(ii, jj) -=
                      (-(c_[j] * mu_p_times_previous_E + p0[q] * mu_p_times_E) *
                         normal -
                       (p_einstein_diffusion_coefficient * f[j]) * normal -
                       tau_stabilized * c_[j]) *
                      xi[i] * JxW;
                  }
              }
          }
      }
  }



  template <int dim>
  template <typename prm, Component c>
  inline void
  NPSolver<dim>::add_tc_matrix_terms_to_tt_rhs(
    Ddhdg::NPSolver<dim>::ScratchData &scratch,
    Ddhdg::NPSolver<dim>::PerTaskData &task_data,
    const unsigned int                 face)
  {
    if (c != V and c != n and c != p)
      AssertThrow(false, InvalidComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    const FEValuesExtractors::Scalar tr_c_extractor =
      this->get_trace_component_extractor(c, true);

    auto &c0 = scratch.previous_c_face.at(c);
    auto &f0 = scratch.previous_d_face.at(c);

    const auto &E0 = scratch.previous_d_face[Component::V];

    double epsilon_times_previous_E_times_normal = 0.;
    double mu_n_times_previous_E_times_normal    = 0.;
    double mu_p_times_previous_E_times_normal    = 0.;

    dealii::Tensor<2, dim> n_einstein_diffusion_coefficient;
    dealii::Tensor<2, dim> p_einstein_diffusion_coefficient;

    const double tau = this->parameters->tau.at(c);

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        if (c == Component::V)
          {
            epsilon_times_previous_E_times_normal =
              scratch.epsilon_face[q] * E0[q] * normal;
          }
        if (c == Component::n)
          {
            mu_n_times_previous_E_times_normal =
              scratch.mu_n_face[q] * E0[q] * normal;
            n_einstein_diffusion_coefficient =
              scratch
                .template compute_einstein_diffusion_coefficient<Component::n>(
                  q);
          }
        if (c == Component::p)
          {
            mu_p_times_previous_E_times_normal =
              scratch.mu_p_face[q] * E0[q] * normal;
            p_einstein_diffusion_coefficient =
              scratch
                .template compute_einstein_diffusion_coefficient<Component::p>(
                  q);
          }

        const double tau_stabilized =
          scratch.template compute_stabilized_tau<c>(tau, normal, q);

        for (unsigned int i = 0;
             i < scratch.fe_trace_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii = scratch.fe_trace_support_on_face[face][i];
            const double       xi =
              scratch.fe_face_values_trace_restricted[tr_c_extractor].value(ii,
                                                                            q);
            if (c == V)
              {
                task_data.tt_vector[ii] +=
                  (-epsilon_times_previous_E_times_normal -
                   tau_stabilized * c0[q]) *
                  xi * JxW;
              }
            if (c == n)
              {
                task_data.tt_vector[ii] +=
                  (-c0[q] * mu_n_times_previous_E_times_normal +
                   n_einstein_diffusion_coefficient * f0[q] * normal +
                   tau_stabilized * c0[q]) *
                  xi * JxW;
              }
            if (c == p)
              {
                task_data.tt_vector[ii] +=
                  (c0[q] * mu_p_times_previous_E_times_normal +
                   p_einstein_diffusion_coefficient * f0[q] * normal +
                   tau_stabilized * c0[q]) *
                  xi * JxW;
              }
          }
      }
  }



  template <int dim>
  template <typename prm, Component c>
  inline void
  NPSolver<dim>::assemble_tt_matrix(
    Ddhdg::NPSolver<dim>::ScratchData &scratch,
    Ddhdg::NPSolver<dim>::PerTaskData &task_data,
    const unsigned int                 face)
  {
    if (c != V and c != n and c != p)
      AssertThrow(false, InvalidComponent());

    auto &tr_c = scratch.tr_c.at(c);
    auto &xi   = scratch.tr_c.at(c);

    const double tau  = this->parameters->tau.at(c);
    const double sign = (c == V) ? -1 : 1;

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();
    const unsigned int trace_dofs_per_face =
      scratch.fe_trace_support_on_face[face].size();

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);
        copy_fe_values_for_trace(scratch, face, q);

        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        const double tau_stabilized =
          scratch.template compute_stabilized_tau<c>(tau, normal, q);

        // Integrals of trace functions (both test and trial)
        for (unsigned int i = 0; i < trace_dofs_per_face; ++i)
          {
            const unsigned int ii = scratch.fe_trace_support_on_face[face][i];
            for (unsigned int j = 0; j < trace_dofs_per_face; ++j)
              {
                const unsigned int jj =
                  scratch.fe_trace_support_on_face[face][j];
                task_data.tt_matrix(ii, jj) +=
                  sign * tau_stabilized * tr_c[j] * xi[i] * JxW;
              }
          }
      }
  }



  template <int dim>
  template <typename prm, Component c>
  inline void
  NPSolver<dim>::add_tt_matrix_terms_to_tt_rhs(
    Ddhdg::NPSolver<dim>::ScratchData &scratch,
    Ddhdg::NPSolver<dim>::PerTaskData &task_data,
    const unsigned int                 face)
  {
    if (c != V and c != n and c != p)
      AssertThrow(false, InvalidComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();
    const unsigned int trace_dofs_per_face =
      scratch.fe_trace_support_on_face[face].size();

    const double tau  = this->parameters->tau.at(c);
    const double sign = (c == Component::V) ? -1 : 1;

    const FEValuesExtractors::Scalar tr_c_extractor =
      this->get_trace_component_extractor(c, true);

    const auto &tr_c0 = scratch.previous_tr_c_face.at(c);

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        const double tau_stabilized =
          scratch.template compute_stabilized_tau<c>(tau, normal, q);

        for (unsigned int i = 0; i < trace_dofs_per_face; ++i)
          {
            const unsigned int ii = scratch.fe_trace_support_on_face[face][i];
            const double       xi =
              scratch.fe_face_values_trace_restricted[tr_c_extractor].value(ii,
                                                                            q);
            task_data.tt_vector[ii] +=
              -sign * tau_stabilized * tr_c0[q] * xi * JxW;
          }
      }
  }



  template <int dim>
  template <Ddhdg::Component c>
  inline void
  NPSolver<dim>::apply_dbc_on_face(
    Ddhdg::NPSolver<dim>::ScratchData &           scratch,
    Ddhdg::NPSolver<dim>::PerTaskData &           task_data,
    const Ddhdg::DirichletBoundaryCondition<dim> &dbc,
    unsigned int                                  face)
  {
    if (c != V and c != n and c != p)
      Assert(false, InvalidComponent());

    auto &tr_c  = scratch.tr_c.at(c);
    auto &tr_c0 = scratch.previous_tr_c_face.at(c);

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();
    const unsigned int trace_dofs_per_face =
      scratch.fe_trace_support_on_face[face].size();

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_for_trace(scratch, face, q);

        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);

        const Point<dim> quadrature_point =
          scratch.fe_face_values_trace_restricted.quadrature_point(q);
        const double dbc_value = dbc.evaluate(quadrature_point) - tr_c0[q];

        for (unsigned int i = 0; i < trace_dofs_per_face; ++i)
          {
            const unsigned int ii = scratch.fe_trace_support_on_face[face][i];
            for (unsigned int j = 0; j < trace_dofs_per_face; ++j)
              {
                const unsigned int jj =
                  scratch.fe_trace_support_on_face[face][j];
                task_data.tt_matrix(ii, jj) += tr_c[i] * tr_c[j] * JxW;
              }

            task_data.tt_vector[ii] += tr_c[i] * dbc_value * JxW;
          }
      }
  }



  template <int dim>
  template <Ddhdg::Component c>
  void
  NPSolver<dim>::apply_nbc_on_face(
    Ddhdg::NPSolver<dim>::ScratchData &         scratch,
    Ddhdg::NPSolver<dim>::PerTaskData &         task_data,
    const Ddhdg::NeumannBoundaryCondition<dim> &nbc,
    unsigned int                                face)
  {
    if (c != V and c != n and c != p)
      Assert(false, InvalidComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();
    const unsigned int trace_dofs_per_face =
      scratch.fe_trace_support_on_face[face].size();

    auto &tr_c = scratch.tr_c.at(c);

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_for_trace(scratch, face, q);

        const double     JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Point<dim> quadrature_point =
          scratch.fe_face_values_trace_restricted.quadrature_point(q);
        const double nbc_value =
          (c == V ? nbc.evaluate(quadrature_point) :
                    nbc.evaluate(quadrature_point) / Constants::Q);

        for (unsigned int i = 0; i < trace_dofs_per_face; ++i)
          {
            const unsigned int ii = scratch.fe_trace_support_on_face[face][i];
            task_data.tt_vector(ii) += tr_c[i] * nbc_value * JxW;
          }
      }
  }


  template <int dim>
  template <typename prm>
  inline void
  NPSolver<dim>::add_border_products_to_cc_matrix(
    Ddhdg::NPSolver<dim>::ScratchData &scratch,
    const unsigned int                 face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();
    const unsigned int cell_dofs_per_face =
      scratch.fe_cell_support_on_face[face].size();

    auto &E  = scratch.d.at(Component::V);
    auto &V  = scratch.c.at(Component::V);
    auto &Wn = scratch.d.at(Component::n);
    auto &n  = scratch.c.at(Component::n);
    auto &Wp = scratch.d.at(Component::p);
    auto &p  = scratch.c.at(Component::p);

    auto &z1 = scratch.c.at(Component::V);
    auto &z2 = scratch.c.at(Component::n);
    auto &z3 = scratch.c.at(Component::p);

    const auto  n0 = scratch.previous_c_face.at(Component::n);
    const auto  p0 = scratch.previous_c_face.at(Component::p);
    const auto &E0 = scratch.previous_d_face.at(Component::V);

    const double V_tau = this->parameters->tau.at(Component::V);
    const double n_tau = this->parameters->tau.at(Component::n);
    const double p_tau = this->parameters->tau.at(Component::p);

    double V_tau_stabilized = 0;
    double n_tau_stabilized = 0;
    double p_tau_stabilized = 0;

    dealii::Tensor<1, dim> mu_n_times_previous_E;
    dealii::Tensor<1, dim> mu_p_times_previous_E;

    dealii::Tensor<2, dim> n_einstein_diffusion_coefficient;
    dealii::Tensor<2, dim> p_einstein_diffusion_coefficient;

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);

        // Integrals on the border of the cell of the cell functions
        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        if (prm::is_n_enabled)
          mu_n_times_previous_E = scratch.mu_n_face[q] * E0[q];

        if (prm::is_p_enabled)
          mu_p_times_previous_E = scratch.mu_p_face[q] * E0[q];

        if (prm::is_n_enabled)
          n_einstein_diffusion_coefficient =
            scratch
              .template compute_einstein_diffusion_coefficient<Component::n>(q);

        if (prm::is_p_enabled)
          p_einstein_diffusion_coefficient =
            scratch
              .template compute_einstein_diffusion_coefficient<Component::p>(q);

        if (prm::is_V_enabled)
          V_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::V>(V_tau,
                                                                  normal,
                                                                  q);
        if (prm::is_n_enabled)
          n_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::n>(n_tau,
                                                                  normal,
                                                                  q);
        if (prm::is_p_enabled)
          p_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::p>(p_tau,
                                                                  normal,
                                                                  q);

        for (unsigned int i = 0; i < cell_dofs_per_face; ++i)
          {
            const unsigned int ii = scratch.fe_cell_support_on_face[face][i];
            for (unsigned int j = 0; j < cell_dofs_per_face; ++j)
              {
                const unsigned int jj =
                  scratch.fe_cell_support_on_face[face][j];

                if (prm::is_V_enabled)
                  scratch.cc_matrix(ii, jj) +=
                    (scratch.epsilon_face[q] * E[j] * normal * z1[i] +
                     V_tau_stabilized * V[j] * z1[i]) *
                    JxW;
                if (prm::is_n_enabled)
                  scratch.cc_matrix(ii, jj) +=
                    ((n[j] * mu_n_times_previous_E +
                      scratch.mu_n_face[q] * E[j] * n0[q]) *
                       normal * z2[i] -
                     n_einstein_diffusion_coefficient * Wn[j] * normal * z2[i] -
                     n_tau_stabilized * n[j] * z2[i]) *
                    JxW;
                if (prm::is_p_enabled)
                  scratch.cc_matrix(ii, jj) +=
                    (-(p[j] * mu_p_times_previous_E +
                       scratch.mu_p_face[q] * E[j] * p0[q]) *
                       normal * z3[i] -
                     p_einstein_diffusion_coefficient * Wp[j] * normal * z3[i] -
                     p_tau_stabilized * p[j] * z3[i]) *
                    JxW;
              }
          }
      }
  }



  template <int dim>
  template <typename prm>
  inline void
  NPSolver<dim>::add_border_products_to_cc_rhs(
    Ddhdg::NPSolver<dim>::ScratchData &scratch,
    const unsigned int                 face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    const FEValuesExtractors::Scalar electric_potential =
      this->get_component_extractor(Component::V);
    const FEValuesExtractors::Scalar electron_density =
      this->get_component_extractor(Component::n);
    const FEValuesExtractors::Scalar hole_density =
      this->get_component_extractor(Component::p);

    auto &V0  = scratch.previous_c_face.at(Component::V);
    auto &E0  = scratch.previous_d_face.at(Component::V);
    auto &n0  = scratch.previous_c_face.at(Component::n);
    auto &Wn0 = scratch.previous_d_face.at(Component::n);
    auto &p0  = scratch.previous_c_face.at(Component::p);
    auto &Wp0 = scratch.previous_d_face.at(Component::p);

    const double V_tau = this->parameters->tau.at(Component::V);
    const double n_tau = this->parameters->tau.at(Component::n);
    const double p_tau = this->parameters->tau.at(Component::p);

    double V_tau_stabilized = 0.;
    double n_tau_stabilized = 0.;
    double p_tau_stabilized = 0.;

    double epsilon_times_E0_times_normal = 0.;

    dealii::Tensor<2, dim> n_einstein_diffusion_coefficient;
    dealii::Tensor<2, dim> p_einstein_diffusion_coefficient;

    double Jn_flux = 0.;
    double Jp_flux = 0.;

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        if (prm::is_V_enabled)
          epsilon_times_E0_times_normal =
            (scratch.epsilon_face[q] * E0[q]) * normal;

        if (prm::is_n_enabled)
          n_einstein_diffusion_coefficient =
            scratch
              .template compute_einstein_diffusion_coefficient<Component::n>(q);

        if (prm::is_p_enabled)
          p_einstein_diffusion_coefficient =
            scratch
              .template compute_einstein_diffusion_coefficient<Component::p>(q);

        if (prm::is_n_enabled)
          Jn_flux = (n0[q] * (scratch.mu_n_face[q] * E0[q]) -
                     (n_einstein_diffusion_coefficient * Wn0[q])) *
                    normal;

        if (prm::is_p_enabled)
          Jp_flux = (-p0[q] * (scratch.mu_p_face[q] * E0[q]) -
                     (p_einstein_diffusion_coefficient * Wp0[q])) *
                    normal;

        if (prm::is_V_enabled)
          V_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::V>(V_tau,
                                                                  normal,
                                                                  q);
        if (prm::is_n_enabled)
          n_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::n>(n_tau,
                                                                  normal,
                                                                  q);
        if (prm::is_p_enabled)
          p_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::p>(p_tau,
                                                                  normal,
                                                                  q);

        for (unsigned int i = 0;
             i < scratch.fe_cell_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii  = scratch.fe_cell_support_on_face[face][i];
            const unsigned int iii = scratch.enabled_component_indices[ii];

            if (prm::is_V_enabled)
              {
                const double z1 =
                  scratch.fe_face_values_cell[electric_potential].value(iii, q);
                scratch.cc_rhs[ii] += (-epsilon_times_E0_times_normal * z1 -
                                       V_tau_stabilized * V0[q] * z1) *
                                      JxW;
              }
            if (prm::is_n_enabled)
              {
                const double z2 =
                  scratch.fe_face_values_cell[electron_density].value(iii, q);
                scratch.cc_rhs[ii] +=
                  (-Jn_flux * z2 + n_tau_stabilized * n0[q] * z2) * JxW;
              }
            if (prm::is_p_enabled)
              {
                const double z3 =
                  scratch.fe_face_values_cell[hole_density].value(iii, q);
                scratch.cc_rhs[ii] +=
                  (-Jp_flux * z3 + p_tau_stabilized * p0[q] * z3) * JxW;
              }
          }
      }
  }



  template <int dim>
  template <typename prm>
  inline void
  NPSolver<dim>::add_trace_terms_to_cc_rhs(
    Ddhdg::NPSolver<dim>::ScratchData &scratch,
    const unsigned int                 face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    double tau_stabilized;

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);

        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        for (const auto c : this->enabled_components)
          {
            auto &f                    = scratch.d.at(c);
            auto &c_                   = scratch.c.at(c);
            auto &tr_c_solution_values = scratch.tr_c_solution_values.at(c);

            const double tau  = this->parameters->tau.at(c);
            const double sign = (c == Component::V) ? 1 : -1;

            switch (c)
              {
                case Component::V:
                  tau_stabilized =
                    scratch.template compute_stabilized_tau<Component::V>(
                      tau, normal, q);
                  break;
                case Component::n:
                  tau_stabilized =
                    scratch.template compute_stabilized_tau<Component::n>(
                      tau, normal, q);
                  break;
                case Component::p:
                  tau_stabilized =
                    scratch.template compute_stabilized_tau<Component::p>(
                      tau, normal, q);
                  break;
                default:
                  Assert(false, InvalidComponent());
                  tau_stabilized = 1.;
              }

            for (unsigned int i = 0;
                 i < scratch.fe_cell_support_on_face[face].size();
                 ++i)
              {
                const unsigned int ii =
                  scratch.fe_cell_support_on_face[face][i];
                scratch.cc_rhs(ii) +=
                  (-f[i] * normal + sign * tau_stabilized * c_[i]) *
                  tr_c_solution_values[q] * JxW;
              }
          }
      }
  }



  template <int dim>
  template <typename prm, Component c>
  void
  NPSolver<dim>::assemble_flux_conditions(
    ScratchData &            scratch,
    PerTaskData &            task_data,
    const bool               has_dirichlet_conditions,
    const bool               has_neumann_conditions,
    const types::boundary_id face_boundary_id,
    const unsigned int       face)
  {
    if (has_dirichlet_conditions)
      {
        const DirichletBoundaryCondition<dim> dbc =
          (prm::thermodyn_eq) ?
            DirichletBoundaryCondition<dim>(
              std::make_shared<dealii::ZeroFunction<dim>>(), Component::V) :
            this->boundary_handler->get_dirichlet_conditions_for_id(
              face_boundary_id, c);
        this->apply_dbc_on_face<c>(scratch, task_data, dbc, face);
      }
    else
      {
        this->assemble_tc_matrix<prm, c>(scratch, face);
        this->add_tc_matrix_terms_to_tt_rhs<prm, c>(scratch, task_data, face);
        this->assemble_tt_matrix<prm, c>(scratch, task_data, face);
        this->add_tt_matrix_terms_to_tt_rhs<prm, c>(scratch, task_data, face);
        if (has_neumann_conditions)
          {
            const NeumannBoundaryCondition<dim> nbc =
              (prm::thermodyn_eq) ?
                NeumannBoundaryCondition<dim>(
                  std::make_shared<dealii::ZeroFunction<dim>>(), c) :
                this->boundary_handler->get_neumann_conditions_for_id(
                  face_boundary_id, c);
            this->apply_nbc_on_face<c>(scratch, task_data, nbc, face);
          }
      }
  }



  template <int dim>
  template <typename prm>
  void
  NPSolver<dim>::assemble_flux_conditions_wrapper(
    const Component                         c,
    ScratchData &                           scratch,
    PerTaskData &                           task_data,
    const std::map<Ddhdg::Component, bool> &has_dirichlet_conditions,
    const std::map<Ddhdg::Component, bool> &has_neumann_conditions,
    const types::boundary_id                face_boundary_id,
    const unsigned int                      face)
  {
    switch (c)
      {
        case Component::V:
          assemble_flux_conditions<prm, Component::V>(
            scratch,
            task_data,
            has_dirichlet_conditions.at(Component::V),
            has_neumann_conditions.at(Component::V),
            face_boundary_id,
            face);
          break;
        case Component::n:
          assemble_flux_conditions<prm, Component::n>(
            scratch,
            task_data,
            has_dirichlet_conditions.at(Component::n),
            has_neumann_conditions.at(Component::n),
            face_boundary_id,
            face);
          break;
        case Component::p:
          assemble_flux_conditions<prm, Component::p>(
            scratch,
            task_data,
            has_dirichlet_conditions.at(Component::p),
            has_neumann_conditions.at(Component::p),
            face_boundary_id,
            face);
          break;
        default:
          Assert(false, InvalidComponent());
          break;
      }
  }



  template <int dim>
  template <typename prm>
  void
  NPSolver<dim>::assemble_system_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData &                                         scratch,
    PerTaskData &                                         task_data)
  {
    typename DoFHandler<dim>::active_cell_iterator loc_cell(&(*triangulation),
                                                            cell->level(),
                                                            cell->index(),
                                                            &dof_handler_cell);
    typename DoFHandler<dim>::active_cell_iterator trace_cell(
      &(*triangulation), cell->level(), cell->index(), &dof_handler_trace);

    // Reset every value that could be related to the previous cell
    scratch.cc_matrix = 0;
    scratch.cc_rhs    = 0;
    if (!task_data.trace_reconstruct)
      {
        scratch.ct_matrix   = 0;
        scratch.tc_matrix   = 0;
        task_data.tt_matrix = 0;
        task_data.tt_vector = 0;
      }
    scratch.fe_values_cell.reinit(loc_cell);

    // We use the following function to copy every value that we need from the
    // fe_values objects into the scratch. This also compute the physical
    // parameters (like permittivity or the recombination term) on the
    // quadrature points of the cell
    this->prepare_data_on_cell_quadrature_points(scratch);

    // Integrals on the overall cell
    // This function computes every L2 product that we need to compute in
    // order to assemble the matrix for the current cell.
    this->add_cell_products_to_cc_matrix<prm>(scratch);

    // This function, instead, computes the l2 products that are needed for
    // the right hand term
    this->add_cell_products_to_cc_rhs<prm>(scratch);

    // Now we must perform the L2 product on the boundary, i.e. for each face
    // of the cell
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
         ++face)
      {
        scratch.fe_face_values_cell.reinit(loc_cell, face);
        scratch.fe_face_values_trace.reinit(trace_cell, face);
        scratch.fe_face_values_trace_restricted.reinit(cell, face);

        // If we have already solved the system for the trace, copy the values
        // of the global solution in the scratch that stores the values for
        // this cell
        if (task_data.trace_reconstruct)
          {
            for (const auto c : Ddhdg::all_components())
              {
                const FEValuesExtractors::Scalar extractor =
                  this->get_trace_component_extractor(c);
                scratch.fe_face_values_trace[extractor].get_function_values(
                  this->update_trace, scratch.tr_c_solution_values[c]);
              }
          }

        // Now I create some maps to store, for each component, if there is a
        // boundary condition for it
        const types::boundary_id face_boundary_id =
          cell->face(face)->boundary_id();
        std::map<Ddhdg::Component, bool> has_dirichlet_conditions;
        std::map<Ddhdg::Component, bool> has_neumann_conditions;

        // Now we populate the previous maps
        if (prm::thermodyn_eq)
          {
            if (cell->face(face)->at_boundary())
              {
                has_dirichlet_conditions.insert({Component::V, true});
                has_dirichlet_conditions.insert({Component::n, false});
                has_dirichlet_conditions.insert({Component::p, false});

                has_neumann_conditions.insert({Component::V, false});
                has_neumann_conditions.insert({Component::n, true});
                has_neumann_conditions.insert({Component::p, true});
              }
            else
              for (const auto c : all_components())
                {
                  has_dirichlet_conditions.insert({c, false});
                  has_neumann_conditions.insert({c, false});
                }
          }
        else
          {
            for (const auto c : this->enabled_components)
              {
                has_dirichlet_conditions.insert(
                  {c,
                   this->boundary_handler->has_dirichlet_boundary_conditions(
                     face_boundary_id, c)});
                has_neumann_conditions.insert(
                  {c,
                   this->boundary_handler->has_neumann_boundary_conditions(
                     face_boundary_id, c)});
              }
          }

        // Before assembling the other parts of the matrix, we need the values
        // of epsilon, mu and D_n on the quadrature points of the current
        // face. Moreover, we want to populate the scratch object with the
        // values of the solution at the previous step. All this jobs are
        // accomplished by the next function
        this->prepare_data_on_face_quadrature_points(scratch);

        // The following function adds some terms to cc_rhs. Usually, all the
        // functions are coupled (like assemble_matrix_XXX and
        // add_matrix_XXX_terms_to_l_rhs) because the first function is the
        // function that assembles the matrix and the second one is the
        // function that adds the corresponding terms to cc_rhs (i.e. the
        // product of the matrix with the previous solution, so that the
        // solution of the system is the update from the previous solution to
        // the new one). The following function is the only exception: indeed,
        // the terms that it generates are always needed while the terms that
        // its corresponding function generates are useful only if we are not
        // reconstructing the solution from the trace
        this->add_ct_matrix_terms_to_cc_rhs<prm>(scratch, face);

        // Assembly the other matrices (the cc_matrix has been assembled
        // calling the add_cell_products_to_cc_matrix method)
        if (!task_data.trace_reconstruct)
          {
            this->assemble_ct_matrix<prm>(scratch, face);
            for (const Component c : this->enabled_components)
              {
                this->assemble_flux_conditions_wrapper<prm>(
                  c,
                  scratch,
                  task_data,
                  has_dirichlet_conditions,
                  has_neumann_conditions,
                  face_boundary_id,
                  face);
              }
          }

        // These are the last terms of the ll matrix, the ones that are
        // generated by L2 products only on the boundary of the cell
        this->add_border_products_to_cc_matrix<prm>(scratch, face);
        this->add_border_products_to_cc_rhs<prm>(scratch, face);

        if (task_data.trace_reconstruct)
          this->add_trace_terms_to_cc_rhs<prm>(scratch, face);
      }

    inversion_mutex.lock();
    scratch.cc_matrix.gauss_jordan();
    inversion_mutex.unlock();

    if (!task_data.trace_reconstruct)
      {
        scratch.tc_matrix.mmult(scratch.tmp_matrix, scratch.cc_matrix);
        scratch.tmp_matrix.vmult_add(task_data.tt_vector, scratch.cc_rhs);
        scratch.tmp_matrix.mmult(task_data.tt_matrix, scratch.ct_matrix, true);
        cell->get_dof_indices(task_data.dof_indices);
      }
    else
      {
        scratch.cc_matrix.vmult(scratch.restricted_tmp_rhs, scratch.cc_rhs);
        scratch.tmp_rhs = 0;
        for (unsigned int i = 0; i < scratch.enabled_component_indices.size();
             i++)
          scratch.tmp_rhs[scratch.enabled_component_indices[i]] =
            scratch.restricted_tmp_rhs[i];
        loc_cell->set_dof_values(scratch.tmp_rhs, update_cell);
      }
  }



  template <int dim>
  void
  NPSolver<dim>::copy_local_to_global(const PerTaskData &data)
  {
    if (!data.trace_reconstruct)
      this->constraints.distribute_local_to_global(data.tt_matrix,
                                                   data.tt_vector,
                                                   data.dof_indices,
                                                   this->system_matrix,
                                                   this->system_rhs);
  }



  template <int dim>
  void
  NPSolver<dim>::solve_linear_problem()
  {
    std::cout << "    RHS norm   : " << this->system_rhs.l2_norm() << std::endl
              << "    Matrix norm: " << this->system_matrix.linfty_norm()
              << std::endl;

    SolverControl solver_control(this->system_matrix.m() * 10,
                                 1e-10 * this->system_rhs.l2_norm());

    if (parameters->iterative_linear_solver)
      {
        SolverGMRES<> linear_solver(solver_control);
        linear_solver.solve(this->system_matrix,
                            this->system_solution,
                            this->system_rhs,
                            PreconditionIdentity());
        std::cout << "    Number of GMRES iterations: "
                  << solver_control.last_step() << std::endl;
      }
    else
      {
        SparseDirectUMFPACK Ainv;
        Ainv.initialize(this->system_matrix);
        Ainv.vmult(this->system_solution, this->system_rhs);
      }
    constraints.distribute(this->system_solution);
  }


  template <int dim>
  void
  NPSolver<dim>::build_restricted_to_trace_dof_map()
  {
    const unsigned int trace_dofs = this->fe_trace->dofs_per_cell;
    const unsigned int trace_restricted_dofs =
      this->fe_trace_restricted->dofs_per_cell;

    this->restricted_to_trace_dof_map.clear();
    this->restricted_to_trace_dof_map.resize(
      this->dof_handler_trace_restricted.n_dofs());

    std::map<unsigned int, Component> component_index;
    for (const Component c : Ddhdg::all_components())
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



  template <int dim>
  void
  NPSolver<dim>::copy_restricted_to_trace()
  {
    const unsigned int restricted_global_dofs =
      this->dof_handler_trace_restricted.n_dofs();

    for (unsigned int i = 0; i < restricted_global_dofs; i++)
      {
        const unsigned int j  = this->restricted_to_trace_dof_map[i];
        this->update_trace[j] = this->system_solution[i];
      }
  }



  template <int dim>
  NonlinearIterationResults
  NPSolver<dim>::private_run(const double absolute_tol,
                             const double relative_tol,
                             const int    max_number_of_iterations,
                             const bool   compute_thermodynamic_equilibrium)
  {
    if (!this->initialized)
      setup_overall_system();

    bool   convergence_reached = false;
    int    step                = 0;
    double update_cell_norm    = 0.;
    double current_solution_cell_norm;

    for (step = 1;
         step <= max_number_of_iterations || max_number_of_iterations < 0;
         step++)
      {
        std::cout << "Computing step number " << step << std::endl;

        this->update_trace    = 0;
        this->update_cell     = 0;
        this->system_matrix   = 0;
        this->system_rhs      = 0;
        this->system_solution = 0;

        if (parameters->multithreading)
          assemble_system_multithreaded(false,
                                        compute_thermodynamic_equilibrium);
        else
          assemble_system(false, compute_thermodynamic_equilibrium);

        solve_linear_problem();
        this->copy_restricted_to_trace();

        if (parameters->multithreading)
          assemble_system_multithreaded(true,
                                        compute_thermodynamic_equilibrium);
        else
          assemble_system(true, compute_thermodynamic_equilibrium);

        update_cell_norm           = this->update_cell.linfty_norm();
        current_solution_cell_norm = this->current_solution_cell.linfty_norm();

        std::cout << "Difference in norm compared to the previous step: "
                  << update_cell_norm << std::endl;

        this->current_solution_trace += this->update_trace;
        this->current_solution_cell += this->update_cell;

        if (update_cell_norm < absolute_tol)
          {
            std::cout << "Update is smaller than absolute tolerance. "
                      << "CONVERGENCE REACHED" << std::endl;
            convergence_reached = true;
            break;
          }
        if (update_cell_norm < relative_tol * current_solution_cell_norm)
          {
            std::cout << "Update is smaller than relative tolerance. "
                      << "CONVERGENCE REACHED" << std::endl;
            convergence_reached = true;
            break;
          }
      }

    return NonlinearIterationResults(convergence_reached,
                                     step,
                                     update_cell_norm);
  }



  template <int dim>
  NonlinearIterationResults
  NPSolver<dim>::run(const double absolute_tol,
                     const double relative_tol,
                     const int    max_number_of_iterations)
  {
    return this->private_run(absolute_tol,
                             relative_tol,
                             max_number_of_iterations,
                             false);
  }



  template <int dim>
  NonlinearIterationResults
  NPSolver<dim>::run()
  {
    return this->run(
      this->parameters->nonlinear_solver_absolute_tolerance,
      this->parameters->nonlinear_solver_relative_tolerance,
      this->parameters->nonlinear_solver_max_number_of_iterations);
  }


  template <int dim>
  NonlinearIterationResults
  NPSolver<dim>::compute_thermodynamic_equilibrium(
    const double absolute_tol,
    const double relative_tol,
    const int    max_number_of_iterations)
  {
    std::map<Component, bool> current_active_components;
    for (Component c : all_components())
      {
        current_active_components[c] = this->is_enabled(c);
      }

    this->set_enabled_components(true, false, false);

    NonlinearIterationResults iterations = this->private_run(
      absolute_tol, relative_tol, max_number_of_iterations, true);

    this->set_enabled_components(current_active_components[Component::V],
                                 current_active_components[Component::n],
                                 current_active_components[Component::p]);

    return iterations;
  }



  template <int dim>
  NonlinearIterationResults
  NPSolver<dim>::compute_thermodynamic_equilibrium()
  {
    return this->compute_thermodynamic_equilibrium(
      this->parameters->nonlinear_solver_absolute_tolerance,
      this->parameters->nonlinear_solver_relative_tolerance,
      this->parameters->nonlinear_solver_max_number_of_iterations);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                                        expected_solution,
    const Ddhdg::Component              c,
    const dealii::VectorTools::NormType norm) const
  {
    Assert(expected_solution->n_components == 1, FunctionMustBeScalar());
    Vector<double> difference_per_cell(triangulation->n_active_cells());

    unsigned int component_index = get_component_index(c) * (1 + dim) + dim;

    const unsigned int n_of_components = (dim + 1) * all_components().size();
    const auto         component_selection =
      dealii::ComponentSelectFunction<dim>(component_index, n_of_components);

    std::map<unsigned int, const std::shared_ptr<const dealii::Function<dim>>>
      c_map;
    c_map.insert({component_index, expected_solution});
    FunctionByComponents<dim> expected_solution_multidim =
      FunctionByComponents<dim>(n_of_components, c_map);

    VectorTools::integrate_difference(
      this->dof_handler_cell,
      this->current_solution_cell,
      expected_solution_multidim,
      difference_per_cell,
      QGauss<dim>(this->get_number_of_quadrature_points()),
      norm,
      &component_selection);

    const double global_error =
      VectorTools::compute_global_error(*(this->triangulation),
                                        difference_per_cell,
                                        norm);
    return global_error;
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                                        expected_solution,
    const Ddhdg::Displacement           d,
    const dealii::VectorTools::NormType norm) const
  {
    Assert(expected_solution->n_components == dim,
           FunctionMustHaveDimComponents());
    Vector<double> difference_per_cell(triangulation->n_active_cells());

    Component c = displacement2component(d);

    unsigned int       component_index = get_component_index(c);
    const unsigned int n_of_components = (dim + 1) * all_components().size();

    const std::pair<const unsigned int, const unsigned int> selection_interval =
      {component_index * (dim + 1), component_index * (dim + 1) + dim};
    const auto component_selection =
      dealii::ComponentSelectFunction<dim>(selection_interval, n_of_components);

    std::map<unsigned int, const std::shared_ptr<const dealii::Function<dim>>>
      c_map;
    for (unsigned int i = 0; i < dim; i++)
      {
        c_map.insert(
          {component_index * (dim + 1) + i,
           std::make_shared<ComponentFunction<dim>>(expected_solution, i)});
      }
    FunctionByComponents<dim> expected_solution_multidim =
      FunctionByComponents<dim>(n_of_components, c_map);

    VectorTools::integrate_difference(
      this->dof_handler_cell,
      this->current_solution_cell,
      expected_solution_multidim,
      difference_per_cell,
      QGauss<dim>(this->get_number_of_quadrature_points()),
      norm,
      &component_selection);

    const double global_error =
      VectorTools::compute_global_error(*(this->triangulation),
                                        difference_per_cell,
                                        norm);
    return global_error;
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_error_on_trace(
    const std::shared_ptr<const dealii::Function<dim, double>>
                                        expected_solution,
    const Ddhdg::Component              c,
    const dealii::VectorTools::NormType norm) const
  {
    Assert(expected_solution->n_components == 1, FunctionMustBeScalar());
    Assert(norm == VectorTools::L2_norm || norm == VectorTools::Linfty_norm,
           ExcNotImplemented())

      const unsigned int           c_index = get_component_index(c);
    std::map<unsigned int, double> difference_per_face;

    const QGauss<dim - 1> face_quadrature_formula(
      this->get_number_of_quadrature_points());
    const UpdateFlags flags(update_values | update_quadrature_points |
                            update_JxW_values);

    const unsigned int n_face_q_points = face_quadrature_formula.size();
    const unsigned int dofs_per_cell   = this->fe_trace->dofs_per_cell;

    FEFaceValues<dim> fe_face_trace_values(*(this->fe_trace),
                                           face_quadrature_formula,
                                           flags);

    const FEValuesExtractors::Scalar c_extractor =
      this->get_trace_component_extractor(c);

    // Check which dofs are related to our component
    std::vector<unsigned int> on_current_component;
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const unsigned int current_index =
          this->fe_trace->system_to_block_index(i).first;
        if (current_index == c_index)
          on_current_component.push_back(i);
      }
    const unsigned int dofs_per_component = on_current_component.size();

    // Check which component related dofs are relevant for each face
    std::vector<std::vector<unsigned int>> component_support_on_face(
      GeometryInfo<dim>::faces_per_cell);
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
         ++face)
      for (unsigned int i = 0; i < dofs_per_component; ++i)
        {
          const unsigned int dof_index = on_current_component[i];
          if (this->fe_trace->has_support_on_face(dof_index, face))
            component_support_on_face[face].push_back(i);
        }
    const unsigned int dofs_per_face_on_component =
      component_support_on_face[0].size();

    Vector<double> current_solution_node_values(dofs_per_cell);

    std::vector<Point<dim>> face_quadrature_points(n_face_q_points);
    std::vector<double>     expected_solution_on_q(n_face_q_points);
    std::vector<double>     current_solution_on_q(n_face_q_points);

    for (const auto &cell : dof_handler_trace.active_cell_iterators())
      {
        cell->get_dof_values(this->current_solution_trace,
                             current_solution_node_values);

        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            unsigned int face_uid = cell->face_index(face);

            // If this face has already been examined from another cell, skip
            // it
            if (difference_per_face.find(face_uid) != difference_per_face.end())
              continue;

            fe_face_trace_values.reinit(cell, face);

            // Compute the position of the quadrature points
            for (unsigned int q = 0; q < n_face_q_points; ++q)
              face_quadrature_points[q] =
                fe_face_trace_values.quadrature_point(q);

            // Compute the value of the current solution on the quadrature
            // points
            for (unsigned int q = 0; q < n_face_q_points; ++q)
              {
                double solution_on_q = 0;
                for (unsigned int i = 0; i < dofs_per_face_on_component; i++)
                  {
                    const unsigned int ii =
                      on_current_component[component_support_on_face[face][i]];
                    solution_on_q +=
                      fe_face_trace_values[c_extractor].value(ii, q) *
                      current_solution_node_values[ii];
                  }
                current_solution_on_q[q] = solution_on_q;
              }

            // Compute the value of the expected solution on the quadrature
            // points
            expected_solution->value_list(face_quadrature_points,
                                          expected_solution_on_q);

            // Now it's time to perform the integration
            double difference_norm = 0;
            if (norm == VectorTools::L2_norm)
              {
                for (unsigned int q = 0; q < n_face_q_points; ++q)
                  {
                    const double JxW = fe_face_trace_values.JxW(q);
                    double       difference_on_q =
                      current_solution_on_q[q] - expected_solution_on_q[q];
                    difference_norm += difference_on_q * difference_on_q * JxW;
                  }
              }
            if (norm == VectorTools::Linfty_norm)
              {
                for (unsigned int q = 0; q < n_face_q_points; ++q)
                  {
                    double difference_on_q =
                      current_solution_on_q[q] - expected_solution_on_q[q];
                    double abs_difference = (difference_on_q > 0) ?
                                              difference_on_q :
                                              -difference_on_q;
                    if (abs_difference > difference_norm)
                      difference_norm = abs_difference;
                  }
              }

            // We save the result in the map for each face
            difference_per_face.insert({face_uid, difference_norm});
          }
      }

    // Now we need to aggregate data
    switch (norm)
      {
          case VectorTools::L2_norm: {
            double norm_sum = 0;
            for (const auto &nrm : difference_per_face)
              {
                norm_sum += nrm.second;
              }
            return sqrt(norm_sum);
          }
          case VectorTools::Linfty_norm: {
            double max_value = 0;
            for (const auto &nrm : difference_per_face)
              {
                max_value = (max_value > nrm.second) ? max_value : nrm.second;
              }
            return max_value;
          }
        default:
          break;
      }
    return 1e99;
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                           expected_solution,
    const Ddhdg::Component c) const
  {
    return this->estimate_error(expected_solution,
                                c,
                                dealii::VectorTools::L2_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                              expected_solution,
    const Ddhdg::Displacement d) const
  {
    return this->estimate_error(expected_solution,
                                d,
                                dealii::VectorTools::L2_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error_on_trace(
    const std::shared_ptr<const dealii::Function<dim, double>>
                           expected_solution,
    const Ddhdg::Component c) const
  {
    return this->estimate_error_on_trace(expected_solution,
                                         c,
                                         dealii::VectorTools::L2_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                           expected_solution,
    const Ddhdg::Component c) const
  {
    return this->estimate_error(expected_solution,
                                c,
                                dealii::VectorTools::H1_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                              expected_solution,
    const Ddhdg::Displacement d) const
  {
    return this->estimate_error(expected_solution,
                                d,
                                dealii::VectorTools::H1_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                           expected_solution,
    const Ddhdg::Component c) const
  {
    return this->estimate_error(expected_solution,
                                c,
                                dealii::VectorTools::Linfty_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                              expected_solution,
    const Ddhdg::Displacement d) const
  {
    return this->estimate_error(expected_solution,
                                d,
                                dealii::VectorTools::Linfty_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error_on_trace(
    const std::shared_ptr<const dealii::Function<dim, double>>
                           expected_solution,
    const Ddhdg::Component c) const
  {
    return this->estimate_error_on_trace(expected_solution,
                                         c,
                                         dealii::VectorTools::Linfty_norm);
  }



  template <int dim>
  void
  NPSolver<dim>::output_results(const std::string &solution_filename,
                                const bool         save_update) const
  {
    std::ofstream output(solution_filename);
    DataOut<dim>  data_out;
    // DataOutBase::VtkFlags flags;
    // flags.write_higher_order_cells = true;
    // data_out.set_flags(flags);

    const std::set<Component> components      = all_components();
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

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      component_interpretation(
        n_of_components * (dim + 1),
        DataComponentInterpretation::component_is_part_of_vector);
    for (unsigned int i = 0; i < n_of_components; i++)
      component_interpretation[(dim + 1) * i + dim] =
        DataComponentInterpretation::component_is_scalar;

    data_out.add_data_vector(this->dof_handler_cell,
                             this->current_solution_cell,
                             names,
                             component_interpretation);

    if (save_update)
      data_out.add_data_vector(this->dof_handler_cell,
                               this->update_cell,
                               update_names,
                               component_interpretation);

    data_out.build_patches(StaticMappingQ1<dim>::mapping,
                           fe_trace_restricted->degree,
                           DataOut<dim>::curved_inner_cells);
    data_out.write_vtk(output);
  }



  template <>
  void
  NPSolver<1>::output_results(const std::string &solution_filename,
                              const std::string &trace_filename,
                              const bool         save_update) const
  {
    (void)solution_filename;
    (void)trace_filename;
    (void)save_update;
    AssertThrow(false, NoTraceIn1D());
  }



  template <int dim>
  void
  NPSolver<dim>::output_results(const std::string &solution_filename,
                                const std::string &trace_filename,
                                const bool         save_update) const
  {
    output_results(solution_filename, save_update);

    std::ofstream     face_output(trace_filename);
    DataOutFaces<dim> data_out_face(false);

    const std::set<Component> components      = all_components();
    const unsigned int        n_of_components = components.size();

    std::vector<std::string> face_names;
    for (const Component c : components)
      face_names.push_back(get_component_name(c));

    std::vector<std::string> update_face_names;
    for (const auto &n : face_names)
      update_face_names.push_back(n + "_updates");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      face_component_type(n_of_components,
                          DataComponentInterpretation::component_is_scalar);
    data_out_face.add_data_vector(this->dof_handler_trace,
                                  this->current_solution_trace,
                                  face_names,
                                  face_component_type);

    if (save_update)
      data_out_face.add_data_vector(this->dof_handler_trace,
                                    this->update_trace,
                                    update_face_names,
                                    face_component_type);

    data_out_face.build_patches(fe_trace_restricted->degree);
    data_out_face.write_vtk(face_output);
  }



  template <int dim>
  void
  NPSolver<dim>::print_convergence_table(
    std::shared_ptr<Ddhdg::ConvergenceTable>     error_table,
    std::shared_ptr<const dealii::Function<dim>> expected_V_solution,
    std::shared_ptr<const dealii::Function<dim>> expected_n_solution,
    std::shared_ptr<const dealii::Function<dim>> expected_p_solution,
    unsigned int                                 n_cycles,
    unsigned int                                 initial_refinements)
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
                                  initial_refinements);
  }



  template <int dim>
  void
  NPSolver<dim>::print_convergence_table(
    std::shared_ptr<Ddhdg::ConvergenceTable>     error_table,
    std::shared_ptr<const dealii::Function<dim>> expected_V_solution,
    std::shared_ptr<const dealii::Function<dim>> expected_n_solution,
    std::shared_ptr<const dealii::Function<dim>> expected_p_solution,
    std::shared_ptr<const dealii::Function<dim>> initial_V_function,
    std::shared_ptr<const dealii::Function<dim>> initial_n_function,
    std::shared_ptr<const dealii::Function<dim>> initial_p_function,
    unsigned int                                 n_cycles,
    unsigned int                                 initial_refinements)
  {
    this->refine_grid(initial_refinements);

    std::map<unsigned int, const std::shared_ptr<const dealii::Function<dim>>>
      components;
    components.insert({dim, expected_V_solution});
    components.insert({2 * dim + 1, expected_n_solution});
    components.insert({3 * dim + 2, expected_p_solution});
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
          this->refine_grid(1);
      }
    error_table->output_table(std::cout);
  }


  template class NPSolver<1>;
  template class NPSolver<2>;
  template class NPSolver<3>;

} // namespace Ddhdg
