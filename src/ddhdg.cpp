#include "ddhdg.h"

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

namespace Ddhdg
{
  template <int dim>
  std::unique_ptr<dealii::Triangulation<dim>>
  Solver<dim>::copy_triangulation(
    const std::shared_ptr<const dealii::Triangulation<dim>> triangulation)
  {
    std::unique_ptr<dealii::Triangulation<dim>> new_triangulation =
      std::make_unique<dealii::Triangulation<dim>>();
    new_triangulation->copy_triangulation(*triangulation);
    return new_triangulation;
  }



  template <int dim>
  std::map<Component, std::vector<double>>
  Solver<dim>::ScratchData::initialize_double_map_on_components(
    const unsigned int n)
  {
    std::map<Component, std::vector<double>> m;
    for (const auto c : Ddhdg::AllComponents)
      {
        m.insert({c, std::vector<double>(n)});
      }
    return m;
  }



  template <int dim>
  std::map<Component, std::vector<Tensor<1, dim>>>
  Solver<dim>::ScratchData::initialize_tensor_map_on_components(
    const unsigned int n)
  {
    std::map<Component, std::vector<Tensor<1, dim>>> m;
    for (const auto c : Ddhdg::AllComponents)
      {
        m.insert({c, std::vector<Tensor<1, dim>>(n)});
      }
    return m;
  }



  template <int dim>
  std::map<Component, std::vector<double>>
  Solver<dim>::ScratchData::initialize_double_map_on_n_and_p(
    const unsigned int k)
  {
    std::map<Component, std::vector<double>> m;
    m.insert({Component::n, std::vector<double>(k)});
    m.insert({Component::p, std::vector<double>(k)});
    return m;
  }



  template <int dim>
  FESystem<dim>
  Solver<dim>::generate_fe_system(
    const std::map<Component, unsigned int> &degree,
    const bool                               local)
  {
    std::vector<const FiniteElement<dim> *> fe_systems;
    std::vector<unsigned int>               multiplicities;

    if (local)
      for (std::pair<Component, unsigned int> c_degree : degree)
        {
          fe_systems.push_back(new FE_DGQ<dim>(c_degree.second));
          fe_systems.push_back(new FE_DGQ<dim>(c_degree.second));
          multiplicities.push_back(dim);
          multiplicities.push_back(1);
        }
    else
      for (std::pair<Component, unsigned int> c_degree : degree)
        {
          fe_systems.push_back(new FE_FaceQ<dim>(c_degree.second));
          multiplicities.push_back(1);
        }

    const FESystem<dim> output(fe_systems, multiplicities);

    for (const FiniteElement<dim> *fe : fe_systems)
      delete (fe);

    return output;
  }



  template <int dim>
  Solver<dim>::Solver(const std::shared_ptr<const Problem<dim>>     problem,
                      const std::shared_ptr<const SolverParameters> parameters)
    : triangulation(copy_triangulation(problem->triangulation))
    , permittivity(problem->permittivity)
    , n_electron_mobility(problem->n_electron_mobility)
    , n_recombination_term(problem->n_recombination_term)
    , p_electron_mobility(problem->p_electron_mobility)
    , p_recombination_term(problem->p_recombination_term)
    , temperature(problem->temperature)
    , doping(problem->doping)
    , boundary_handler(problem->boundary_handler)
    , parameters(std::make_unique<SolverParameters>(*parameters))
    , fe_local(generate_fe_system(parameters->degree, true))
    , dof_handler_local(*triangulation)
    , fe(generate_fe_system(parameters->degree, false))
    , dof_handler(*triangulation)
  {}



  template <int dim>
  void
  Solver<dim>::setup_system()
  {
    dof_handler_local.distribute_dofs(fe_local);
    dof_handler.distribute_dofs(fe);

    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    current_solution.reinit(dof_handler.n_dofs());
    update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    current_solution_local.reinit(dof_handler_local.n_dofs());
    update_local.reinit(dof_handler_local.n_dofs());

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    {
      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      sparsity_pattern.copy_from(dsp);
    }
    system_matrix.reinit(sparsity_pattern);

    this->initialized = true;
  }



  template <int dim>
  dealii::ComponentMask
  Solver<dim>::get_component_mask(const Component component)
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
          Assert(false, UnknownComponent());
          break;
      }
    return mask;
  }



  template <int dim>
  dealii::ComponentMask
  Solver<dim>::get_component_mask(const Displacement displacement)
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
          Assert(false, UnknownDisplacement());
          break;
      }
    return mask;
  }



  template <int dim>
  dealii::ComponentMask
  Solver<dim>::get_trace_component_mask(const Component component)
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
          Assert(false, UnknownComponent());
          break;
      }
    return mask;
  }



  template <int dim>
  std::shared_ptr<dealii::Function<dim>>
  Solver<dim>::extend_function_on_all_components(
    const std::shared_ptr<const dealii::Function<dim>> f,
    const Component                                    c)
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
          Assert(false, UnknownComponent());
          break;
      }
    std::shared_ptr<dealii::Function<dim>> function_extended =
      std::make_shared<FunctionByComponents<dim>>(3 * (dim + 1), f_components);
    return function_extended;
  }



  template <int dim>
  std::shared_ptr<dealii::Function<dim>>
  Solver<dim>::extend_function_on_all_components(
    const std::shared_ptr<const dealii::Function<dim>> f,
    const Displacement                                 d)
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
          Assert(false, UnknownDisplacement());
          break;
      }
    std::shared_ptr<dealii::Function<dim>> function_extended =
      std::make_shared<FunctionByComponents<dim>>(3 * (dim + 1), f_components);
    return function_extended;
  }



  template <int dim>
  std::shared_ptr<dealii::Function<dim>>
  Solver<dim>::extend_function_on_all_trace_components(
    const std::shared_ptr<const dealii::Function<dim>> f,
    const Component                                    c)
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
          Assert(false, UnknownComponent());
          break;
      }
    std::shared_ptr<dealii::Function<dim>> function_extended =
      std::make_shared<FunctionByComponents<dim>>(3, f_components);
    return function_extended;
  }



  template <int dim>
  void
  Solver<dim>::set_component(
    const Component                                    c,
    const std::shared_ptr<const dealii::Function<dim>> c_function)
  {
    if (!this->initialized)
      this->setup_system();

    const Displacement f = component2displacement(c);

    auto c_function_extended =
      this->extend_function_on_all_components(c_function, c);
    auto c_grad     = std::make_shared<Gradient<dim>>(c_function);
    auto f_function = std::make_shared<Opposite<dim>>(c_grad);
    auto f_function_extended =
      this->extend_function_on_all_components(f_function, f);

    dealii::VectorTools::interpolate(this->dof_handler_local,
                                     *c_function_extended,
                                     this->current_solution_local,
                                     this->get_component_mask(c));

    dealii::VectorTools::interpolate(this->dof_handler_local,
                                     *f_function_extended,
                                     this->current_solution_local,
                                     this->get_component_mask(f));

    if (dim == 2 || dim == 3)
      {
        auto c_function_trace_extended =
          this->extend_function_on_all_trace_components(c_function, c);
        dealii::VectorTools::interpolate(this->dof_handler,
                                         *c_function_trace_extended,
                                         this->current_solution,
                                         this->get_trace_component_mask(c));
      }
    else if (dim == 1)
      {
        const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);
        const UpdateFlags     flags(update_values | update_quadrature_points);

        FEFaceValues<dim>  fe_face_trace_values(this->fe,
                                               face_quadrature_formula,
                                               flags);
        const unsigned int n_face_q_points =
          fe_face_trace_values.get_quadrature().size();
        Assert(n_face_q_points == 1, ExcDimensionMismatch(n_face_q_points, 1));

        const unsigned int dofs_per_cell =
          fe_face_trace_values.get_fe().dofs_per_cell;
        Assert(dofs_per_cell == 6, ExcDimensionMismatch(dofs_per_cell, 6));

        Vector<double> local_values(dofs_per_cell);

        std::vector<Point<dim>> face_quadrature_points(n_face_q_points);
        dealii::FEValuesExtractors::Scalar extractor =
          this->get_trace_component_extractor(c);
        unsigned int nonzero_shape_functions = 0;
        double       current_value;
        double       function_value;

        for (const auto &cell : this->dof_handler.active_cell_iterators())
          {
            cell->get_dof_values(this->current_solution, local_values);
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
                Assert(
                  nonzero_shape_functions > 0,
                  ExcMessage(
                    "No shape function found that is different from zero on "
                    "the current node"));
                Assert(
                  nonzero_shape_functions == 1,
                  ExcMessage(
                    "More than one shape function found that is different from "
                    "zero on the current node"));
              }
            cell->set_dof_values(local_values, this->current_solution);
          }
      }
  }



  template <int dim>
  void
  Solver<dim>::set_current_solution(
    const std::shared_ptr<const dealii::Function<dim>> V_function,
    const std::shared_ptr<const dealii::Function<dim>> n_function,
    const std::shared_ptr<const dealii::Function<dim>> p_function,
    const bool                                         use_projection)
  {
    if (!this->initialized)
      this->setup_system();

    if (!use_projection)
      {
        this->set_component(Component::V, V_function);
        this->set_component(Component::n, n_function);
        this->set_component(Component::p, p_function);
        return;
      }

    const QGauss<dim>     quadrature_formula(fe.degree + 1);
    const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    // In this scope, the code takes care of project the right solution in the
    // FEM space of the cells. After that, we will project the solution on the
    // trace. I split the code in two parts because, in this way, it is possible
    // to handle the situations in which the space of the cell is different from
    // the space of the trace (for example, for HDG(A))
    {
      // Create sparsity pattern for the cells (currently, we have only the one
      // for the traces)
      AffineConstraints<double> projection_constraints;
      projection_constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler_local,
                                              projection_constraints);
      projection_constraints.close();
      SparsityPattern projection_sparsity_pattern;
      {
        DynamicSparsityPattern dsp(dof_handler_local.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler_local,
                                        dsp,
                                        constraints,
                                        false);
        projection_sparsity_pattern.copy_from(dsp);
      }

      const UpdateFlags local_flags(update_values | update_gradients |
                                    update_JxW_values |
                                    update_quadrature_points);
      const UpdateFlags flags(update_values | update_normal_vectors |
                              update_quadrature_points | update_JxW_values);

      FEValues<dim>     fe_values_local(this->fe_local,
                                    quadrature_formula,
                                    local_flags);
      FEFaceValues<dim> fe_face_values(this->fe_local,
                                       face_quadrature_formula,
                                       flags);

      const unsigned int n_q_points = fe_values_local.get_quadrature().size();
      const unsigned int n_face_q_points =
        fe_face_values.get_quadrature().size();

      const unsigned int loc_dofs_per_cell =
        fe_values_local.get_fe().dofs_per_cell;

      const FEValuesExtractors::Vector electric_field =
        this->get_displacement_extractor(Displacement::E);
      const FEValuesExtractors::Scalar electric_potential =
        this->get_component_extractor(Component::V);
      const FEValuesExtractors::Vector electron_displacement =
        this->get_displacement_extractor(Displacement::Wn);
      const FEValuesExtractors::Scalar electron_density =
        this->get_component_extractor(Component::n);

      SparseMatrix<double> projection_matrix;
      projection_matrix.reinit(projection_sparsity_pattern);
      Vector<double> projection_rhs;
      projection_rhs.reinit(dof_handler_local.n_dofs());

      FullMatrix<double> local_matrix(loc_dofs_per_cell, loc_dofs_per_cell);
      Vector<double>     local_residual(loc_dofs_per_cell);
      std::vector<types::global_dof_index> dof_indices(loc_dofs_per_cell);

      // Temporary buffer for the values of the local base function on a
      // quadrature point
      std::vector<double>         V(loc_dofs_per_cell);
      std::vector<Tensor<1, dim>> E(loc_dofs_per_cell);
      std::vector<double>         E_div(loc_dofs_per_cell);
      std::vector<double>         n(loc_dofs_per_cell);
      std::vector<Tensor<1, dim>> Wn(loc_dofs_per_cell);
      std::vector<double>         Wn_div(loc_dofs_per_cell);

      std::vector<Point<dim>> cell_quadrature_points(n_q_points);
      std::vector<Point<dim>> face_quadrature_points(n_face_q_points);

      std::vector<double> evaluated_v(n_q_points);
      std::vector<double> evaluated_n(n_q_points);

      std::vector<double> evaluated_v_face(n_face_q_points);
      std::vector<double> evaluated_n_face(n_face_q_points);

      for (const auto &cell : this->dof_handler_local.active_cell_iterators())
        {
          local_matrix   = 0.;
          local_residual = 0.;

          fe_values_local.reinit(cell);

          cell->get_dof_indices(dof_indices);

          // Get the position of the quadrature points
          for (unsigned int q = 0; q < n_q_points; ++q)
            cell_quadrature_points[q] = fe_values_local.quadrature_point(q);

          // Evaluated the analytic functions over the quadrature points
          V_function->value_list(cell_quadrature_points, evaluated_v);
          n_function->value_list(cell_quadrature_points, evaluated_n);

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              // Copy data of the shape function
              for (unsigned int k = 0; k < loc_dofs_per_cell; ++k)
                {
                  V[k]     = fe_values_local[electric_potential].value(k, q);
                  E[k]     = fe_values_local[electric_field].value(k, q);
                  E_div[k] = fe_values_local[electric_field].divergence(k, q);
                  n[k]     = fe_values_local[electron_density].value(k, q);
                  Wn[k]    = fe_values_local[electron_displacement].value(k, q);
                  Wn_div[k] =
                    fe_values_local[electron_displacement].divergence(k, q);
                }

              const double JxW = fe_values_local.JxW(q);

              for (unsigned int i = 0; i < loc_dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < loc_dofs_per_cell; ++j)
                    {
                      local_matrix(i, j) += (V[j] * V[i] + E[i] * E[j] +
                                             n[i] * n[j] + Wn[i] * Wn[j]) *
                                            JxW;
                    }
                  local_residual[i] += (evaluated_v[q] * (V[i] + E_div[i]) +
                                        evaluated_n[q] * (n[i] + Wn_div[i])) *
                                       JxW;
                }
            }
          for (unsigned int face_number = 0;
               face_number < GeometryInfo<dim>::faces_per_cell;
               ++face_number)
            {
              fe_face_values.reinit(cell, face_number);

              for (unsigned int q = 0; q < n_face_q_points; ++q)
                face_quadrature_points[q] = fe_face_values.quadrature_point(q);

              V_function->value_list(face_quadrature_points, evaluated_v_face);
              n_function->value_list(face_quadrature_points, evaluated_n_face);

              for (unsigned int q = 0; q < n_face_q_points; ++q)
                face_quadrature_points[q] = fe_face_values.quadrature_point(q);

              for (unsigned int q = 0; q < n_face_q_points; ++q)
                {
                  const double JxW    = fe_face_values.JxW(q);
                  const auto   normal = fe_face_values.normal_vector(q);

                  for (unsigned int k = 0; k < loc_dofs_per_cell; ++k)
                    {
                      const auto E_face =
                        fe_face_values[electric_field].value(k, q);
                      const auto W_face =
                        fe_face_values[electron_displacement].value(k, q);
                      local_residual[k] +=
                        (-evaluated_v_face[q] * (E_face * normal) -
                         evaluated_n_face[q] * (W_face * normal)) *
                        JxW;
                    }
                }
            }
          projection_constraints.distribute_local_to_global(local_matrix,
                                                            local_residual,
                                                            dof_indices,
                                                            projection_matrix,
                                                            projection_rhs);
        }
      SolverControl solver_control(projection_matrix.m() * 10,
                                   1e-10 * projection_rhs.l2_norm());
      SolverGMRES<> linear_solver(solver_control);
      linear_solver.solve(projection_matrix,
                          current_solution_local,
                          projection_rhs,
                          PreconditionIdentity());
    }

    // This is the part for the trace
    {
      const UpdateFlags flags(update_values | update_normal_vectors |
                              update_quadrature_points | update_JxW_values);

      FEFaceValues<dim>  fe_face_trace_values(this->fe,
                                             face_quadrature_formula,
                                             flags);
      const unsigned int n_face_q_points =
        fe_face_trace_values.get_quadrature().size();

      const unsigned int dofs_per_cell =
        fe_face_trace_values.get_fe().dofs_per_cell;

      const FEValuesExtractors::Scalar electric_potential =
        this->get_trace_component_extractor(Component::V);
      const FEValuesExtractors::Scalar electron_density =
        this->get_trace_component_extractor(Component::n);

      std::vector<double> V(dofs_per_cell);
      std::vector<double> n(dofs_per_cell);

      std::vector<Point<dim>> face_quadrature_points(n_face_q_points);

      std::vector<double> evaluated_v_face(n_face_q_points);
      std::vector<double> evaluated_n_face(n_face_q_points);

      FullMatrix<double> local_trace_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double>     local_trace_residual(dofs_per_cell);
      std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          local_trace_matrix   = 0;
          local_trace_residual = 0;
          cell->get_dof_indices(dof_indices);
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              fe_face_trace_values.reinit(cell, face);
              for (unsigned int q = 0; q < n_face_q_points; ++q)
                face_quadrature_points[q] =
                  fe_face_trace_values.quadrature_point(q);

              V_function->value_list(face_quadrature_points, evaluated_v_face);
              n_function->value_list(face_quadrature_points, evaluated_n_face);

              for (unsigned int q = 0; q < n_face_q_points; ++q)
                {
                  // Copy data of the shape function
                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                      V[k] =
                        fe_face_trace_values[electric_potential].value(k, q);
                      n[k] = fe_face_trace_values[electron_density].value(k, q);
                    }

                  const double JxW = fe_face_trace_values.JxW(q);

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          local_trace_matrix(i, j) +=
                            (V[j] * V[i] + n[i] * n[j]) * JxW;
                        }
                      local_trace_residual[i] += (evaluated_v_face[q] * V[i] +
                                                  evaluated_n_face[q] * n[i]) *
                                                 JxW;
                    }
                }
            }
          this->constraints.distribute_local_to_global(local_trace_matrix,
                                                       local_trace_residual,
                                                       dof_indices,
                                                       system_matrix,
                                                       system_rhs);
        }
      SolverControl solver_control(system_matrix.m() * 10,
                                   1e-10 * system_rhs.l2_norm());
      SolverGMRES<> linear_solver(solver_control);
      linear_solver.solve(system_matrix,
                          current_solution,
                          system_rhs,
                          PreconditionIdentity());
    }
  }


  template <int dim>
  dealii::FEValuesExtractors::Scalar
  Solver<dim>::get_component_extractor(const Component component)
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
          Assert(false, UnknownComponent());
          break;
      }
    return extractor;
  }



  template <int dim>
  dealii::FEValuesExtractors::Vector
  Solver<dim>::get_displacement_extractor(const Displacement displacement)
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
          Assert(false, UnknownDisplacement());
          break;
      }
    return extractor;
  }



  template <int dim>
  dealii::FEValuesExtractors::Scalar
  Solver<dim>::get_trace_component_extractor(const Component component)
  {
    dealii::FEValuesExtractors::Scalar extractor;
    switch (component)
      {
        case Component::V:
          extractor = dealii::FEValuesExtractors::Scalar(0);
          break;
        case Component::n:
          extractor = dealii::FEValuesExtractors::Scalar(1);
          break;
        case Component::p:
          extractor = dealii::FEValuesExtractors::Scalar(2);
          break;
        default:
          Assert(false, UnknownComponent());
          break;
      }
    return extractor;
  }



  template <int dim>
  void
  Solver<dim>::assemble_system_multithreaded(bool trace_reconstruct)
  {
    Assert(this->initialized, dealii::ExcNotInitialized());

    const QGauss<dim>     quadrature_formula(fe.degree + 1);
    const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    const UpdateFlags local_flags(update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points);
    const UpdateFlags local_face_flags(update_values);
    const UpdateFlags flags(update_values | update_normal_vectors |
                            update_quadrature_points | update_JxW_values);
    PerTaskData       task_data(fe.dofs_per_cell, trace_reconstruct);
    ScratchData       scratch(fe,
                        fe_local,
                        quadrature_formula,
                        face_quadrature_formula,
                        local_flags,
                        local_face_flags,
                        flags);
    WorkStream::run(dof_handler.begin_active(),
                    dof_handler.end(),
                    *this,
                    &Solver<dim>::assemble_system_one_cell,
                    &Solver<dim>::copy_local_to_global,
                    scratch,
                    task_data);
  }

  template <int dim>
  void
  Solver<dim>::assemble_system(bool trace_reconstruct)
  {
    Assert(this->initialized, dealii::ExcNotInitialized());

    const QGauss<dim>     quadrature_formula(fe.degree + 1);
    const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    const UpdateFlags local_flags(update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points);
    const UpdateFlags local_face_flags(update_values);
    const UpdateFlags flags(update_values | update_normal_vectors |
                            update_quadrature_points | update_JxW_values);
    PerTaskData       task_data(fe.dofs_per_cell, trace_reconstruct);
    ScratchData       scratch(fe,
                        fe_local,
                        quadrature_formula,
                        face_quadrature_formula,
                        local_flags,
                        local_face_flags,
                        flags);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        assemble_system_one_cell(cell, scratch, task_data);
        copy_local_to_global(task_data);
      }
  }



  template <int dim>
  void
  Solver<dim>::prepare_data_on_cell_quadrature_points(
    Ddhdg::Solver<dim>::ScratchData &scratch)
  {
    const unsigned int n_q_points =
      scratch.fe_values_local.get_quadrature().size();

    // Copy the values of the previous solution regarding the previous cell in
    // the scratch
    for (const auto c : Ddhdg::AllComponents)
      {
        const Displacement               d = component2displacement(c);
        const FEValuesExtractors::Scalar c_extractor =
          this->get_component_extractor(c);
        const FEValuesExtractors::Vector d_extractor =
          this->get_displacement_extractor(d);

        scratch.fe_values_local[c_extractor].get_function_values(
          current_solution_local, scratch.previous_c_cell[c]);
        scratch.fe_values_local[d_extractor].get_function_values(
          current_solution_local, scratch.previous_f_cell[c]);
      }

    // Get the position of the quadrature points
    for (unsigned int q = 0; q < n_q_points; ++q)
      scratch.cell_quadrature_points[q] =
        scratch.fe_values_local.quadrature_point(q);

    // Compute the value of epsilon
    this->permittivity->compute_absolute_permittivity(
      scratch.cell_quadrature_points, scratch.epsilon_cell);

    // Compute the value of mu
    this->n_electron_mobility->compute_electron_mobility(
      scratch.cell_quadrature_points, scratch.mu_n_cell);
    this->p_electron_mobility->compute_electron_mobility(
      scratch.cell_quadrature_points, scratch.mu_p_cell);

    // Compute the value of T
    this->temperature->value_list(scratch.cell_quadrature_points,
                                  scratch.T_cell);

    // Compute the value of the doping
    this->doping->value_list(scratch.cell_quadrature_points,
                             scratch.doping_cell);

    // Compute the value of the recombination term and its derivative respect to
    // n and p
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



  template <int dim>
  void
  Solver<dim>::add_cell_products_to_ll_matrix(
    Ddhdg::Solver<dim>::ScratchData &scratch)
  {
    const unsigned int n_q_points =
      scratch.fe_values_local.get_quadrature().size();
    const unsigned int loc_dofs_per_cell =
      scratch.fe_values_local.get_fe().dofs_per_cell;

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
    // vector). These may seem useless, but the make the code that assemble the
    // matrix a lot more understandable
    auto &E = scratch.f.at(Component::V);
    auto &V = scratch.c.at(Component::V);

    auto &Wn = scratch.f.at(Component::n);
    auto &n  = scratch.c.at(Component::n);

    auto &Wp = scratch.f.at(Component::p);
    auto &p  = scratch.c.at(Component::p);

    auto &q1      = scratch.f.at(Component::V);
    auto &q1_div  = scratch.f_div.at(Component::V);
    auto &z1      = scratch.c.at(Component::V);
    auto &z1_grad = scratch.c_grad.at(Component::V);

    auto &q2      = scratch.f.at(Component::n);
    auto &q2_div  = scratch.f_div.at(Component::n);
    auto &z2      = scratch.c.at(Component::n);
    auto &z2_grad = scratch.c_grad.at(Component::n);

    auto &q3      = scratch.f.at(Component::p);
    auto &q3_div  = scratch.f_div.at(Component::p);
    auto &z3      = scratch.c.at(Component::p);
    auto &z3_grad = scratch.c_grad.at(Component::p);

    const auto &n0 = scratch.previous_c_cell.at(Component::n);
    const auto &p0 = scratch.previous_c_cell.at(Component::p);
    const auto &E0 = scratch.previous_f_cell.at(Component::V);

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double JxW = scratch.fe_values_local.JxW(q);
        for (unsigned int k = 0; k < loc_dofs_per_cell; ++k)
          {
            q1[k] = scratch.fe_values_local[electric_field].value(k, q);
            q1_div[k] =
              scratch.fe_values_local[electric_field].divergence(k, q);
            z1[k] = scratch.fe_values_local[electric_potential].value(k, q);
            z1_grad[k] =
              scratch.fe_values_local[electric_potential].gradient(k, q);

            q2[k] = scratch.fe_values_local[electron_displacement].value(k, q);
            q2_div[k] =
              scratch.fe_values_local[electron_displacement].divergence(k, q);
            z2[k] = scratch.fe_values_local[electron_density].value(k, q);
            z2_grad[k] =
              scratch.fe_values_local[electron_density].gradient(k, q);

            q3[k] = scratch.fe_values_local[hole_displacement].value(k, q);
            q3_div[k] =
              scratch.fe_values_local[hole_displacement].divergence(k, q);
            z3[k]      = scratch.fe_values_local[hole_density].value(k, q);
            z3_grad[k] = scratch.fe_values_local[hole_density].gradient(k, q);
          }

        const dealii::Tensor<1, dim> mu_n_times_previous_E =
          scratch.mu_n_cell[q] * E0[q];

        const dealii::Tensor<1, dim> mu_p_times_previous_E =
          scratch.mu_p_cell[q] * E0[q];

        const dealii::Tensor<2, dim> n_einstein_diffusion_coefficient =
          scratch.template compute_einstein_diffusion_coefficient<Component::n,
                                                                  false>(q);
        const dealii::Tensor<2, dim> p_einstein_diffusion_coefficient =
          scratch.template compute_einstein_diffusion_coefficient<Component::p,
                                                                  false>(q);

        const double dr_n_n = scratch.dr_n_cell.at(Component::n)[q];
        const double dr_n_p = scratch.dr_n_cell.at(Component::p)[q];

        const double dr_p_n = scratch.dr_p_cell.at(Component::n)[q];
        const double dr_p_p = scratch.dr_p_cell.at(Component::p)[q];

        for (unsigned int i = 0; i < loc_dofs_per_cell; ++i)
          for (unsigned int j = 0; j < loc_dofs_per_cell; ++j)
            {
              scratch.ll_matrix(i, j) +=
                (-V[j] * q1_div[i] + E[j] * q1[i] -
                 (scratch.epsilon_cell[q] * E[j]) * z1_grad[i] +
                 Constants::Q * (n[j] - p[j]) * z1[i] - n[j] * q2_div[i] +
                 Wn[j] * q2[i] - n[j] * (mu_n_times_previous_E * z2_grad[i]) +
                 n0[q] * ((scratch.mu_n_cell[q] * E[j]) * z2_grad[i]) +
                 (n_einstein_diffusion_coefficient * Wn[j]) * z2_grad[i] -
                 (dr_n_n * n[j] + dr_n_p * p[j]) * z2[i] / Constants::Q -
                 p[j] * q3_div[i] + Wp[j] * q3[i] +
                 p[j] * (mu_p_times_previous_E * z3_grad[i]) -
                 p0[q] * ((scratch.mu_p_cell[q] * E[j]) * z3_grad[i]) +
                 (p_einstein_diffusion_coefficient * Wp[j]) * z3_grad[i] -
                 (dr_p_n * n[j] + dr_p_p * p[j]) * z3[i] / Constants::Q) *
                JxW;
            }
      }
  }



  template <int dim>
  void
  Solver<dim>::add_cell_products_to_l_rhs(
    Ddhdg::Solver<dim>::ScratchData &scratch)
  {
    const unsigned int n_q_points =
      scratch.fe_values_local.get_quadrature().size();
    const unsigned int loc_dofs_per_cell =
      scratch.fe_values_local.get_fe().dofs_per_cell;

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
    const auto &E0  = scratch.previous_f_cell.at(Component::V);
    const auto &n0  = scratch.previous_c_cell.at(Component::n);
    const auto &Wn0 = scratch.previous_f_cell.at(Component::n);
    const auto &p0  = scratch.previous_c_cell.at(Component::p);
    const auto &Wp0 = scratch.previous_f_cell.at(Component::p);

    const auto &c0 = scratch.doping_cell;

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double JxW = scratch.fe_values_local.JxW(q);

        const dealii::Tensor<2, dim> n_einstein_diffusion_coefficient =
          scratch.template compute_einstein_diffusion_coefficient<Component::n,
                                                                  false>(q);
        const dealii::Tensor<2, dim> p_einstein_diffusion_coefficient =
          scratch.template compute_einstein_diffusion_coefficient<Component::p,
                                                                  false>(q);

        const auto Jn = n0[q] * (scratch.mu_n_cell[q] * E0[q]) -
                        (n_einstein_diffusion_coefficient * Wn0[q]);
        const auto Jp = -p0[q] * (scratch.mu_p_cell[q] * E0[q]) -
                        (p_einstein_diffusion_coefficient * Wp0[q]);

        for (unsigned int i = 0; i < loc_dofs_per_cell; ++i)
          {
            const dealii::Tensor<1, dim> q1 =
              scratch.fe_values_local[electric_field].value(i, q);
            const double q1_div =
              scratch.fe_values_local[electric_field].divergence(i, q);
            const double z1 =
              scratch.fe_values_local[electric_potential].value(i, q);
            const dealii::Tensor<1, dim> z1_grad =
              scratch.fe_values_local[electric_potential].gradient(i, q);
            const dealii::Tensor<1, dim> q2 =
              scratch.fe_values_local[electron_displacement].value(i, q);
            const double q2_div =
              scratch.fe_values_local[electron_displacement].divergence(i, q);
            const double z2 =
              scratch.fe_values_local[electron_density].value(i, q);
            const dealii::Tensor<1, dim> z2_grad =
              scratch.fe_values_local[electron_density].gradient(i, q);
            const dealii::Tensor<1, dim> q3 =
              scratch.fe_values_local[hole_displacement].value(i, q);
            const double q3_div =
              scratch.fe_values_local[hole_displacement].divergence(i, q);
            const double z3 = scratch.fe_values_local[hole_density].value(i, q);
            const dealii::Tensor<1, dim> z3_grad =
              scratch.fe_values_local[hole_density].gradient(i, q);

            scratch.l_rhs[i] +=
              (V0[q] * q1_div - E0[q] * q1 +
               (scratch.epsilon_cell[q] * E0[q]) * z1_grad +
               Constants::Q * (c0[q] - n0[q] + p0[q]) * z1 + n0[q] * q2_div -
               Wn0[q] * q2 + scratch.r_n_cell[q] / Constants::Q * z2 +
               Jn * z2_grad + p0[q] * q3_div - Wp0[q] * q3 +
               +scratch.r_p_cell[q] / Constants::Q * z3 + Jp * z3_grad) *
              JxW;
          }
      }
  }



  template <int dim>
  void
  Solver<dim>::prepare_data_on_face_quadrature_points(
    Ddhdg::Solver<dim>::ScratchData &scratch)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      scratch.face_quadrature_points[q] =
        scratch.fe_face_values.quadrature_point(q);

    for (const auto c : Ddhdg::AllComponents)
      {
        const Displacement d = component2displacement(c);

        const FEValuesExtractors::Scalar c_extractor =
          this->get_component_extractor(c);
        const FEValuesExtractors::Vector d_extractor =
          this->get_displacement_extractor(d);
        const FEValuesExtractors::Scalar tr_c_extractor =
          this->get_trace_component_extractor(c);

        scratch.fe_face_values_local[c_extractor].get_function_values(
          current_solution_local, scratch.previous_c_face[c]);
        scratch.fe_face_values_local[d_extractor].get_function_values(
          current_solution_local, scratch.previous_f_face[c]);
        scratch.fe_face_values[tr_c_extractor].get_function_values(
          current_solution, scratch.previous_tr_c_face[c]);
      }

    this->permittivity->compute_absolute_permittivity(
      scratch.face_quadrature_points, scratch.epsilon_face);

    this->n_electron_mobility->compute_electron_mobility(
      scratch.face_quadrature_points, scratch.mu_n_face);
    this->p_electron_mobility->compute_electron_mobility(
      scratch.face_quadrature_points, scratch.mu_p_face);

    this->temperature->value_list(scratch.face_quadrature_points,
                                  scratch.T_face);
  }



  template <int dim>
  inline void
  Solver<dim>::copy_fe_values_on_scratch(
    Ddhdg::Solver<dim>::ScratchData &scratch,
    const unsigned int               face,
    const unsigned int               q)
  {
    for (const auto c : Ddhdg::AllComponents)
      {
        const Displacement               d = component2displacement(c);
        const FEValuesExtractors::Scalar c_extractor =
          this->get_component_extractor(c);
        const FEValuesExtractors::Vector d_extractor =
          this->get_displacement_extractor(d);

        auto &f  = scratch.f[c];
        auto &c_ = scratch.c[c];

        for (unsigned int k = 0;
             k < scratch.fe_local_support_on_face[face].size();
             ++k)
          {
            const unsigned int kk = scratch.fe_local_support_on_face[face][k];
            f[k]  = scratch.fe_face_values_local[d_extractor].value(kk, q);
            c_[k] = scratch.fe_face_values_local[c_extractor].value(kk, q);
          }
      }
  }



  template <int dim>
  inline void
  Solver<dim>::copy_fe_values_for_trace(
    Ddhdg::Solver<dim>::ScratchData &scratch,
    const unsigned int               face,
    const unsigned int               q)
  {
    for (const auto c : Ddhdg::AllComponents)
      {
        const FEValuesExtractors::Scalar extractor =
          this->get_trace_component_extractor(c);
        auto &tr_c = scratch.tr_c[c];
        for (unsigned int k = 0; k < scratch.fe_support_on_face[face].size();
             ++k)
          {
            tr_c[k] = scratch.fe_face_values[extractor].value(
              scratch.fe_support_on_face[face][k], q);
          }
      }
  }



  template <int dim>
  inline void
  Solver<dim>::assemble_lf_matrix(Ddhdg::Solver<dim>::ScratchData &scratch,
                                  const unsigned int               face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    auto &q1   = scratch.f.at(Component::V);
    auto &z1   = scratch.c.at(Component::V);
    auto &tr_V = scratch.tr_c.at(Component::V);

    auto &q2   = scratch.f.at(Component::n);
    auto &z2   = scratch.c.at(Component::n);
    auto &tr_n = scratch.tr_c.at(Component::n);

    auto &q3   = scratch.f.at(Component::p);
    auto &z3   = scratch.c.at(Component::p);
    auto &tr_p = scratch.tr_c.at(Component::p);

    const double V_tau = this->parameters->tau.at(Component::V);
    const double n_tau = this->parameters->tau.at(Component::n);
    const double p_tau = this->parameters->tau.at(Component::p);

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);
        copy_fe_values_for_trace(scratch, face, q);

        const double         JxW    = scratch.fe_face_values.JxW(q);
        const Tensor<1, dim> normal = scratch.fe_face_values.normal_vector(q);

        const dealii::Tensor<2, dim> n_einstein_diffusion_coefficient =
          scratch.template compute_einstein_diffusion_coefficient<Component::n>(
            q);
        const dealii::Tensor<2, dim> p_einstein_diffusion_coefficient =
          scratch.template compute_einstein_diffusion_coefficient<Component::p>(
            q);

        const double V_tau_stabilized =
          scratch.epsilon_face[q] * normal * normal * V_tau;
        const double n_tau_stabilized =
          n_einstein_diffusion_coefficient * normal * normal * n_tau;
        const double p_tau_stabilized =
          p_einstein_diffusion_coefficient * normal * normal * p_tau;

        for (unsigned int i = 0;
             i < scratch.fe_local_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii = scratch.fe_local_support_on_face[face][i];
            for (unsigned int j = 0;
                 j < scratch.fe_support_on_face[face].size();
                 ++j)
              {
                const unsigned int jj = scratch.fe_support_on_face[face][j];

                // Integrals of trace functions using as test function
                // the restriction of local test function on the border
                // i is the index of the test function
                scratch.lf_matrix(ii, jj) +=
                  (tr_V[j] * (q1[i] * normal) -
                   V_tau_stabilized * tr_V[j] * z1[i] +
                   tr_n[j] * (q2[i] * normal) +
                   n_tau_stabilized * (tr_n[j] * z2[i]) +
                   tr_p[j] * (q3[i] * normal) +
                   p_tau_stabilized * (tr_p[j] * z3[i])) *
                  JxW;
              }
          }
      }
  }



  template <int dim>
  inline void
  Solver<dim>::add_lf_matrix_terms_to_l_rhs(
    Ddhdg::Solver<dim>::ScratchData &scratch,
    const unsigned int               face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

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

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        const double         JxW    = scratch.fe_face_values.JxW(q);
        const Tensor<1, dim> normal = scratch.fe_face_values.normal_vector(q);

        const dealii::Tensor<2, dim> n_einstein_diffusion_coefficient =
          scratch.template compute_einstein_diffusion_coefficient<Component::n>(
            q);
        const dealii::Tensor<2, dim> p_einstein_diffusion_coefficient =
          scratch.template compute_einstein_diffusion_coefficient<Component::p>(
            q);

        const double V_tau_stabilized =
          scratch.epsilon_face[q] * normal * normal * V_tau;
        const double n_tau_stabilized =
          n_einstein_diffusion_coefficient * normal * normal * n_tau;
        const double p_tau_stabilized =
          p_einstein_diffusion_coefficient * normal * normal * p_tau;

        for (unsigned int i = 0;
             i < scratch.fe_local_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii = scratch.fe_local_support_on_face[face][i];

            const dealii::Tensor<1, dim> q1 =
              scratch.fe_face_values_local[electric_field].value(ii, q);
            const double z1 =
              scratch.fe_face_values_local[electric_potential].value(ii, q);
            const dealii::Tensor<1, dim> q2 =
              scratch.fe_face_values_local[electron_displacement].value(ii, q);
            const double z2 =
              scratch.fe_face_values_local[electron_density].value(ii, q);
            const dealii::Tensor<1, dim> q3 =
              scratch.fe_face_values_local[hole_displacement].value(ii, q);
            const double z3 =
              scratch.fe_face_values_local[hole_density].value(ii, q);

            scratch.l_rhs(ii) +=
              (-tr_V0[q] * (q1 * normal) + V_tau_stabilized * tr_V0[q] * z1 -
               tr_n0[q] * (q2 * normal) - n_tau_stabilized * tr_n0[q] * z2 -
               tr_p0[q] * (q3 * normal) - p_tau_stabilized * tr_p0[q] * z3) *
              JxW;
          }
      }
  }



  template <int dim>
  template <Component c>
  inline void
  Solver<dim>::assemble_fl_matrix(Ddhdg::Solver<dim>::ScratchData &scratch,
                                  const unsigned int               face)
  {
    if (c != V and c != n and c != p)
      AssertThrow(false, UnknownComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    auto &c_ = scratch.c.at(c);
    auto &f  = scratch.f.at(c);
    auto &xi = scratch.tr_c.at(c);

    const auto &E = scratch.previous_f_face.at(Component::V);

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

        const double         JxW    = scratch.fe_face_values.JxW(q);
        const Tensor<1, dim> normal = scratch.fe_face_values.normal_vector(q);

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

        const dealii::Tensor<2, dim> stabilizing_tensor =
          (c == Component::V) ?
            scratch.epsilon_face[q] :
            scratch.template compute_einstein_diffusion_coefficient<c>(q);

        const double tau_stabilized =
          stabilizing_tensor * normal * normal * tau;

        for (unsigned int i = 0; i < scratch.fe_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii = scratch.fe_support_on_face[face][i];
            for (unsigned int j = 0;
                 j < scratch.fe_local_support_on_face[face].size();
                 ++j)
              {
                const unsigned int jj =
                  scratch.fe_local_support_on_face[face][j];

                // Integrals of the local functions restricted on the
                // border. The sign is reversed to be used in the
                // Schur complement (and therefore this is the
                // opposite of the right matrix that describes the
                // problem on this cell)
                if (c == V)
                  {
                    scratch.fl_matrix(ii, jj) -=
                      ((scratch.epsilon_face[q] * f[j]) * normal +
                       tau_stabilized * c_[j]) *
                      xi[i] * JxW;
                  }
                if (c == n)
                  {
                    const dealii::Tensor<1, dim> mu_n_times_E =
                      scratch.mu_n_face[q] * scratch.f[Component::V][j];

                    scratch.fl_matrix(ii, jj) -=
                      ((c_[j] * mu_n_times_previous_E + n0[q] * mu_n_times_E) *
                         normal -
                       (n_einstein_diffusion_coefficient * f[j]) * normal -
                       tau_stabilized * c_[j]) *
                      xi[i] * JxW;
                  }
                if (c == p)
                  {
                    const dealii::Tensor<1, dim> mu_p_times_E =
                      scratch.mu_p_face[q] * scratch.f[Component::V][j];

                    scratch.fl_matrix(ii, jj) -=
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
  template <Component c>
  inline void
  Solver<dim>::add_fl_matrix_terms_to_f_rhs(
    Ddhdg::Solver<dim>::ScratchData &scratch,
    Ddhdg::Solver<dim>::PerTaskData &task_data,
    const unsigned int               face)
  {
    if (c != V and c != n and c != p)
      AssertThrow(false, UnknownComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    const FEValuesExtractors::Scalar tr_c_extractor =
      this->get_trace_component_extractor(c);

    auto &c0 = scratch.previous_c_face.at(c);
    auto &f0 = scratch.previous_f_face.at(c);

    const auto &E0 = scratch.previous_f_face[Component::V];

    double epsilon_times_previous_E_times_normal = 0.;
    double mu_n_times_previous_E_times_normal    = 0.;
    double mu_p_times_previous_E_times_normal    = 0.;

    dealii::Tensor<2, dim> n_einstein_diffusion_coefficient;
    dealii::Tensor<2, dim> p_einstein_diffusion_coefficient;

    const double tau = this->parameters->tau.at(c);

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        const double         JxW    = scratch.fe_face_values.JxW(q);
        const Tensor<1, dim> normal = scratch.fe_face_values.normal_vector(q);

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

        const dealii::Tensor<2, dim> stabilizing_tensor =
          (c == Component::V) ?
            scratch.epsilon_face[q] :
            scratch.template compute_einstein_diffusion_coefficient<c>(q);
        const double tau_stabilized =
          stabilizing_tensor * normal * normal * tau;

        for (unsigned int i = 0; i < scratch.fe_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii = scratch.fe_support_on_face[face][i];
            const double       xi =
              scratch.fe_face_values[tr_c_extractor].value(ii, q);
            if (c == V)
              {
                task_data.cell_vector[ii] +=
                  (-epsilon_times_previous_E_times_normal -
                   tau_stabilized * c0[q]) *
                  xi * JxW;
              }
            if (c == n)
              {
                task_data.cell_vector[ii] +=
                  (-c0[q] * mu_n_times_previous_E_times_normal +
                   n_einstein_diffusion_coefficient * f0[q] * normal +
                   tau_stabilized * c0[q]) *
                  xi * JxW;
              }
            if (c == p)
              {
                task_data.cell_vector[ii] +=
                  (c0[q] * mu_p_times_previous_E_times_normal +
                   p_einstein_diffusion_coefficient * f0[q] * normal +
                   tau_stabilized * c0[q]) *
                  xi * JxW;
              }
          }
      }
  }



  template <int dim>
  template <Component c>
  inline void
  Solver<dim>::assemble_cell_matrix(Ddhdg::Solver<dim>::ScratchData &scratch,
                                    Ddhdg::Solver<dim>::PerTaskData &task_data,
                                    const unsigned int               face)
  {
    if (c != V and c != n and c != p)
      AssertThrow(false, UnknownComponent());

    auto &tr_c = scratch.tr_c.at(c);
    auto &xi   = scratch.tr_c.at(c);

    const double tau  = this->parameters->tau.at(c);
    const double sign = (c == V) ? -1 : 1;

    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);
        copy_fe_values_for_trace(scratch, face, q);

        const double         JxW    = scratch.fe_face_values.JxW(q);
        const Tensor<1, dim> normal = scratch.fe_face_values.normal_vector(q);

        const dealii::Tensor<2, dim> stabilizing_tensor =
          (c == Component::V) ?
            scratch.epsilon_face[q] :
            scratch.template compute_einstein_diffusion_coefficient<c>(q);
        const double tau_stabilized =
          stabilizing_tensor * normal * normal * tau;

        // Integrals of trace functions (both test and trial)
        for (unsigned int i = 0; i < scratch.fe_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii = scratch.fe_support_on_face[face][i];
            for (unsigned int j = 0;
                 j < scratch.fe_support_on_face[face].size();
                 ++j)
              {
                const unsigned int jj = scratch.fe_support_on_face[face][j];
                task_data.cell_matrix(ii, jj) +=
                  sign * tau_stabilized * tr_c[j] * xi[i] * JxW;
              }
          }
      }
  }



  template <int dim>
  template <Component c>
  inline void
  Solver<dim>::add_cell_matrix_terms_to_f_rhs(
    Ddhdg::Solver<dim>::ScratchData &scratch,
    Ddhdg::Solver<dim>::PerTaskData &task_data,
    const unsigned int               face)
  {
    if (c != V and c != n and c != p)
      AssertThrow(false, UnknownComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    const double tau  = this->parameters->tau.at(c);
    const double sign = (c == Component::V) ? -1 : 1;

    const FEValuesExtractors::Scalar tr_c_extractor =
      this->get_trace_component_extractor(c);

    const auto &tr_c0 = scratch.previous_tr_c_face.at(c);

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        const double         JxW    = scratch.fe_face_values.JxW(q);
        const Tensor<1, dim> normal = scratch.fe_face_values.normal_vector(q);

        const dealii::Tensor<2, dim> stabilizing_tensor =
          (c == Component::V) ?
            scratch.epsilon_face[q] :
            scratch.template compute_einstein_diffusion_coefficient<c>(q);
        const double tau_stabilized =
          stabilizing_tensor * normal * normal * tau;

        for (unsigned int i = 0; i < scratch.fe_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii = scratch.fe_support_on_face[face][i];
            const double       xi =
              scratch.fe_face_values[tr_c_extractor].value(ii, q);
            task_data.cell_vector[ii] +=
              -sign * tau_stabilized * tr_c0[q] * xi * JxW;
          }
      }
  }



  template <int dim>
  template <Ddhdg::Component c>
  inline void
  Solver<dim>::apply_dbc_on_face(
    Ddhdg::Solver<dim>::ScratchData &             scratch,
    Ddhdg::Solver<dim>::PerTaskData &             task_data,
    const Ddhdg::DirichletBoundaryCondition<dim> &dbc,
    unsigned int                                  face)
  {
    if (c != V and c != n and c != p)
      Assert(false, UnknownComponent());

    auto &tr_c  = scratch.tr_c.at(c);
    auto &tr_c0 = scratch.previous_tr_c_face.at(c);

    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_for_trace(scratch, face, q);

        const double JxW = scratch.fe_face_values.JxW(q);

        const Point<dim> quadrature_point =
          scratch.fe_face_values.quadrature_point(q);
        const double dbc_value = dbc.evaluate(quadrature_point) - tr_c0[q];

        for (unsigned int i = 0; i < scratch.fe_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii = scratch.fe_support_on_face[face][i];
            for (unsigned int j = 0;
                 j < scratch.fe_support_on_face[face].size();
                 ++j)
              {
                const unsigned int jj = scratch.fe_support_on_face[face][j];
                task_data.cell_matrix(ii, jj) += tr_c[i] * tr_c[j] * JxW;
              }

            task_data.cell_vector[ii] += tr_c[i] * dbc_value * JxW;
          }
      }
  }



  template <int dim>
  template <Ddhdg::Component c>
  void
  Solver<dim>::apply_nbc_on_face(
    Ddhdg::Solver<dim>::ScratchData &           scratch,
    Ddhdg::Solver<dim>::PerTaskData &           task_data,
    const Ddhdg::NeumannBoundaryCondition<dim> &nbc,
    unsigned int                                face)
  {
    if (c != V and c != n and c != p)
      Assert(false, UnknownComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    auto &tr_c = scratch.tr_c.at(c);

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_for_trace(scratch, face, q);

        const double     JxW = scratch.fe_face_values.JxW(q);
        const Point<dim> quadrature_point =
          scratch.fe_face_values.quadrature_point(q);
        const double nbc_value =
          (c == V ? nbc.evaluate(quadrature_point) :
                    nbc.evaluate(quadrature_point) / Constants::Q);

        for (unsigned int i = 0; i < scratch.fe_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii = scratch.fe_support_on_face[face][i];
            task_data.cell_vector(ii) += tr_c[i] * nbc_value * JxW;
          }
      }
  }


  template <int dim>
  inline void
  Solver<dim>::add_border_products_to_ll_matrix(
    Ddhdg::Solver<dim>::ScratchData &scratch,
    const unsigned int               face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    auto &E  = scratch.f.at(Component::V);
    auto &V  = scratch.c.at(Component::V);
    auto &Wn = scratch.f.at(Component::n);
    auto &n  = scratch.c.at(Component::n);
    auto &Wp = scratch.f.at(Component::p);
    auto &p  = scratch.c.at(Component::p);

    auto &z1 = scratch.c.at(Component::V);
    auto &z2 = scratch.c.at(Component::n);
    auto &z3 = scratch.c.at(Component::p);

    const auto  n0 = scratch.previous_c_face.at(Component::n);
    const auto  p0 = scratch.previous_c_face.at(Component::p);
    const auto &E0 = scratch.previous_f_face.at(Component::V);

    const double V_tau = this->parameters->tau.at(Component::V);
    const double n_tau = this->parameters->tau.at(Component::n);
    const double p_tau = this->parameters->tau.at(Component::p);

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);

        // Integrals on the border of the cell of the local functions
        const double         JxW    = scratch.fe_face_values.JxW(q);
        const Tensor<1, dim> normal = scratch.fe_face_values.normal_vector(q);

        const dealii::Tensor<1, dim> mu_n_times_previous_E =
          scratch.mu_n_face[q] * E0[q];
        const dealii::Tensor<1, dim> mu_p_times_previous_E =
          scratch.mu_p_face[q] * E0[q];
        const dealii::Tensor<2, dim> n_einstein_diffusion_coefficient =
          scratch.template compute_einstein_diffusion_coefficient<Component::n>(
            q);
        const dealii::Tensor<2, dim> p_einstein_diffusion_coefficient =
          scratch.template compute_einstein_diffusion_coefficient<Component::p>(
            q);

        const double V_tau_stabilized =
          scratch.epsilon_face[q] * normal * normal * V_tau;
        const double n_tau_stabilized =
          n_einstein_diffusion_coefficient * normal * normal * n_tau;
        const double p_tau_stabilized =
          p_einstein_diffusion_coefficient * normal * normal * p_tau;

        for (unsigned int i = 0;
             i < scratch.fe_local_support_on_face[face].size();
             ++i)
          {
            for (unsigned int j = 0;
                 j < scratch.fe_local_support_on_face[face].size();
                 ++j)
              {
                const unsigned int ii =
                  scratch.fe_local_support_on_face[face][i];
                const unsigned int jj =
                  scratch.fe_local_support_on_face[face][j];
                scratch.ll_matrix(ii, jj) +=
                  (scratch.epsilon_face[q] * E[j] * normal * z1[i] +
                   V_tau_stabilized * V[j] * z1[i] +
                   (n[j] * mu_n_times_previous_E +
                    scratch.mu_n_face[q] * E[j] * n0[q]) *
                     normal * z2[i] -
                   n_einstein_diffusion_coefficient * Wn[j] * normal * z2[i] -
                   n_tau_stabilized * n[j] * z2[i] -
                   (p[j] * mu_p_times_previous_E +
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
  inline void
  Solver<dim>::add_border_products_to_l_rhs(
    Ddhdg::Solver<dim>::ScratchData &scratch,
    const unsigned int               face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    const FEValuesExtractors::Scalar electric_potential =
      this->get_component_extractor(Component::V);
    const FEValuesExtractors::Scalar electron_density =
      this->get_component_extractor(Component::n);
    const FEValuesExtractors::Scalar hole_density =
      this->get_component_extractor(Component::p);

    auto &V0  = scratch.previous_c_face.at(Component::V);
    auto &E0  = scratch.previous_f_face.at(Component::V);
    auto &n0  = scratch.previous_c_face.at(Component::n);
    auto &Wn0 = scratch.previous_f_face.at(Component::n);
    auto &p0  = scratch.previous_c_face.at(Component::p);
    auto &Wp0 = scratch.previous_f_face.at(Component::p);

    const double V_tau = this->parameters->tau.at(Component::V);
    const double n_tau = this->parameters->tau.at(Component::n);
    const double p_tau = this->parameters->tau.at(Component::p);

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        const double         JxW    = scratch.fe_face_values.JxW(q);
        const Tensor<1, dim> normal = scratch.fe_face_values.normal_vector(q);

        const double epsilon_times_E0_times_normal =
          (scratch.epsilon_face[q] * E0[q]) * normal;

        const dealii::Tensor<2, dim> n_einstein_diffusion_coefficient =
          scratch.template compute_einstein_diffusion_coefficient<Component::n>(
            q);
        const dealii::Tensor<2, dim> p_einstein_diffusion_coefficient =
          scratch.template compute_einstein_diffusion_coefficient<Component::p>(
            q);

        const double Jn_flux = (n0[q] * (scratch.mu_n_face[q] * E0[q]) -
                                (n_einstein_diffusion_coefficient * Wn0[q])) *
                               normal;
        const double Jp_flux = (-p0[q] * (scratch.mu_p_face[q] * E0[q]) -
                                (p_einstein_diffusion_coefficient * Wp0[q])) *
                               normal;

        const double V_tau_stabilized =
          scratch.epsilon_face[q] * normal * normal * V_tau;
        const double n_tau_stabilized =
          n_einstein_diffusion_coefficient * normal * normal * n_tau;
        const double p_tau_stabilized =
          p_einstein_diffusion_coefficient * normal * normal * p_tau;

        for (unsigned int i = 0;
             i < scratch.fe_local_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii = scratch.fe_local_support_on_face[face][i];

            const double z1 =
              scratch.fe_face_values_local[electric_potential].value(ii, q);
            const double z2 =
              scratch.fe_face_values_local[electron_density].value(ii, q);
            const double z3 =
              scratch.fe_face_values_local[hole_density].value(ii, q);

            scratch.l_rhs[ii] += (-epsilon_times_E0_times_normal * z1 -
                                  V_tau_stabilized * V0[q] * z1 - Jn_flux * z2 +
                                  n_tau_stabilized * n0[q] * z2 - Jp_flux * z3 +
                                  p_tau_stabilized * p0[q] * z3) *
                                 JxW;
          }
      }
  }



  template <int dim>
  inline void
  Solver<dim>::add_trace_terms_to_l_rhs(
    Ddhdg::Solver<dim>::ScratchData &scratch,
    const unsigned int               face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);

        const double         JxW    = scratch.fe_face_values.JxW(q);
        const Tensor<1, dim> normal = scratch.fe_face_values.normal_vector(q);

        for (const auto c : Ddhdg::AllComponents)
          {
            auto &f                    = scratch.f.at(c);
            auto &c_                   = scratch.c.at(c);
            auto &tr_c_solution_values = scratch.tr_c_solution_values.at(c);

            const double tau  = this->parameters->tau.at(c);
            const double sign = (c == Component::V) ? 1 : -1;

            const dealii::Tensor<2, dim> stabilizing_tensor =
              (c == Component::V) ?
                scratch.epsilon_face[q] :
                (c == Component::n) ?
                scratch.template compute_einstein_diffusion_coefficient<
                  Component::n>(q) :
                scratch.template compute_einstein_diffusion_coefficient<
                  Component::p>(q);
            const double tau_stabilized =
              stabilizing_tensor * normal * normal * tau;

            for (unsigned int i = 0;
                 i < scratch.fe_local_support_on_face[face].size();
                 ++i)
              {
                const unsigned int ii =
                  scratch.fe_local_support_on_face[face][i];
                scratch.l_rhs(ii) +=
                  (-f[i] * normal + sign * tau_stabilized * c_[i]) *
                  tr_c_solution_values[q] * JxW;
              }
          }
      }
  }



  template <int dim>
  void
  Solver<dim>::assemble_system_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData &                                         scratch,
    PerTaskData &                                         task_data)
  {
    typename DoFHandler<dim>::active_cell_iterator loc_cell(&(*triangulation),
                                                            cell->level(),
                                                            cell->index(),
                                                            &dof_handler_local);

    // Reset every value that could be related to the previous cell
    scratch.ll_matrix = 0;
    scratch.l_rhs     = 0;
    if (!task_data.trace_reconstruct)
      {
        scratch.lf_matrix     = 0;
        scratch.fl_matrix     = 0;
        task_data.cell_matrix = 0;
        task_data.cell_vector = 0;
      }
    scratch.fe_values_local.reinit(loc_cell);

    // We use the following function to copy every value that we need from the
    // fe_values objects into the scratch. This also compute the physical
    // parameters (like permittivity or the recombination term) on the
    // quadrature points of the cell
    this->prepare_data_on_cell_quadrature_points(scratch);

    // Integrals on the overall cell
    // This function computes every L2 product that we need to compute in order
    // to assemble the matrix for the current cell.
    this->add_cell_products_to_ll_matrix(scratch);

    // This function, instead, computes the l2 products that are needed for
    // the right hand term
    this->add_cell_products_to_l_rhs(scratch);

    // Now we must perform the L2 product on the boundary, i.e. for each face of
    // the cell
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
         ++face)
      {
        scratch.fe_face_values_local.reinit(loc_cell, face);
        scratch.fe_face_values.reinit(cell, face);

        // If we have already solved the system for the trace, copy the values
        // of the global solution in the scratch that stores the values for this
        // cell
        if (task_data.trace_reconstruct)
          {
            for (const auto c : Ddhdg::AllComponents)
              {
                const FEValuesExtractors::Scalar extractor =
                  this->get_trace_component_extractor(c);
                scratch.fe_face_values[extractor].get_function_values(
                  update, scratch.tr_c_solution_values[c]);
              }
          }

        // Now I create some maps to store, for each component, if there is a
        // boundary condition for it
        const auto face_boundary_id = cell->face(face)->boundary_id();
        std::map<Ddhdg::Component, bool> has_dirichlet_conditions;
        std::map<Ddhdg::Component, bool> has_neumann_conditions;

        // Now we populate the previous maps
        for (const auto c : Ddhdg::AllComponents)
          {
            has_dirichlet_conditions.insert(
              {c,
               boundary_handler->has_dirichlet_boundary_conditions(
                 face_boundary_id, c)});
            has_neumann_conditions.insert(
              {c,
               boundary_handler->has_neumann_boundary_conditions(
                 face_boundary_id, c)});
          }

        // Before assembling the other parts of the matrix, we need the values
        // of epsilon, mu and D_n on the quadrature points of the current face.
        // Moreover, we want to populate the scratch object with the values of
        // the solution at the previous step. All this jobs are accomplished by
        // the next function
        prepare_data_on_face_quadrature_points(scratch);

        // The following function adds some terms to l_rhs. Usually, all the
        // functions are coupled (like assemble_matrix_XXX and
        // add_matrix_XXX_terms_to_l_rhs) because the first function is the
        // function that assembles the matrix and the second one is the function
        // that adds the corresponding terms to l_rhs (i.e. the product of the
        // matrix with the previous solution, so that the solution of the system
        // is the update from the previous solution to the new one).
        // The following function is the only exception: indeed, the terms that
        // it generates are always needed while the terms that its
        // corresponding function generates are useful only if we are not
        // reconstructing the solution from the trace
        add_lf_matrix_terms_to_l_rhs(scratch, face);

        // Assembly the other matrices (the ll_matrix has been assembled
        // calling the add_cell_products_to_ll_matrix method)
        if (!task_data.trace_reconstruct)
          {
            assemble_lf_matrix(scratch, face);

            if (has_dirichlet_conditions[V])
              {
                const auto dbc =
                  boundary_handler->get_dirichlet_conditions_for_id(
                    face_boundary_id, V);
                apply_dbc_on_face<V>(scratch, task_data, dbc, face);
              }
            else
              {
                assemble_fl_matrix<V>(scratch, face);
                add_fl_matrix_terms_to_f_rhs<V>(scratch, task_data, face);
                assemble_cell_matrix<V>(scratch, task_data, face);
                add_cell_matrix_terms_to_f_rhs<V>(scratch, task_data, face);
                if (has_neumann_conditions[V])
                  {
                    const auto nbc =
                      boundary_handler->get_neumann_conditions_for_id(
                        face_boundary_id, V);
                    this->apply_nbc_on_face<V>(scratch, task_data, nbc, face);
                  }
              }
            if (has_dirichlet_conditions[n])
              {
                const auto dbc =
                  boundary_handler->get_dirichlet_conditions_for_id(
                    face_boundary_id, n);
                apply_dbc_on_face<n>(scratch, task_data, dbc, face);
              }
            else
              {
                assemble_fl_matrix<n>(scratch, face);
                add_fl_matrix_terms_to_f_rhs<n>(scratch, task_data, face);
                assemble_cell_matrix<n>(scratch, task_data, face);
                add_cell_matrix_terms_to_f_rhs<n>(scratch, task_data, face);
                if (has_neumann_conditions[n])
                  {
                    const auto nbc =
                      boundary_handler->get_neumann_conditions_for_id(
                        face_boundary_id, n);
                    this->apply_nbc_on_face<n>(scratch, task_data, nbc, face);
                  }
              }
            if (has_dirichlet_conditions[p])
              {
                const auto dbc =
                  boundary_handler->get_dirichlet_conditions_for_id(
                    face_boundary_id, p);
                apply_dbc_on_face<p>(scratch, task_data, dbc, face);
              }
            else
              {
                assemble_fl_matrix<p>(scratch, face);
                add_fl_matrix_terms_to_f_rhs<p>(scratch, task_data, face);
                assemble_cell_matrix<p>(scratch, task_data, face);
                add_cell_matrix_terms_to_f_rhs<p>(scratch, task_data, face);
                if (has_neumann_conditions[p])
                  {
                    const auto nbc =
                      boundary_handler->get_neumann_conditions_for_id(
                        face_boundary_id, p);
                    this->apply_nbc_on_face<p>(scratch, task_data, nbc, face);
                  }
              }
          }

        // These are the last terms of the ll matrix, the ones that are
        // generated by L2 products only on the boundary of the cell
        add_border_products_to_ll_matrix(scratch, face);
        add_border_products_to_l_rhs(scratch, face);

        if (task_data.trace_reconstruct)
          add_trace_terms_to_l_rhs(scratch, face);
      }

    inversion_mutex.lock();
    scratch.ll_matrix.gauss_jordan();
    inversion_mutex.unlock();

    if (!task_data.trace_reconstruct)
      {
        scratch.fl_matrix.mmult(scratch.tmp_matrix, scratch.ll_matrix);
        scratch.tmp_matrix.vmult_add(task_data.cell_vector, scratch.l_rhs);
        scratch.tmp_matrix.mmult(task_data.cell_matrix,
                                 scratch.lf_matrix,
                                 true);
        cell->get_dof_indices(task_data.dof_indices);
      }
    else
      {
        scratch.ll_matrix.vmult(scratch.tmp_rhs, scratch.l_rhs);
        loc_cell->set_dof_values(scratch.tmp_rhs, update_local);
      }
  }



  template <int dim>
  void
  Solver<dim>::copy_local_to_global(const PerTaskData &data)
  {
    if (!data.trace_reconstruct)
      constraints.distribute_local_to_global(data.cell_matrix,
                                             data.cell_vector,
                                             data.dof_indices,
                                             system_matrix,
                                             system_rhs);
  }



  template <int dim>
  void
  Solver<dim>::solve_linear_problem()
  {
    std::cout << "    RHS norm   : " << system_rhs.l2_norm() << std::endl
              << "    Matrix norm: " << system_matrix.linfty_norm()
              << std::endl;

    SolverControl solver_control(system_matrix.m() * 10,
                                 1e-10 * system_rhs.l2_norm());

    if (parameters->iterative_linear_solver)
      {
        SolverGMRES<> linear_solver(solver_control);
        linear_solver.solve(system_matrix,
                            update,
                            system_rhs,
                            PreconditionIdentity());
        std::cout << "    Number of GMRES iterations: "
                  << solver_control.last_step() << std::endl;
      }
    else
      {
        SparseDirectUMFPACK Ainv;
        Ainv.initialize(system_matrix);
        Ainv.vmult(update, system_rhs);
      }
    constraints.distribute(update);
  }



  template <int dim>
  NonlinearIteratorStatus
  Solver<dim>::run(const double absolute_tol,
                   const double relative_tol,
                   const int    max_number_of_iterations)
  {
    if (!this->initialized)
      setup_system();

    bool   convergence_reached = false;
    int    step                = 0;
    double local_update_norm   = 0.;
    double current_solution_local_norm;

    for (step = 1;
         step <= max_number_of_iterations || max_number_of_iterations < 0;
         step++)
      {
        std::cout << "Computing step number " << step << std::endl;

        this->update        = 0;
        this->update_local  = 0;
        this->system_matrix = 0;
        this->system_rhs    = 0;

        if (parameters->multithreading)
          assemble_system_multithreaded(false);
        else
          assemble_system(false);

        solve_linear_problem();

        if (parameters->multithreading)
          assemble_system_multithreaded(true);
        else
          assemble_system(true);

        local_update_norm = this->update_local.linfty_norm();
        current_solution_local_norm =
          this->current_solution_local.linfty_norm();

        std::cout << "Difference in norm compared to the previous step: "
                  << local_update_norm << std::endl;

        this->current_solution += this->update;
        this->current_solution_local += this->update_local;

        if (local_update_norm < absolute_tol)
          {
            std::cout << "Update is smaller than absolute tolerance. "
                      << "CONVERGENCE REACHED" << std::endl;
            convergence_reached = true;
            break;
          }
        if (local_update_norm < relative_tol * current_solution_local_norm)
          {
            std::cout << "Update is smaller than relative tolerance. "
                      << "CONVERGENCE REACHED" << std::endl;
            convergence_reached = true;
            break;
          }
      }

    return NonlinearIteratorStatus(convergence_reached,
                                   step,
                                   local_update_norm);
  }



  template <int dim>
  NonlinearIteratorStatus
  Solver<dim>::run()
  {
    return this->run(parameters->nonlinear_solver_absolute_tolerance,
                     parameters->nonlinear_solver_relative_tolerance,
                     parameters->nonlinear_solver_max_number_of_iterations);
  }



  template <int dim>
  double
  Solver<dim>::estimate_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                                        expected_solution,
    const Ddhdg::Component              c,
    const dealii::VectorTools::NormType norm) const
  {
    Vector<double> difference_per_cell(triangulation->n_active_cells());

    unsigned int component_index = 0;
    switch (c)
      {
        case Component::V:
          component_index = dim;
          break;
        case Component::n:
          component_index = 2 * dim + 1;
          break;
        case Component::p:
          component_index = 3 * dim + 2;
          break;
        default:
          AssertThrow(false, UnknownComponent())
      }

    const unsigned int n_of_components = (dim + 1) * 3;
    const auto         component_selection =
      dealii::ComponentSelectFunction<dim>(component_index, n_of_components);

    std::map<unsigned int, const std::shared_ptr<const dealii::Function<dim>>>
      c_map;
    c_map.insert({component_index, expected_solution});
    FunctionByComponents<dim> expected_solution_multidim =
      FunctionByComponents<dim>(n_of_components, c_map);

    VectorTools::integrate_difference(this->dof_handler_local,
                                      this->current_solution_local,
                                      expected_solution_multidim,
                                      difference_per_cell,
                                      QGauss<dim>(fe.degree + 1),
                                      norm,
                                      &component_selection);

    const double global_error =
      VectorTools::compute_global_error(*triangulation,
                                        difference_per_cell,
                                        norm);
    return global_error;
  }



  template <int dim>
  double
  Solver<dim>::estimate_l2_error(
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
  Solver<dim>::estimate_h1_error(
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
  Solver<dim>::estimate_linfty_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                           expected_solution,
    const Ddhdg::Component c) const
  {
    return this->estimate_error(expected_solution,
                                c,
                                dealii::VectorTools::Linfty_norm);
  }



  template <int dim>
  void
  Solver<dim>::output_results(const std::string &solution_filename,
                              const bool         save_update) const
  {
    std::ofstream output(solution_filename);
    DataOut<dim>  data_out;
    // DataOutBase::VtkFlags flags;
    // flags.write_higher_order_cells = true;
    // data_out.set_flags(flags);

    std::vector<std::string> names(dim, "electric_field");
    names.emplace_back("electric_potential");
    for (int i = 0; i < dim; i++)
      names.emplace_back("electron_displacement");
    names.emplace_back("electron_density");
    for (int i = 0; i < dim; i++)
      names.emplace_back("hole_displacement");
    names.emplace_back("hole_density");

    std::vector<std::string> update_names;
    update_names.reserve(3 * (dim + 1));
    for (const auto &n : names)
      update_names.push_back(n + "_updates");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      component_interpretation(
        3 * (dim + 1),
        DataComponentInterpretation::component_is_part_of_vector);
    component_interpretation[dim] =
      DataComponentInterpretation::component_is_scalar;
    component_interpretation[2 * dim + 1] =
      DataComponentInterpretation::component_is_scalar;
    component_interpretation[3 * dim + 2] =
      DataComponentInterpretation::component_is_scalar;

    data_out.add_data_vector(this->dof_handler_local,
                             this->current_solution_local,
                             names,
                             component_interpretation);

    if (save_update)
      data_out.add_data_vector(this->dof_handler_local,
                               this->update_local,
                               update_names,
                               component_interpretation);

    data_out.build_patches(StaticMappingQ1<dim>::mapping,
                           fe.degree,
                           DataOut<dim>::curved_inner_cells);
    data_out.write_vtk(output);
  }



  template <>
  void
  Solver<1>::output_results(const std::string &solution_filename,
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
  Solver<dim>::output_results(const std::string &solution_filename,
                              const std::string &trace_filename,
                              const bool         save_update) const
  {
    output_results(solution_filename, save_update);

    std::ofstream            face_output(trace_filename);
    DataOutFaces<dim>        data_out_face(false);
    std::vector<std::string> face_names(3);
    face_names[0] = "electric_potential";
    face_names[1] = "electron_density";
    face_names[2] = "hole_density";

    std::vector<std::string> update_face_names;
    update_face_names.reserve(3);
    for (const auto &n : face_names)
      update_face_names.push_back(n + "_updates");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      face_component_type(3, DataComponentInterpretation::component_is_scalar);
    data_out_face.add_data_vector(this->dof_handler,
                                  this->current_solution,
                                  face_names,
                                  face_component_type);

    if (save_update)
      data_out_face.add_data_vector(this->dof_handler,
                                    this->update,
                                    update_face_names,
                                    face_component_type);

    data_out_face.build_patches(fe.degree);
    data_out_face.write_vtk(face_output);
  }



  template <int dim>
  void
  Solver<dim>::print_convergence_table(
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
  Solver<dim>::print_convergence_table(
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
        const NonlinearIteratorStatus iter_status = this->run();

        converged            = iter_status.converged;
        iterations           = iter_status.iterations;
        last_step_difference = iter_status.last_update_norm;

        error_table->error_from_exact(dof_handler_local,
                                      current_solution_local,
                                      expected_solution);

        // this->output_results("solution_" + std::to_string(cycle) + ".vtk",
        //                      "trace_" + std::to_string(cycle) + ".vtk");

        if (cycle != n_cycles - 1)
          this->refine_grid(1);
      }
    error_table->output_table(std::cout);
  }


  template class Solver<1>;

  template class Solver<2>;

  template class Solver<3>;

} // end of namespace Ddhdg
