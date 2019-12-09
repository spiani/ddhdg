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
  Solver<dim>::Solver(const std::shared_ptr<const Problem<dim>>     problem,
                      const std::shared_ptr<const SolverParameters> parameters)
    : triangulation(copy_triangulation(problem->triangulation))
    , permittivity(problem->permittivity)
    , electron_mobility(problem->electron_mobility)
    , recombination_term(problem->recombination_term)
    , temperature(problem->temperature)
    , boundary_handler(problem->boundary_handler)
    , parameters(parameters)
    , fe_local(FE_DGQ<dim>(parameters->V_degree),
               dim,
               FE_DGQ<dim>(parameters->V_degree),
               1,
               FE_DGQ<dim>(parameters->n_degree),
               dim,
               FE_DGQ<dim>(parameters->n_degree),
               1)
    , dof_handler_local(*triangulation)
    , fe(FE_FaceQ<dim>(parameters->V_degree),
         1,
         FE_FaceQ<dim>(parameters->n_degree),
         1)
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

    previous_solution.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    previous_solution_local.reinit(dof_handler_local.n_dofs());
    solution_local.reinit(dof_handler_local.n_dofs());

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
    dealii::ComponentMask mask(2 * dim + 2, false);
    switch (component)
      {
        case Component::V:
          mask.set(dim, true);
          break;
        case Component::n:
          mask.set(2 * dim + 1, true);
          break;
        default:
          AssertThrow(false, UnknownComponent());
      }
    return mask;
  }



  template <int dim>
  dealii::ComponentMask
  Solver<dim>::get_component_mask(const Displacement displacement)
  {
    dealii::ComponentMask mask(2 * dim + 2, false);
    switch (displacement)
      {
        case Displacement::E:
          for (unsigned int i = 0; i < dim; i++)
            mask.set(i, true);
          break;
        case Displacement::W:
          for (unsigned int i = dim + 1; i < 2 * dim + 1; i++)
            mask.set(i, true);
          break;
        default:
          AssertThrow(false, UnknownDisplacement());
      }
    return mask;
  }



  template <int dim>
  dealii::ComponentMask
  Solver<dim>::get_trace_component_mask(const Component component)
  {
    dealii::ComponentMask mask(2, false);
    switch (component)
      {
        case Component::V:
          mask.set(0, true);
          break;
        case Component::n:
          mask.set(1, true);
          break;
        default:
          AssertThrow(false, UnknownComponent());
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
        default:
          AssertThrow(false, UnknownComponent());
      }
    std::shared_ptr<dealii::Function<dim>> function_extended =
      std::make_shared<FunctionByComponents<dim>>(2 * dim + 2, f_components);
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
          case W: {
            for (unsigned int i = 0; i < dim; i++)
              {
                f_components.insert(
                  {dim + 1 + i,
                   std::make_shared<const ComponentFunction<dim>>(f, i)});
              }
            break;
          }
        default:
          AssertThrow(false, UnknownDisplacement());
      }
    std::shared_ptr<dealii::Function<dim>> function_extended =
      std::make_shared<FunctionByComponents<dim>>(2 * dim + 2, f_components);
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
        default:
          AssertThrow(false, UnknownComponent());
      }
    std::shared_ptr<dealii::Function<dim>> function_extended =
      std::make_shared<FunctionByComponents<dim>>(2, f_components);
    return function_extended;
  }



  template <int dim>
  void
  Solver<dim>::set_V_component(
    const std::shared_ptr<const dealii::Function<dim>> V_function)
  {
    if (!this->initialized)
      this->setup_system();

    auto V_function_extended =
      this->extend_function_on_all_components(V_function, V);
    auto V_function_trace_extended =
      this->extend_function_on_all_trace_components(V_function, V);
    auto V_grad     = std::make_shared<Gradient<dim>>(V_function);
    auto E_function = std::make_shared<Opposite<dim>>(V_grad);
    auto E_function_extended =
      this->extend_function_on_all_components(E_function, E);

    dealii::VectorTools::interpolate(this->dof_handler_local,
                                     *V_function_extended,
                                     this->solution_local,
                                     this->get_component_mask(V));

    dealii::VectorTools::interpolate(this->dof_handler,
                                     *V_function_trace_extended,
                                     this->solution,
                                     this->get_trace_component_mask(V));

    dealii::VectorTools::interpolate(this->dof_handler_local,
                                     *E_function_extended,
                                     this->solution_local,
                                     this->get_component_mask(E));
  }



  template <int dim>
  void
  Solver<dim>::set_n_component(
    const std::shared_ptr<const dealii::Function<dim>> n_function)
  {
    if (!this->initialized)
      this->setup_system();

    auto n_function_extended =
      this->extend_function_on_all_components(n_function, n);
    auto n_function_trace_extended =
      this->extend_function_on_all_trace_components(n_function, n);
    auto n_grad     = std::make_shared<Gradient<dim>>(n_function);
    auto W_function = std::make_shared<Opposite<dim>>(n_grad);
    auto W_function_extended =
      this->extend_function_on_all_components(W_function, W);

    // Set n on the cells
    dealii::VectorTools::interpolate(this->dof_handler_local,
                                     *n_function_extended,
                                     this->solution_local,
                                     this->get_component_mask(n));

    // Set n on the trace
    dealii::VectorTools::interpolate(this->dof_handler,
                                     *n_function_trace_extended,
                                     this->solution,
                                     this->get_trace_component_mask(n));

    // Set W on the cells
    dealii::VectorTools::interpolate(this->dof_handler_local,
                                     *W_function_extended,
                                     this->solution_local,
                                     this->get_component_mask(W));
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
        default:
          AssertThrow(false, UnknownComponent());
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
  Solver<dim>::add_cell_products_to_ll_matrix(
    Ddhdg::Solver<dim>::ScratchData &scratch)
  {
    const unsigned int n_q_points =
      scratch.fe_values_local.get_quadrature().size();
    const unsigned int loc_dofs_per_cell =
      scratch.fe_values_local.get_fe().dofs_per_cell;

    const FEValuesExtractors::Vector electric_field(0);
    const FEValuesExtractors::Scalar electric_potential(dim);
    const FEValuesExtractors::Vector electron_displacement(dim + 1);
    const FEValuesExtractors::Scalar electron_density(2 * dim + 1);

    // Get the values of E on the quadrature points of the cell computed during
    // the previous iteration
    scratch.fe_values_local[electric_field].get_function_values(
      previous_solution_local, scratch.previous_E);

    // The same for n
    scratch.fe_values_local[electron_density].get_function_values(
      previous_solution_local, scratch.previous_n);

    // Get the position of the quadrature points
    for (unsigned int q = 0; q < n_q_points; ++q)
      scratch.cell_quadrature_points[q] =
        scratch.fe_values_local.quadrature_point(q);

    // Compute the value of epsilon
    permittivity->compute_absolute_permittivity(scratch.cell_quadrature_points,
                                                scratch.epsilon_cell);

    // Compute the value of mu
    electron_mobility->compute_electron_mobility(scratch.cell_quadrature_points,
                                                 scratch.mu_cell);

    // Compute the value of T
    temperature->value_list(scratch.cell_quadrature_points, scratch.T_cell);

    // Compute the value of the recombination term and its derivative respect to
    // n
    recombination_term->compute_multiple_recombination_terms(
      scratch.previous_n, scratch.cell_quadrature_points, scratch.r_cell);
    recombination_term->compute_multiple_derivatives_of_recombination_terms(
      scratch.previous_n, scratch.cell_quadrature_points, scratch.dr_cell);

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double JxW = scratch.fe_values_local.JxW(q);
        for (unsigned int k = 0; k < loc_dofs_per_cell; ++k)
          {
            scratch.E[k] = scratch.fe_values_local[electric_field].value(k, q);
            scratch.E_div[k] =
              scratch.fe_values_local[electric_field].divergence(k, q);
            scratch.V[k] =
              scratch.fe_values_local[electric_potential].value(k, q);
            scratch.V_grad[k] =
              scratch.fe_values_local[electric_potential].gradient(k, q);

            scratch.W[k] =
              scratch.fe_values_local[electron_displacement].value(k, q);
            scratch.W_div[k] =
              scratch.fe_values_local[electron_displacement].divergence(k, q);
            scratch.n[k] =
              scratch.fe_values_local[electron_density].value(k, q);
            scratch.n_grad[k] =
              scratch.fe_values_local[electron_density].gradient(k, q);
          }

        const dealii::Tensor<1, dim> mu_times_previous_E =
          scratch.mu_cell[q] * scratch.previous_E[q];

        const dealii::Tensor<2, dim> einstein_diffusion_coefficient =
          Constants::KB / Constants::Q * scratch.T_cell[q] * scratch.mu_cell[q];

        for (unsigned int i = 0; i < loc_dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < loc_dofs_per_cell; ++j)
              {
                scratch.ll_matrix(i, j) +=
                  (-scratch.V[j] * scratch.E_div[i] +
                   scratch.E[j] * scratch.E[i] -
                   (scratch.epsilon_cell[q] * scratch.E[j]) *
                     scratch.V_grad[i] -
                   scratch.n[j] * scratch.V[i] -
                   scratch.n[j] * scratch.W_div[i] +
                   scratch.W[j] * scratch.W[i] -
                   scratch.n[j] * (mu_times_previous_E * scratch.n_grad[i]) +
                   (einstein_diffusion_coefficient * scratch.W[j]) *
                     scratch.n_grad[i] -
                   scratch.dr_cell[q] * scratch.n[j] * scratch.n[i] /
                     Constants::Q) *
                  JxW;
              }
            scratch.l_rhs[i] += (scratch.previous_n[q] * scratch.r_cell[q] -
                                 scratch.previous_n[q] * scratch.dr_cell[q]) /
                                Constants::Q * scratch.n[i] * JxW;
          }
      }
  }

  template <int dim>
  inline void
  Solver<dim>::copy_fe_values_on_scratch(
    Ddhdg::Solver<dim>::ScratchData &scratch,
    const unsigned int               face,
    const unsigned int               q)
  {
    const FEValuesExtractors::Vector electric_field(0);
    const FEValuesExtractors::Scalar electric_potential(dim);
    const FEValuesExtractors::Vector electron_displacement(dim + 1);
    const FEValuesExtractors::Scalar electron_density(2 * dim + 1);

    for (unsigned int k = 0; k < scratch.fe_local_support_on_face[face].size();
         ++k)
      {
        const unsigned int kk = scratch.fe_local_support_on_face[face][k];
        scratch.E[k] =
          scratch.fe_face_values_local[electric_field].value(kk, q);
        scratch.V[k] =
          scratch.fe_face_values_local[electric_potential].value(kk, q);
        scratch.W[k] =
          scratch.fe_face_values_local[electron_displacement].value(kk, q);
        scratch.n[k] =
          scratch.fe_face_values_local[electron_density].value(kk, q);
      }
  }

  template <int dim>
  inline void
  Solver<dim>::copy_fe_values_for_trace(
    Ddhdg::Solver<dim>::ScratchData &scratch,
    const unsigned int               face,
    const unsigned int               q)
  {
    const FEValuesExtractors::Scalar electric_field_trace(0);
    const FEValuesExtractors::Scalar electron_density_trace(1);

    for (unsigned int k = 0; k < scratch.fe_support_on_face[face].size(); ++k)
      {
        scratch.tr_V[k] = scratch.fe_face_values[electric_field_trace].value(
          scratch.fe_support_on_face[face][k], q);
        scratch.tr_n[k] = scratch.fe_face_values[electron_density_trace].value(
          scratch.fe_support_on_face[face][k], q);
      }
  }

  template <int dim>
  inline void
  Solver<dim>::assemble_lf_matrix(Ddhdg::Solver<dim>::ScratchData &scratch,
                                  const unsigned int               face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);
        copy_fe_values_for_trace(scratch, face, q);

        const double         JxW    = scratch.fe_face_values.JxW(q);
        const Tensor<1, dim> normal = scratch.fe_face_values.normal_vector(q);

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
                  (scratch.tr_V[j] * (scratch.E[i] * normal) -
                   parameters->tau * scratch.tr_V[j] * scratch.V[i] +
                   scratch.tr_n[j] * (scratch.W[i] * normal) -
                   parameters->tau * (scratch.tr_n[j] * scratch.n[i])) *
                  JxW;
              }
          }
      }
  }

  template <int dim>
  template <Component c>
  inline void
  Solver<dim>::assemble_fl_matrix(Ddhdg::Solver<dim>::ScratchData &scratch,
                                  const unsigned int               face)
  {
    if (c != V and c != n)
      AssertThrow(false, UnknownComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);
        copy_fe_values_for_trace(scratch, face, q);

        const double         JxW    = scratch.fe_face_values.JxW(q);
        const Tensor<1, dim> normal = scratch.fe_face_values.normal_vector(q);

        const dealii::Tensor<1, dim> mu_times_previous_E =
          scratch.mu_face[q] * scratch.previous_tr_E[q];

        const dealii::Tensor<2, dim> einstein_diffusion_coefficient =
          Constants::KB / Constants::Q * scratch.T_face[q] * scratch.mu_face[q];

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

                // Integrals of the local functions restricted on the
                // border. Here j is the index of the test function
                // (because we are saving the elements in the matrix
                // swapped) The sign is reversed to be used in the
                // Schur complement (and therefore this is the
                // opposite of the right matrix that describes the
                // problem on this cell)
                if (c == V)
                  {
                    scratch.fl_matrix(jj, ii) -=
                      ((scratch.epsilon_face[q] * scratch.E[i]) * normal +
                       parameters->tau * scratch.V[i]) *
                      scratch.tr_V[j] * JxW;
                  }
                if (c == n)
                  {
                    scratch.fl_matrix(jj, ii) -=
                      (scratch.n[i] * (mu_times_previous_E * normal) -
                       (einstein_diffusion_coefficient * scratch.W[i]) *
                         normal +
                       parameters->tau * scratch.n[i]) *
                      scratch.tr_n[j] * JxW;
                  }
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
    if (c != V and c != n)
      AssertThrow(false, UnknownComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);
        copy_fe_values_for_trace(scratch, face, q);

        const double JxW = scratch.fe_face_values.JxW(q);

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

                if (c == V)
                  {
                    task_data.cell_matrix(ii, jj) += -parameters->tau *
                                                     scratch.tr_V[i] *
                                                     scratch.tr_V[j] * JxW;
                  }
                if (c == n)
                  {
                    task_data.cell_matrix(ii, jj) += -parameters->tau *
                                                     scratch.tr_n[i] *
                                                     scratch.tr_n[j] * JxW;
                  }
              }
          }
      }
  }



  template <int dim>
  template <Ddhdg::Component c>
  inline void
  Solver<dim>::apply_dbc_on_face(
    Ddhdg::Solver<dim>::ScratchData &            scratch,
    Ddhdg::Solver<dim>::PerTaskData &            task_data,
    const Ddhdg::DirichletBoundaryCondition<dim> dbc,
    unsigned int                                 face)
  {
    // Is this ok??? I need a trick to have a variable that is scratch.tr_V
    // when I am working with the potential and scratch.tr_n when working with
    // the electron density
    std::vector<double> *tr_c_pointer;
    switch (c)
      {
        case Component::V:
          tr_c_pointer = &scratch.tr_V;
          break;
        case Component::n:
          tr_c_pointer = &scratch.tr_n;
          break;
        default:
          AssertThrow(false, UnknownComponent());
      }
    std::vector<double> &tr_c = *tr_c_pointer;

    if (c != V and c != n)
      AssertThrow(false, UnknownComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);
        copy_fe_values_for_trace(scratch, face, q);

        const double JxW = scratch.fe_face_values.JxW(q);

        const Point<dim> quadrature_point =
          scratch.fe_face_values.quadrature_point(q);
        const double dbc_value = dbc.evaluate(quadrature_point);

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
  Solver<dim>::apply_nbc_on_face(Ddhdg::Solver<dim>::ScratchData &scratch,
                                 Ddhdg::Solver<dim>::PerTaskData &task_data,
                                 unsigned int                     face,
                                 dealii::types::boundary_id       face_id)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    auto boundary_condition =
      boundary_handler->get_neumann_conditions_for_id(face_id, c);

    std::vector<double> *tr_c_pointer;
    switch (c)
      {
        case Component::V:
          tr_c_pointer = &scratch.tr_V;
          break;
        case Component::n:
          tr_c_pointer = &scratch.tr_n;
          break;
        default:
          AssertThrow(false, UnknownComponent());
      }
    std::vector<double> &tr_c = *tr_c_pointer;

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        const double     JxW = scratch.fe_face_values.JxW(q);
        const Point<dim> quadrature_point =
          scratch.fe_face_values.quadrature_point(q);
        const double nbc_value = boundary_condition.evaluate(quadrature_point);
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

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);

        // Integrals on the border of the cell of the local functions
        const double         JxW    = scratch.fe_face_values.JxW(q);
        const Tensor<1, dim> normal = scratch.fe_face_values.normal_vector(q);

        const dealii::Tensor<1, dim> mu_times_previous_E =
          scratch.mu_face[q] * scratch.previous_tr_E[q];

        const dealii::Tensor<2, dim> einstein_diffusion_coefficient =
          Constants::KB / Constants::Q * scratch.T_face[q] * scratch.mu_face[q];

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
                  (scratch.epsilon_face[q] * scratch.E[j] * normal *
                     scratch.V[i] +
                   parameters->tau * scratch.V[j] * scratch.V[i] +
                   mu_times_previous_E * normal * scratch.n[j] * scratch.n[i] -
                   einstein_diffusion_coefficient * scratch.W[j] * normal *
                     scratch.n[i] +
                   parameters->tau * scratch.n[j] * scratch.n[i]) *
                  JxW;
              }
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

        for (unsigned int i = 0;
             i < scratch.fe_local_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii = scratch.fe_local_support_on_face[face][i];
            scratch.l_rhs(ii) +=
              (-scratch.E[i] * normal + scratch.V[i] * parameters->tau) *
              scratch.tr_V_solution_values[q] * JxW;
            scratch.l_rhs(ii) +=
              (-scratch.W[i] * normal + scratch.n[i] * parameters->tau) *
              scratch.tr_n_solution_values[q] * JxW;
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

    // Integrals on the overall cell
    // This function computes every L2 product that we need to compute in order
    // to solve the problem related to the current cell. Moreover, this function
    // also compute the l2 products that are needed to compute the right hand
    // term
    this->add_cell_products_to_ll_matrix(scratch);

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
            const FEValuesExtractors::Scalar electric_field_trace(0);
            const FEValuesExtractors::Scalar electron_density_trace(1);

            scratch.fe_face_values[electric_field_trace].get_function_values(
              solution, scratch.tr_V_solution_values);
            scratch.fe_face_values[electron_density_trace].get_function_values(
              solution, scratch.tr_n_solution_values);
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
        // of epsilon, mu and D_n on the quadrature points of the current face
        const unsigned int n_face_q_points =
          scratch.fe_face_values_local.get_quadrature().size();
        for (unsigned int q = 0; q < n_face_q_points; ++q)
          scratch.face_quadrature_points[q] =
            scratch.fe_face_values.quadrature_point(q);

        const FEValuesExtractors::Vector electric_field(0);
        scratch.fe_face_values_local[electric_field].get_function_values(
          previous_solution_local, scratch.previous_tr_E);

        permittivity->compute_absolute_permittivity(
          scratch.face_quadrature_points, scratch.epsilon_face);

        electron_mobility->compute_electron_mobility(
          scratch.face_quadrature_points, scratch.mu_face);

        temperature->value_list(scratch.face_quadrature_points, scratch.T_face);

        // Assembly the other matrices (the ll_matrix has been assembled
        // calling the add_cell_products_to_ll_matrix method)
        if (!task_data.trace_reconstruct)
          {
            assemble_lf_matrix(scratch, face);

            if (!has_dirichlet_conditions[V])
              {
                assemble_fl_matrix<V>(scratch, face);
                assemble_cell_matrix<V>(scratch, task_data, face);
              }
            else
              {
                const auto dbc =
                  boundary_handler->get_dirichlet_conditions_for_id(
                    face_boundary_id, V);
                apply_dbc_on_face<V>(scratch, task_data, dbc, face);
              }

            if (!has_dirichlet_conditions[n])
              {
                assemble_fl_matrix<n>(scratch, face);
                assemble_cell_matrix<n>(scratch, task_data, face);
              }
            else
              {
                const auto dbc =
                  boundary_handler->get_dirichlet_conditions_for_id(
                    face_boundary_id, n);
                apply_dbc_on_face<n>(scratch, task_data, dbc, face);
              }
          }

        // These are the last terms of the ll matrix, the ones that are
        // generated by L2 products only on the boundary of the cell
        add_border_products_to_ll_matrix(scratch, face);

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
        loc_cell->set_dof_values(scratch.tmp_rhs, solution_local);
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
                                 1e-11 * system_rhs.l2_norm());

    if (parameters->iterative_linear_solver)
      {
        SolverGMRES<> linear_solver(solver_control);
        linear_solver.solve(system_matrix,
                            solution,
                            system_rhs,
                            PreconditionIdentity());
        std::cout << "    Number of GMRES iterations: "
                  << solver_control.last_step() << std::endl;
      }
    else
      {
        SparseDirectUMFPACK Ainv;
        Ainv.initialize(system_matrix);
        Ainv.vmult(solution, system_rhs);
      }
    constraints.distribute(solution);
  }

  template <int dim>
  void
  Solver<dim>::run(const double                         tolerance,
                   const dealii::VectorTools::NormType &norm,
                   const int max_number_of_iterations)
  {
    if (!this->initialized)
      setup_system();

    dealii::Vector<double> difference_per_cell(triangulation->n_active_cells());

    dealii::Vector<double> difference;
    difference.reinit(dof_handler_local.n_dofs());

    for (int step = 1;
         step <= max_number_of_iterations && max_number_of_iterations > 0;
         step++)
      {
        std::cout << "Computing step number " << step << std::endl;
        previous_solution       = solution;
        previous_solution_local = solution_local;
        solution                = 0;
        solution_local          = 0;

        if (parameters->multithreading)
          assemble_system_multithreaded(false);
        else
          assemble_system(false);

        solve_linear_problem();

        if (parameters->multithreading)
          assemble_system_multithreaded(true);
        else
          assemble_system(true);

        difference = 0.;
        difference += solution_local;
        difference -= previous_solution_local;

        VectorTools::integrate_difference(dof_handler_local,
                                          difference,
                                          dealii::Functions::ZeroFunction<dim>(
                                            2 * dim + 2),
                                          difference_per_cell,
                                          QGauss<dim>(fe.degree + 1),
                                          norm);

        const double global_difference_norm =
          VectorTools::compute_global_error(*triangulation,
                                            difference_per_cell,
                                            norm);
        std::cout << "Difference in norm compared to the previous step: "
                  << global_difference_norm << std::endl;
        if (global_difference_norm < tolerance)
          {
            std::cout
              << "Difference is smaller than tolerance. CONVERGENCE REACHED"
              << std::endl;
            break;
          }
      }
  }

  template <int dim>
  void
  Solver<dim>::run(const double tolerance, const int max_number_of_iterations)
  {
    this->run(tolerance,
              parameters->nonlinear_solver_tolerance_norm,
              max_number_of_iterations);
  }

  template <int dim>
  void
  Solver<dim>::run()
  {
    this->run(parameters->nonlinear_solver_tolerance,
              parameters->nonlinear_solver_tolerance_norm,
              parameters->nonlinear_solver_max_number_of_iterations);
  }

  template <int dim>
  double
  Solver<dim>::estimate_l2_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                           expected_solution,
    const Ddhdg::Component c) const
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
        default:
          AssertThrow(false, UnknownComponent())
      }

    const unsigned int n_of_components = (dim + 1) * 2;
    const auto         component_selection =
      dealii::ComponentSelectFunction<dim>(component_index, n_of_components);

    std::map<unsigned int, const std::shared_ptr<const dealii::Function<dim>>>
      c_map;
    c_map.insert({component_index, expected_solution});
    FunctionByComponents<dim> expected_solution_multidim =
      FunctionByComponents<dim>(n_of_components, c_map);

    VectorTools::integrate_difference(dof_handler_local,
                                      solution_local,
                                      expected_solution_multidim,
                                      difference_per_cell,
                                      QGauss<dim>(fe.degree + 1),
                                      VectorTools::L2_norm,
                                      &component_selection);

    const double L2_error =
      VectorTools::compute_global_error(*triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);
    return L2_error;
  }

  template <int dim>
  void
  Solver<dim>::output_results(const std::string &solution_filename) const
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

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      component_interpretation(
        2 * (dim + 1),
        DataComponentInterpretation::component_is_part_of_vector);
    component_interpretation[dim] =
      DataComponentInterpretation::component_is_scalar;
    component_interpretation[2 * dim + 1] =
      DataComponentInterpretation::component_is_scalar;

    data_out.add_data_vector(dof_handler_local,
                             solution_local,
                             names,
                             component_interpretation);

    data_out.build_patches(StaticMappingQ1<dim>::mapping,
                           fe.degree,
                           DataOut<dim>::curved_inner_cells);
    data_out.write_vtk(output);
  }

  template <>
  void
  Solver<1>::output_results(const std::string &solution_filename,
                            const std::string &trace_filename) const
  {
    (void)solution_filename;
    (void)trace_filename;
    AssertThrow(false, NoTraceIn1D());
  }

  template <int dim>
  void
  Solver<dim>::output_results(const std::string &solution_filename,
                              const std::string &trace_filename) const
  {
    output_results(solution_filename);

    std::ofstream            face_output(trace_filename);
    DataOutFaces<dim>        data_out_face(false);
    std::vector<std::string> face_name(2, "u_hat");
    face_name[1] = "v_hat";
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      face_component_type(2, DataComponentInterpretation::component_is_scalar);
    data_out_face.add_data_vector(dof_handler,
                                  solution,
                                  face_name,
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
    unsigned int                                 n_cycles,
    unsigned int                                 initial_refinements)
  {
    const std::shared_ptr<const dealii::Function<dim>> initial_V_function =
      std::make_shared<const dealii::Functions::ZeroFunction<dim>>();
    const std::shared_ptr<const dealii::Function<dim>> initial_n_function =
      std::make_shared<const dealii::Functions::ZeroFunction<dim>>();

    this->print_convergence_table(error_table,
                                  expected_V_solution,
                                  expected_n_solution,
                                  initial_V_function,
                                  initial_n_function,
                                  n_cycles,
                                  initial_refinements);
  }



  template <int dim>
  void
  Solver<dim>::print_convergence_table(
    std::shared_ptr<Ddhdg::ConvergenceTable>     error_table,
    std::shared_ptr<const dealii::Function<dim>> expected_V_solution,
    std::shared_ptr<const dealii::Function<dim>> expected_n_solution,
    std::shared_ptr<const dealii::Function<dim>> initial_V_function,
    std::shared_ptr<const dealii::Function<dim>> initial_n_function,
    unsigned int                                 n_cycles,
    unsigned int                                 initial_refinements)
  {
    this->refine_grid(initial_refinements);


    std::map<unsigned int, const std::shared_ptr<const dealii::Function<dim>>>
      components;
    components.insert({dim, expected_V_solution});
    components.insert({2 * dim + 1, expected_n_solution});
    FunctionByComponents expected_solution(6, components);

    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        this->set_V_component(initial_V_function);
        this->set_n_component(initial_n_function);
        this->run();
        error_table->error_from_exact(dof_handler_local,
                                      solution_local,
                                      expected_solution);

        // this->output_results("solution_" + std::to_string(cycle) + ".vtk",
        //                      "trace_" + std::to_string(cycle) + ".vtk");
        this->refine_grid(1);
      }
    error_table->output_table(std::cout);
  }


  template class Solver<1>;

  template class Solver<2>;

  template class Solver<3>;

} // end of namespace Ddhdg
