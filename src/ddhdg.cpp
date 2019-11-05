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
    , boundary_handler(problem->boundary_handler)
    , f(problem->f)
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

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

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
      }
    return mask;
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
      }
    return extractor;
  }

  template <int dim>
  void
  Solver<dim>::assemble_system_multithreaded(bool trace_reconstruct)
  {
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
  Solver<dim>::assemble_system_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData &                                         scratch,
    PerTaskData &                                         task_data)
  {
    typename DoFHandler<dim>::active_cell_iterator loc_cell(&(*triangulation),
                                                            cell->level(),
                                                            cell->index(),
                                                            &dof_handler_local);

    const double tau_stab = parameters->tau;

    const unsigned int n_q_points =
      scratch.fe_values_local.get_quadrature().size();
    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();
    const unsigned int loc_dofs_per_cell =
      scratch.fe_values_local.get_fe().dofs_per_cell;
    const FEValuesExtractors::Vector electric_field(0);
    const FEValuesExtractors::Scalar electric_potential(dim);
    const FEValuesExtractors::Vector electron_displacement(dim + 1);
    const FEValuesExtractors::Scalar electron_density(2 * dim + 1);

    const FEValuesExtractors::Scalar electric_field_trace(0);
    const FEValuesExtractors::Scalar electron_density_trace(1);

    bool   V_has_dirichlet_conditions;
    bool   n_has_dirichlet_conditions;
    double V_dbc_value = 0.;
    double n_dbc_value = 0.;

    bool V_has_neumann_conditions;
    bool n_has_neumann_conditions;

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

    // Integrals on the overall cell of elements of the (u, q) space
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const Point<dim> q_point = scratch.fe_values_local.quadrature_point(q);
        const double     rhs_value = -f->value(q_point);

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

        for (unsigned int i = 0; i < loc_dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < loc_dofs_per_cell; ++j)
              scratch.ll_matrix(i, j) += (-scratch.E[j] * scratch.E[i] +
                                          -scratch.E[j] * scratch.V_grad[i] +
                                          +scratch.V[j] * scratch.E_div[i] +
                                          -scratch.W[j] * scratch.W[i] +
                                          -scratch.W[j] * scratch.n_grad[i] +
                                          +scratch.n[j] * scratch.W_div[i]) *
                                         JxW;
            scratch.l_rhs(i) +=
              scratch.V[i] * rhs_value * JxW + scratch.n[i] * rhs_value * JxW;
          }
      }

    // Integrals on the cell border
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
         ++face)
      {
        const dealii::types::boundary_id face_boundary_id =
          cell->face(face)->boundary_id();
        scratch.fe_face_values_local.reinit(loc_cell, face);
        scratch.fe_face_values.reinit(cell, face);
        if (task_data.trace_reconstruct)
          {
            scratch.fe_face_values[electric_field_trace].get_function_values(
              solution, scratch.tr_V_solution_values);
            scratch.fe_face_values[electron_density_trace].get_function_values(
              solution, scratch.tr_n_solution_values);
          }

        // Check if there are some  boundary conditions on the current face
        V_has_dirichlet_conditions = false;
        V_has_neumann_conditions   = false;
        n_has_dirichlet_conditions = false;
        n_has_neumann_conditions   = false;
        if (cell->face(face)->at_boundary())
          {
            if (boundary_handler->has_dirichlet_boundary_conditions(
                  face_boundary_id))
              {
                dirichlet_boundary_map<dim> boundary_map =
                  boundary_handler->get_dirichlet_conditions_for_id(
                    face_boundary_id);
                if (boundary_map.find(V) != boundary_map.end())
                  {
                    V_has_dirichlet_conditions = true;
                  }
                if (boundary_map.find(n) != boundary_map.end())
                  {
                    n_has_dirichlet_conditions = true;
                  }
              }
            if (boundary_handler->has_neumann_boundary_conditions(
                  face_boundary_id))
              {
                neumann_boundary_map<dim> boundary_map =
                  boundary_handler->get_neumann_conditions_for_id(
                    face_boundary_id);
                if (boundary_map.find(V) != boundary_map.end())
                  {
                    V_has_neumann_conditions = true;
                  }
                if (boundary_map.find(n) != boundary_map.end())
                  {
                    n_has_neumann_conditions = true;
                  }
              }
          }

        for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            const double     JxW = scratch.fe_face_values.JxW(q);
            const Point<dim> quadrature_point =
              scratch.fe_face_values.quadrature_point(q);
            const Tensor<1, dim> normal =
              scratch.fe_face_values.normal_vector(q);

            // If there are Dirichlet boundary conditions, evaluate them on the
            // current quadrature point
            if (V_has_dirichlet_conditions || n_has_dirichlet_conditions)
              {
                const auto boundary_map =
                  boundary_handler->get_dirichlet_conditions_for_id(
                    face_boundary_id);
                if (V_has_dirichlet_conditions)
                  V_dbc_value =
                    boundary_map.find(V)->second.evaluate(quadrature_point);
                if (n_has_dirichlet_conditions)
                  n_dbc_value =
                    boundary_map.find(n)->second.evaluate(quadrature_point);
              }

            for (unsigned int k = 0;
                 k < scratch.fe_local_support_on_face[face].size();
                 ++k)
              {
                const unsigned int kk =
                  scratch.fe_local_support_on_face[face][k];
                scratch.E[k] =
                  scratch.fe_face_values_local[electric_field].value(kk, q);
                scratch.V[k] =
                  scratch.fe_face_values_local[electric_potential].value(kk, q);
                scratch.W[k] =
                  scratch.fe_face_values_local[electron_displacement].value(kk,
                                                                            q);
                scratch.n[k] =
                  scratch.fe_face_values_local[electron_density].value(kk, q);
              }
            if (!task_data.trace_reconstruct)
              {
                for (unsigned int k = 0;
                     k < scratch.fe_support_on_face[face].size();
                     ++k)
                  {
                    scratch.tr_V[k] =
                      scratch.fe_face_values[electric_field_trace].value(
                        scratch.fe_support_on_face[face][k], q);
                    scratch.tr_n[k] =
                      scratch.fe_face_values[electron_density_trace].value(
                        scratch.fe_support_on_face[face][k], q);
                  }
                for (unsigned int i = 0;
                     i < scratch.fe_local_support_on_face[face].size();
                     ++i)
                  {
                    for (unsigned int j = 0;
                         j < scratch.fe_support_on_face[face].size();
                         ++j)
                      {
                        const unsigned int ii =
                          scratch.fe_local_support_on_face[face][i];
                        const unsigned int jj =
                          scratch.fe_support_on_face[face][j];

                        // Integrals of trace functions using as test function
                        // the restriction of local test function on the border
                        // i is the index of the test function
                        scratch.lf_matrix(ii, jj) +=
                          (-(scratch.E[i] * normal) +
                           -tau_stab * scratch.V[i]) *
                          scratch.tr_V[j] * JxW;

                        scratch.lf_matrix(ii, jj) +=
                          (-(scratch.W[i] * normal) +
                           -tau_stab * scratch.n[i]) *
                          scratch.tr_n[j] * JxW;

                        // Integrals of the local functions restricted on the
                        // border. Here j is the index of the test function
                        // (because we are saving the elements in the matrix
                        // swapped) The sign is reversed to be used in the Schur
                        // complement (and therefore this is the opposite of the
                        // right matrix that describes the problem on this cell)
                        if (!V_has_dirichlet_conditions)
                          {
                            scratch.fl_matrix(jj, ii) -=
                              (+(scratch.E[i] * normal) +
                               +tau_stab * scratch.V[i]) *
                              scratch.tr_V[j] * JxW;
                          }
                        if (!n_has_dirichlet_conditions)
                          {
                            scratch.fl_matrix(jj, ii) -=
                              (+(scratch.W[i] * normal) +
                               +tau_stab * scratch.n[i]) *
                              scratch.tr_n[j] * JxW;
                          }
                      }
                  }
                // Integrals of trace functions (both test and trial) for
                // component V
                if (V_has_dirichlet_conditions)
                  {
                    for (unsigned int i = 0;
                         i < scratch.fe_support_on_face[face].size();
                         ++i)
                      {
                        const unsigned int ii =
                          scratch.fe_support_on_face[face][i];
                        for (unsigned int j = 0;
                             j < scratch.fe_support_on_face[face].size();
                             ++j)
                          {
                            const unsigned int jj =
                              scratch.fe_support_on_face[face][j];
                            task_data.cell_matrix(ii, jj) +=
                              scratch.tr_V[i] * scratch.tr_V[j] * JxW;
                          }
                        task_data.cell_vector[ii] +=
                          scratch.tr_V[i] * V_dbc_value * JxW;
                      }
                  }
                else
                  {
                    for (unsigned int i = 0;
                         i < scratch.fe_support_on_face[face].size();
                         ++i)
                      {
                        const unsigned int ii =
                          scratch.fe_support_on_face[face][i];
                        for (unsigned int j = 0;
                             j < scratch.fe_support_on_face[face].size();
                             ++j)
                          {
                            const unsigned int jj =
                              scratch.fe_support_on_face[face][j];
                            task_data.cell_matrix(ii, jj) +=
                              -tau_stab * scratch.tr_V[i] * scratch.tr_V[j] *
                              JxW;
                          }
                      }
                  }
                // Integrals of trace functions (both test and trial) for
                // component n
                if (n_has_dirichlet_conditions)
                  {
                    for (unsigned int i = 0;
                         i < scratch.fe_support_on_face[face].size();
                         ++i)
                      {
                        const unsigned int ii =
                          scratch.fe_support_on_face[face][i];
                        for (unsigned int j = 0;
                             j < scratch.fe_support_on_face[face].size();
                             ++j)
                          {
                            const unsigned int jj =
                              scratch.fe_support_on_face[face][j];
                            task_data.cell_matrix(ii, jj) +=
                              scratch.tr_n[i] * scratch.tr_n[j] * JxW;
                          }
                        task_data.cell_vector[ii] +=
                          scratch.tr_n[i] * n_dbc_value * JxW;
                      }
                  }
                else
                  {
                    for (unsigned int i = 0;
                         i < scratch.fe_support_on_face[face].size();
                         ++i)
                      {
                        const unsigned int ii =
                          scratch.fe_support_on_face[face][i];
                        for (unsigned int j = 0;
                             j < scratch.fe_support_on_face[face].size();
                             ++j)
                          {
                            const unsigned int jj =
                              scratch.fe_support_on_face[face][j];
                            task_data.cell_matrix(ii, jj) +=
                              -tau_stab * scratch.tr_n[i] * scratch.tr_n[j] *
                              JxW;
                          }
                      }
                  }
                if (V_has_neumann_conditions)
                  {
                    const auto boundary_map =
                      boundary_handler->get_neumann_conditions_for_id(
                        face_boundary_id);
                    const double nbc_value =
                      boundary_map.find(V)->second.evaluate(quadrature_point);
                    for (unsigned int i = 0;
                         i < scratch.fe_support_on_face[face].size();
                         ++i)
                      {
                        const unsigned int ii =
                          scratch.fe_support_on_face[face][i];
                        task_data.cell_vector(ii) +=
                          scratch.tr_V[i] * nbc_value * JxW;
                      }
                  }
                if (n_has_neumann_conditions)
                  {
                    const auto boundary_map =
                      boundary_handler->get_neumann_conditions_for_id(
                        face_boundary_id);
                    const double nbc_value =
                      boundary_map.find(n)->second.evaluate(quadrature_point);
                    for (unsigned int i = 0;
                         i < scratch.fe_support_on_face[face].size();
                         ++i)
                      {
                        const unsigned int ii =
                          scratch.fe_support_on_face[face][i];
                        task_data.cell_vector(ii) +=
                          scratch.tr_n[i] * nbc_value * JxW;
                      }
                  }
              }
            // Integrals on the border of the cell of the local functions
            for (unsigned int i = 0;
                 i < scratch.fe_local_support_on_face[face].size();
                 ++i)
              for (unsigned int j = 0;
                   j < scratch.fe_local_support_on_face[face].size();
                   ++j)
                {
                  const unsigned int ii =
                    scratch.fe_local_support_on_face[face][i];
                  const unsigned int jj =
                    scratch.fe_local_support_on_face[face][j];
                  scratch.ll_matrix(ii, jj) +=
                    (scratch.E[j] * normal + tau_stab * scratch.V[j]) *
                    scratch.V[i] * JxW;
                  scratch.ll_matrix(ii, jj) +=
                    (scratch.W[j] * normal + tau_stab * scratch.n[j]) *
                    scratch.n[i] * JxW;
                }

            if (task_data.trace_reconstruct)
              for (unsigned int i = 0;
                   i < scratch.fe_local_support_on_face[face].size();
                   ++i)
                {
                  const unsigned int ii =
                    scratch.fe_local_support_on_face[face][i];
                  scratch.l_rhs(ii) +=
                    (scratch.E[i] * normal + scratch.V[i] * tau_stab) *
                    scratch.tr_V_solution_values[q] * JxW;
                  scratch.l_rhs(ii) +=
                    (scratch.W[i] * normal + scratch.n[i] * tau_stab) *
                    scratch.tr_n_solution_values[q] * JxW;
                }
          }
      }

    scratch.ll_matrix.gauss_jordan();

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
  Solver<dim>::solve()
  {
    std::cout << "RHS norm   : " << system_rhs.l2_norm() << std::endl
              << "Matrix norm: " << system_matrix.linfty_norm() << std::endl;

    SolverControl solver_control(system_matrix.m() * 10,
                                 1e-11 * system_rhs.l2_norm());

    if (parameters->iterative_linear_solver)
      {
        SolverGMRES<> linear_solver(solver_control);
        linear_solver.solve(system_matrix,
                            solution,
                            system_rhs,
                            PreconditionIdentity());
        std::cout << "   Number of GMRES iterations: "
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
  Solver<dim>::run()
  {
    setup_system();

    if (parameters->multithreading)
      assemble_system_multithreaded(false);
    else
      assemble_system(false);

    solve();

    if (parameters->multithreading)
      assemble_system_multithreaded(true);
    else
      assemble_system(true);
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
    this->refine_grid(initial_refinements);


    std::map<unsigned int, const std::shared_ptr<const dealii::Function<dim>>>
      components;
    components.insert({dim, expected_V_solution});
    components.insert({2 * dim + 1, expected_n_solution});
    FunctionByComponents expected_solution(6, components);

    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        this->run();
        error_table->error_from_exact(dof_handler_local,
                                      solution_local,
                                      expected_solution);

        // this->output_results("solution_" + std::to_string(cycle) + ".vtk",
        //"trace_" + std::to_string(cycle) + ".vtk");
        this->refine_grid(1);
      }
    error_table->output_table(std::cout);
  }


  template class Solver<1>;

  template class Solver<2>;

  template class Solver<3>;

} // end of namespace Ddhdg
