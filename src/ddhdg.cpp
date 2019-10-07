#include "ddhdg.h"

#include <deal.II/base/work_stream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>

#include <fstream>


namespace Ddhdg {

    template <int dim>
    Solver<dim>::Solver(
            const Problem<dim>& problem,
            const unsigned int degree
    )
            : triangulation(problem.triangulation)
            , boundary_handler(problem.boundary_handler)
            , f(problem.f)
            , fe_local(FE_DGQ<dim>(degree), dim, FE_DGQ<dim>(degree), 1)
            , dof_handler_local(*triangulation)
            , fe(degree)
            , dof_handler(*triangulation)
    {}



    template <int dim>
    void Solver<dim>::setup_system()
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

        if (boundary_handler->has_dirichlet_boundary_conditions ())
          {
            std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary_map =
                  boundary_handler->get_dirichlet_boundary_map ();
            VectorTools::project_boundary_values (dof_handler,
                                                  dirichlet_boundary_map,
                                                  QGauss<dim - 1> (fe.degree + 1),
                                                  constraints);
          }
        constraints.close();
        {
            DynamicSparsityPattern dsp(dof_handler.n_dofs());
            DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
            sparsity_pattern.copy_from(dsp, fe.dofs_per_face);
        }
        system_matrix.reinit(sparsity_pattern);
    }



    template <int dim>
    void Solver<dim>::assemble_system(bool trace_reconstruct)
    {
        const QGauss<dim>     quadrature_formula(fe.degree + 1);
        const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

        const UpdateFlags local_flags(update_values | update_gradients |
                                      update_JxW_values | update_quadrature_points);
        const UpdateFlags local_face_flags(update_values);
        const UpdateFlags flags(update_values | update_normal_vectors |
                                update_quadrature_points | update_JxW_values);
        PerTaskData task_data(fe.dofs_per_cell, trace_reconstruct);
        ScratchData scratch(fe,
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
    void Solver<dim>::assemble_system_one_cell(
            const typename DoFHandler<dim>::active_cell_iterator &cell,
            ScratchData &                                         scratch,
            PerTaskData &                                         task_data)
    {
        typename DoFHandler<dim>::active_cell_iterator loc_cell(&(*triangulation),
                                                                cell->level(),
                                                                cell->index(),
                                                                &dof_handler_local);
        const double tau_stab = 1.;

        const unsigned int n_q_points =
                scratch.fe_values_local.get_quadrature().size();
        const unsigned int n_face_q_points =
                scratch.fe_face_values_local.get_quadrature().size();
        const unsigned int loc_dofs_per_cell =
                scratch.fe_values_local.get_fe().dofs_per_cell;
        const FEValuesExtractors::Vector fluxes(0);
        const FEValuesExtractors::Scalar scalar(dim);
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
            const double rhs_value = -f->value(q_point);

            const double JxW = scratch.fe_values_local.JxW(q);
            for (unsigned int k = 0; k < loc_dofs_per_cell; ++k)
            {
                scratch.q_phi[k] = scratch.fe_values_local[fluxes].value(k, q);
                scratch.q_phi_div[k] =
                        scratch.fe_values_local[fluxes].divergence(k, q);
                scratch.u_phi[k] = scratch.fe_values_local[scalar].value(k, q);
                scratch.u_phi_grad[k] =
                        scratch.fe_values_local[scalar].gradient(k, q);
            }

            for (unsigned int i = 0; i < loc_dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < loc_dofs_per_cell; ++j)
                    scratch.ll_matrix(i, j) +=
                            (- scratch.q_phi[j] * scratch.q_phi[i] +
                             - scratch.q_phi[j] * scratch.u_phi_grad[i] +
                             + scratch.u_phi[j] * scratch.q_phi_div[i]
                            ) * JxW;
                scratch.l_rhs(i) += scratch.u_phi[i] * rhs_value * JxW;
            }
        }

        // Integrals on the cell border
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
        {
            scratch.fe_face_values_local.reinit(loc_cell, face);
            scratch.fe_face_values.reinit(cell, face);
            if (task_data.trace_reconstruct)
                scratch.fe_face_values.get_function_values(solution,
                                                           scratch.trace_values);

            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
                const double     JxW = scratch.fe_face_values.JxW(q);
                //const Point<dim> quadrature_point =
                //        scratch.fe_face_values.quadrature_point(q);
                const Tensor<1, dim> normal =
                        scratch.fe_face_values.normal_vector(q);

                for (unsigned int k = 0;
                     k < scratch.fe_local_support_on_face[face].size();
                     ++k)
                {
                    const unsigned int kk =
                            scratch.fe_local_support_on_face[face][k];
                    scratch.q_phi[k] =
                            scratch.fe_face_values_local[fluxes].value(kk, q);
                    scratch.u_phi[k] =
                            scratch.fe_face_values_local[scalar].value(kk, q);
                }
                if (!task_data.trace_reconstruct)
                {
                    for (unsigned int k = 0;
                         k < scratch.fe_support_on_face[face].size();
                         ++k)
                        scratch.tr_phi[k] = scratch.fe_face_values.shape_value(
                                scratch.fe_support_on_face[face][k], q);
                    for (unsigned int i = 0;
                         i < scratch.fe_local_support_on_face[face].size();
                         ++i)
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
                                    (-(scratch.q_phi[i] * normal) +
                                     + tau_stab * scratch.u_phi[i]) *
                                    scratch.tr_phi[j] * JxW;

                            // Integrals of the local functions restricted on the border.
                            // Here i is the index of the test function (because we are
                            // saving the elements in the matrix swapped)
                            // The sign is reversed to be used in the Schur complement
                            // (and therefore this is the opposite of the right matrix
                            // that describes the problem on this cell)
                            scratch.fl_matrix(jj, ii) -=
                                    (+ (scratch.q_phi[i] * normal) +
                                     + tau_stab * scratch.u_phi[i]
                                    ) * scratch.tr_phi[j] * JxW;
                        }
                    // Integrals of trace functions (both test and trial)
                    for (unsigned int i = 0;
                         i < scratch.fe_support_on_face[face].size();
                         ++i)
                        for (unsigned int j = 0;
                             j < scratch.fe_support_on_face[face].size();
                             ++j)
                        {
                            const unsigned int ii =
                                    scratch.fe_support_on_face[face][i];
                            const unsigned int jj =
                                    scratch.fe_support_on_face[face][j];
                            task_data.cell_matrix(ii, jj) +=
                                    -tau_stab * scratch.tr_phi[i] *
                                    scratch.tr_phi[j] * JxW;
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
                                (scratch.q_phi[j] * normal +
                                tau_stab * scratch.u_phi[j]) *
                                scratch.u_phi[i] * JxW;
                    }

                if (task_data.trace_reconstruct)
                    for (unsigned int i = 0;
                         i < scratch.fe_local_support_on_face[face].size();
                         ++i)
                    {
                        const unsigned int ii =
                                scratch.fe_local_support_on_face[face][i];
                        scratch.l_rhs(ii)+=
                                (scratch.q_phi[i] * normal +
                                 scratch.u_phi[i] * tau_stab) *
                                scratch.trace_values[q] * JxW;
                    }
            }
        }

        scratch.ll_matrix.gauss_jordan();

        if (! task_data.trace_reconstruct)
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
    void Solver<dim>::copy_local_to_global(const PerTaskData &data)
    {
        if (! data.trace_reconstruct)
            constraints.distribute_local_to_global(data.cell_matrix,
                                                   data.cell_vector,
                                                   data.dof_indices,
                                                   system_matrix,
                                                   system_rhs);
    }



    template <int dim>
    void Solver<dim>::solve()
    {
        SolverControl    solver_control(system_matrix.m() * 10,
                                        1e-11 * system_rhs.l2_norm());
        SolverBicgstab<> linear_solver(solver_control);
        linear_solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
        std::cout << "   Number of BiCGStab iterations: "
                  << solver_control.last_step() << std::endl;
        system_matrix.clear();
        sparsity_pattern.reinit(0, 0, 0, 1);
        constraints.distribute(solution);
        assemble_system(true);
    }



    template <int dim>
    void Solver<dim>::run()
    {
        setup_system();
        assemble_system(false);
        solve();
    }



    template <int dim>
    void Solver<dim>::output_results(const std::string& solution_filename)
    {
        std::ofstream output(solution_filename);
        DataOut<dim> data_out;
        std::vector<std::string> names(dim, "gradient");
        names.emplace_back("solution");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
                component_interpretation(
                dim + 1, DataComponentInterpretation::component_is_part_of_vector);
        component_interpretation[dim] =
                DataComponentInterpretation::component_is_scalar;
        data_out.add_data_vector(dof_handler_local,
                                 solution_local,
                                 names,
                                 component_interpretation);

        data_out.build_patches(fe.degree);
        data_out.write_vtk(output);
    }

    template <>
    void Solver<1>::output_results(const std::string& solution_filename, const std::string& trace_filename)
    {
        (void) solution_filename;
        (void) trace_filename;
        AssertThrow(false, NoTraceIn1D());
    }

    template <int dim>
    void Solver<dim>::output_results(const std::string& solution_filename, const std::string& trace_filename)
    {
        output_results(solution_filename);

        std::ofstream face_output(trace_filename);
        DataOutFaces<dim>        data_out_face(false);
        std::vector<std::string> face_name(1, "u_hat");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
                face_component_type(1, DataComponentInterpretation::component_is_scalar);
        data_out_face.add_data_vector(dof_handler,
                                      solution,
                                      face_name,
                                      face_component_type);
        data_out_face.build_patches(fe.degree);
        data_out_face.write_vtk(face_output);
    }



    template class Solver<1>;
    template class Solver<2>;
    template class Solver<3>;

} // end of namespace Ddhdg
