#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/thread_management.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/chunk_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include <iostream>

#include "boundary_conditions.h"
#include "convergence_table.h"
#include "electron_mobility.h"
#include "permittivity.h"

namespace Ddhdg
{
  using namespace dealii;

  DeclExceptionMsg(NoTraceIn1D, "The trace can not be saved in 1D");

  template <int dim>
  struct Problem
  {
    const std::shared_ptr<const Triangulation<dim>>    triangulation;
    const std::shared_ptr<const Permittivity<dim>>     permittivity;
    const std::shared_ptr<const ElectronMobility<dim>> electron_mobility;
    const std::shared_ptr<const dealii::Function<dim>> temperature;
    const std::shared_ptr<const BoundaryConditionHandler<dim>> boundary_handler;

    explicit Problem(
      const std::shared_ptr<const Triangulation<dim>>    triangulation,
      const std::shared_ptr<const Permittivity<dim>>     permittivity,
      const std::shared_ptr<const ElectronMobility<dim>> electron_mobility,
      const std::shared_ptr<const dealii::Function<dim>> temperature,
      const std::shared_ptr<const BoundaryConditionHandler<dim>>
        boundary_handler)
      : triangulation(triangulation)
      , permittivity(permittivity)
      , electron_mobility(electron_mobility)
      , temperature(temperature)
      , boundary_handler(boundary_handler)
    {}
  };

  struct SolverParameters
  {
    SolverParameters(unsigned int V_degree,
                     unsigned int n_degree,
                     double       nonlinear_solver_tolerance,
                     int          nonlinear_solver_max_number_of_iterations,
                     VectorTools::NormType norm_type = VectorTools::H1_norm,
                     double                tau       = 1.,
                     bool                  iterative_linear_solver = false,
                     bool                  multithreading          = true)
      : V_degree(V_degree)
      , n_degree(n_degree)
      , nonlinear_solver_tolerance(nonlinear_solver_tolerance)
      , nonlinear_solver_max_number_of_iterations(
          nonlinear_solver_max_number_of_iterations)
      , nonlinear_solver_tolerance_norm(norm_type)
      , tau(tau)
      , iterative_linear_solver(iterative_linear_solver)
      , multithreading(multithreading)
    {}

    SolverParameters(unsigned int degree,
                     double       nonlinear_solver_tolerance,
                     int          nonlinear_solver_max_number_of_iterations,
                     VectorTools::NormType norm_type = VectorTools::H1_norm,
                     double                tau       = 1.,
                     bool                  iterative_linear_solver = false,
                     bool                  multithreading          = true)
      : V_degree(degree)
      , n_degree(degree)
      , nonlinear_solver_tolerance(nonlinear_solver_tolerance)
      , nonlinear_solver_max_number_of_iterations(
          nonlinear_solver_max_number_of_iterations)
      , nonlinear_solver_tolerance_norm(norm_type)
      , tau(tau)
      , iterative_linear_solver(iterative_linear_solver)
      , multithreading(multithreading)
    {}

    const unsigned int V_degree;
    const unsigned int n_degree;

    const double                nonlinear_solver_tolerance;
    const int                   nonlinear_solver_max_number_of_iterations;
    const VectorTools::NormType nonlinear_solver_tolerance_norm;

    const double tau                     = 1.;
    const bool   iterative_linear_solver = false;
    const bool   multithreading          = true;
  };

  template <int dim>
  class Solver
  {
  public:
    Solver(std::shared_ptr<const Problem<dim>>     problem,
           std::shared_ptr<const SolverParameters> parameters);

    void
    refine_grid(unsigned int i = 1)
    {
      triangulation->refine_global(i);
      this->setup_system();
    }

    void
    run(double                               tolerance,
        const dealii::VectorTools::NormType &norm,
        int                                  max_number_of_iterations = -1);

    void
    run(double tolerance, int max_number_of_iterations = -1);

    void
    run();

    double
    estimate_l2_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c) const;

    void
    output_results(const std::string &solution_filename) const;

    void
    output_results(const std::string &solution_filename,
                   const std::string &trace_filename) const;

    void
    print_convergence_table(
      std::shared_ptr<Ddhdg::ConvergenceTable>     error_table,
      std::shared_ptr<const dealii::Function<dim>> expected_V_solution,
      std::shared_ptr<const dealii::Function<dim>> expected_n_solution,
      unsigned int                                 n_cycles,
      unsigned int                                 initial_refinements = 0);

  private:
    dealii::ComponentMask
    get_component_mask(Component component);

    dealii::FEValuesExtractors::Scalar
    get_component_extractor(Component component);

    static std::unique_ptr<dealii::Triangulation<dim>>
    copy_triangulation(
      std::shared_ptr<const dealii::Triangulation<dim>> triangulation);

    void
    setup_system();

    void
    assemble_system_multithreaded(bool reconstruct_trace = false);

    void
    assemble_system(bool reconstruct_trace = false);

    void
    solve_linear_problem();

    struct PerTaskData;
    struct ScratchData;
    dealii::Threads::Mutex inversion_mutex;

    void
    assemble_system_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData &                                         scratch,
      PerTaskData &                                         task_data);

    void
    add_cell_products_to_ll_matrix(ScratchData &scratch);

    inline void
    copy_fe_values_on_scratch(ScratchData &scratch,
                              unsigned int face,
                              unsigned int q);

    inline void
    copy_fe_values_for_trace(ScratchData &scratch,
                             unsigned int face,
                             unsigned int q);

    inline void
    assemble_lf_matrix(ScratchData &scratch, unsigned int face);

    template <Component c>
    inline void
    assemble_fl_matrix(ScratchData &scratch, unsigned int face);

    template <Component c>
    inline void
    assemble_cell_matrix(ScratchData &scratch,
                         PerTaskData &task_data,
                         unsigned int face);

    template <Component c>
    inline void
    apply_dbc_on_face(ScratchData &                         scratch,
                      PerTaskData &                         task_data,
                      const DirichletBoundaryCondition<dim> dbc,
                      unsigned int                          face);

    template <Component c>
    inline void
    apply_nbc_on_face(ScratchData &              scratch,
                      PerTaskData &              task_data,
                      unsigned int               face,
                      dealii::types::boundary_id face_id);

    inline void
    add_border_products_to_ll_matrix(ScratchData &scratch, unsigned int face);

    inline void
    add_trace_terms_to_l_rhs(ScratchData &scratch, unsigned int face);

    void
    copy_local_to_global(const PerTaskData &data);

    const std::unique_ptr<Triangulation<dim>>          triangulation;
    const std::shared_ptr<const Permittivity<dim>>     permittivity;
    const std::shared_ptr<const ElectronMobility<dim>> electron_mobility;
    const std::shared_ptr<const dealii::Function<dim>> temperature;
    const std::shared_ptr<const BoundaryConditionHandler<dim>> boundary_handler;

    const std::shared_ptr<const SolverParameters> parameters;

    FESystem<dim>   fe_local;
    DoFHandler<dim> dof_handler_local;
    Vector<double>  solution_local;
    Vector<double>  previous_solution_local;
    FESystem<dim>   fe;
    DoFHandler<dim> dof_handler;
    Vector<double>  solution;
    Vector<double>  previous_solution;
    Vector<double>  system_rhs;

    AffineConstraints<double> constraints;
    SparsityPattern           sparsity_pattern;
    SparseMatrix<double>      system_matrix;
  };

  template <int dim>
  struct Solver<dim>::PerTaskData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_vector;
    std::vector<types::global_dof_index> dof_indices;
    bool                                 trace_reconstruct;

    PerTaskData(const unsigned int n_dofs, const bool trace_reconstruct)
      : cell_matrix(n_dofs, n_dofs)
      , cell_vector(n_dofs)
      , dof_indices(n_dofs)
      , trace_reconstruct(trace_reconstruct)
    {}
  };

  template <int dim>
  struct Solver<dim>::ScratchData
  {
    FEValues<dim>                          fe_values_local;
    FEFaceValues<dim>                      fe_face_values_local;
    FEFaceValues<dim>                      fe_face_values;
    FullMatrix<double>                     ll_matrix;
    FullMatrix<double>                     lf_matrix;
    FullMatrix<double>                     fl_matrix;
    FullMatrix<double>                     tmp_matrix;
    Vector<double>                         l_rhs;
    Vector<double>                         tmp_rhs;
    std::vector<Point<dim>>                cell_quadrature_points;
    std::vector<Point<dim>>                face_quadrature_points;
    std::vector<Tensor<2, dim>>            epsilon_cell;
    std::vector<Tensor<2, dim>>            epsilon_face;
    std::vector<Tensor<2, dim>>            mu_cell;
    std::vector<Tensor<2, dim>>            mu_face;
    std::vector<double>                    T_cell;
    std::vector<double>                    T_face;
    std::vector<Tensor<1, dim>>            previous_E;
    std::vector<Tensor<1, dim>>            previous_tr_E;
    std::vector<Tensor<1, dim>>            E;
    std::vector<double>                    E_div;
    std::vector<double>                    V;
    std::vector<Tensor<1, dim>>            V_grad;
    std::vector<double>                    tr_V;
    std::vector<Tensor<1, dim>>            W;
    std::vector<double>                    W_div;
    std::vector<double>                    n;
    std::vector<Tensor<1, dim>>            n_grad;
    std::vector<double>                    tr_n;
    std::vector<double>                    tr_V_solution_values;
    std::vector<double>                    tr_n_solution_values;
    std::vector<std::vector<unsigned int>> fe_local_support_on_face;
    std::vector<std::vector<unsigned int>> fe_support_on_face;

    ScratchData(const FiniteElement<dim> &fe,
                const FiniteElement<dim> &fe_local,
                const QGauss<dim> &       quadrature_formula,
                const QGauss<dim - 1> &   face_quadrature_formula,
                const UpdateFlags         local_flags,
                const UpdateFlags         local_face_flags,
                const UpdateFlags         flags)
      : fe_values_local(fe_local, quadrature_formula, local_flags)
      , fe_face_values_local(fe_local,
                             face_quadrature_formula,
                             local_face_flags)
      , fe_face_values(fe, face_quadrature_formula, flags)
      , ll_matrix(fe_local.dofs_per_cell, fe_local.dofs_per_cell)
      , lf_matrix(fe_local.dofs_per_cell, fe.dofs_per_cell)
      , fl_matrix(fe.dofs_per_cell, fe_local.dofs_per_cell)
      , tmp_matrix(fe.dofs_per_cell, fe_local.dofs_per_cell)
      , l_rhs(fe_local.dofs_per_cell)
      , tmp_rhs(fe_local.dofs_per_cell)
      , cell_quadrature_points(quadrature_formula.size())
      , face_quadrature_points(face_quadrature_formula.size())
      , epsilon_cell(quadrature_formula.size())
      , epsilon_face(face_quadrature_formula.size())
      , mu_cell(quadrature_formula.size())
      , mu_face(face_quadrature_formula.size())
      , T_cell(quadrature_formula.size())
      , T_face(face_quadrature_formula.size())
      , previous_E(quadrature_formula.size())
      , previous_tr_E(face_quadrature_formula.size())
      , E(fe_local.dofs_per_cell)
      , E_div(fe_local.dofs_per_cell)
      , V(fe_local.dofs_per_cell)
      , V_grad(fe_local.dofs_per_cell)
      , tr_V(fe.dofs_per_cell)
      , W(fe_local.dofs_per_cell)
      , W_div(fe_local.dofs_per_cell)
      , n(fe_local.dofs_per_cell)
      , n_grad(fe_local.dofs_per_cell)
      , tr_n(fe.dofs_per_cell)
      , tr_V_solution_values(face_quadrature_formula.size())
      , tr_n_solution_values(face_quadrature_formula.size())
      , fe_local_support_on_face(GeometryInfo<dim>::faces_per_cell)
      , fe_support_on_face(GeometryInfo<dim>::faces_per_cell)
    {
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        for (unsigned int i = 0; i < fe_local.dofs_per_cell; ++i)
          {
            if (fe_local.has_support_on_face(i, face))
              fe_local_support_on_face[face].push_back(i);
          }
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          {
            if (fe.has_support_on_face(i, face))
              fe_support_on_face[face].push_back(i);
          }
    }

    ScratchData(const ScratchData &sd)
      : fe_values_local(sd.fe_values_local.get_fe(),
                        sd.fe_values_local.get_quadrature(),
                        sd.fe_values_local.get_update_flags())
      , fe_face_values_local(sd.fe_face_values_local.get_fe(),
                             sd.fe_face_values_local.get_quadrature(),
                             sd.fe_face_values_local.get_update_flags())
      , fe_face_values(sd.fe_face_values.get_fe(),
                       sd.fe_face_values.get_quadrature(),
                       sd.fe_face_values.get_update_flags())
      , ll_matrix(sd.ll_matrix)
      , lf_matrix(sd.lf_matrix)
      , fl_matrix(sd.fl_matrix)
      , tmp_matrix(sd.tmp_matrix)
      , l_rhs(sd.l_rhs)
      , tmp_rhs(sd.tmp_rhs)
      , cell_quadrature_points(sd.cell_quadrature_points)
      , face_quadrature_points(sd.face_quadrature_points)
      , epsilon_cell(sd.epsilon_cell)
      , epsilon_face(sd.epsilon_face)
      , mu_cell(sd.mu_cell)
      , mu_face(sd.mu_face)
      , T_cell(sd.T_cell)
      , T_face(sd.T_face)
      , previous_E(sd.previous_E)
      , previous_tr_E(sd.previous_tr_E)
      , E(sd.E)
      , E_div(sd.E_div)
      , V(sd.V)
      , V_grad(sd.V_grad)
      , tr_V(sd.tr_V)
      , W(sd.W)
      , W_div(sd.W_div)
      , n(sd.n)
      , n_grad(sd.n_grad)
      , tr_n(sd.tr_n)
      , tr_V_solution_values(sd.tr_V_solution_values)
      , tr_n_solution_values(sd.tr_n_solution_values)
      , fe_local_support_on_face(sd.fe_local_support_on_face)
      , fe_support_on_face(sd.fe_support_on_face)
    {}
  };

} // end of namespace Ddhdg
