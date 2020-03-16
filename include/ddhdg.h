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
#include "einstein_diffusion_model.h"
#include "electron_mobility.h"
#include "permittivity.h"
#include "recombination_term.h"

namespace Ddhdg
{
  using namespace dealii;

  DeclExceptionMsg(NoTraceIn1D, "The trace can not be saved in 1D");

  template <int dim>
  struct Problem
  {
    const std::shared_ptr<const Triangulation<dim>>     triangulation;
    const std::shared_ptr<const Permittivity<dim>>      permittivity;
    const std::shared_ptr<const ElectronMobility<dim>>  n_electron_mobility;
    const std::shared_ptr<const RecombinationTerm<dim>> n_recombination_term;
    const std::shared_ptr<const ElectronMobility<dim>>  p_electron_mobility;
    const std::shared_ptr<const RecombinationTerm<dim>> p_recombination_term;
    const std::shared_ptr<const dealii::Function<dim>>  temperature;
    const std::shared_ptr<const dealii::Function<dim>>  doping;
    const std::shared_ptr<const BoundaryConditionHandler<dim>> boundary_handler;

    const EinsteinDiffusionModel einstein_diffusion_model;

    explicit Problem(
      const std::shared_ptr<const Triangulation<dim>>     triangulation,
      const std::shared_ptr<const Permittivity<dim>>      permittivity,
      const std::shared_ptr<const ElectronMobility<dim>>  n_electron_mobility,
      const std::shared_ptr<const RecombinationTerm<dim>> n_recombination_term,
      const std::shared_ptr<const ElectronMobility<dim>>  p_electron_mobility,
      const std::shared_ptr<const RecombinationTerm<dim>> p_recombination_term,
      const std::shared_ptr<const dealii::Function<dim>>  temperature,
      const std::shared_ptr<const dealii::Function<dim>>  doping,
      const std::shared_ptr<const BoundaryConditionHandler<dim>>
                                   boundary_handler,
      const EinsteinDiffusionModel einstein_diffusion_model)
      : triangulation(triangulation)
      , permittivity(permittivity)
      , n_electron_mobility(n_electron_mobility)
      , n_recombination_term(n_recombination_term)
      , p_electron_mobility(p_electron_mobility)
      , p_recombination_term(p_recombination_term)
      , temperature(temperature)
      , doping(doping)
      , boundary_handler(boundary_handler)
      , einstein_diffusion_model(einstein_diffusion_model)
    {}
  };

  struct SolverParameters
  {
    explicit SolverParameters(
      const unsigned int V_degree                                  = 2,
      const unsigned int n_degree                                  = 2,
      const unsigned int p_degree                                  = 2,
      const double       nonlinear_solver_absolute_tolerance       = 1e-10,
      const double       nonlinear_solver_relative_tolerance       = 1e-10,
      const int          nonlinear_solver_max_number_of_iterations = 100,
      const double       V_tau                                     = 1.,
      const double       n_tau                                     = 1.,
      const double       p_tau                                     = 1.,
      const bool         iterative_linear_solver                   = false,
      const bool         multithreading                            = true)
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

    SolverParameters(const SolverParameters &solver) = default;

    const std::map<Component, unsigned int> degree;

    const double nonlinear_solver_absolute_tolerance;
    const double nonlinear_solver_relative_tolerance;
    const int    nonlinear_solver_max_number_of_iterations;

    const std::map<Component, double> tau;
    const bool                        iterative_linear_solver;
    bool                              multithreading;
  };

  struct NonlinearIteratorStatus
  {
    const bool         converged;
    const unsigned int iterations;
    const double       last_update_norm;

    NonlinearIteratorStatus(const bool         converged,
                            const unsigned int iterations,
                            const double       last_update_norm)
      : converged(converged)
      , iterations(iterations)
      , last_update_norm(last_update_norm)
    {}
  };

  template <int dim>
  class Solver
  {
  public:
    explicit Solver(std::shared_ptr<const Problem<dim>>     problem,
                    std::shared_ptr<const SolverParameters> parameters =
                      std::make_shared<SolverParameters>());

    void
    refine_grid(unsigned int i = 1)
    {
      triangulation->refine_global(i);
      this->initialized = false;
    }

    void
    set_component(Component                                    c,
                  std::shared_ptr<const dealii::Function<dim>> c_function,
                  bool use_projection = false);

    void
    set_current_solution(
      std::shared_ptr<const dealii::Function<dim>> V_function,
      std::shared_ptr<const dealii::Function<dim>> n_function,
      std::shared_ptr<const dealii::Function<dim>> p_function,
      bool                                         use_projection = false);

    void
    set_multithreading(bool multithreading = true)
    {
      this->parameters->multithreading = multithreading;
    }

    NonlinearIteratorStatus
    run(double absolute_tol,
        double relative_tol,
        int    max_number_of_iterations = -1);

    NonlinearIteratorStatus
    run();

    double
    estimate_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c,
      dealii::VectorTools::NormType                norm) const;

    double
    estimate_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Displacement                                 d,
      dealii::VectorTools::NormType                norm) const;

    double
    estimate_error_on_trace(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c,
      dealii::VectorTools::NormType                norm) const;

    double
    estimate_l2_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c) const;

    double
    estimate_l2_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Displacement                                 d) const;

    double
    estimate_l2_error_on_trace(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c) const;

    double
    estimate_h1_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c) const;

    double
    estimate_h1_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Displacement                                 d) const;

    double
    estimate_linfty_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c) const;

    double
    estimate_linfty_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Displacement                                 d) const;

    double
    estimate_linfty_error_on_trace(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c) const;

    void
    output_results(const std::string &solution_filename,
                   bool               save_update = false) const;

    void
    output_results(const std::string &solution_filename,
                   const std::string &trace_filename,
                   bool               save_update = false) const;

    void
    print_convergence_table(
      std::shared_ptr<Ddhdg::ConvergenceTable>     error_table,
      std::shared_ptr<const dealii::Function<dim>> expected_V_solution,
      std::shared_ptr<const dealii::Function<dim>> expected_n_solution,
      std::shared_ptr<const dealii::Function<dim>> expected_p_solution,
      unsigned int                                 n_cycles,
      unsigned int                                 initial_refinements = 0);

    void
    print_convergence_table(
      std::shared_ptr<Ddhdg::ConvergenceTable>     error_table,
      std::shared_ptr<const dealii::Function<dim>> expected_V_solution,
      std::shared_ptr<const dealii::Function<dim>> expected_n_solution,
      std::shared_ptr<const dealii::Function<dim>> expected_p_solution,
      std::shared_ptr<const dealii::Function<dim>> initial_V_function,
      std::shared_ptr<const dealii::Function<dim>> initial_n_function,
      std::shared_ptr<const dealii::Function<dim>> initial_p_function,
      unsigned int                                 n_cycles,
      unsigned int                                 initial_refinements = 0);

  private:
    static std::unique_ptr<dealii::Triangulation<dim>>
    copy_triangulation(
      std::shared_ptr<const dealii::Triangulation<dim>> triangulation);

    static dealii::FESystem<dim>
    generate_fe_system(const std::map<Component, unsigned int> &degree,
                       bool                                     local = true);

    dealii::ComponentMask
    get_component_mask(Component component) const;

    dealii::ComponentMask
    get_component_mask(Displacement displacement) const;

    dealii::ComponentMask
    get_trace_component_mask(Component component) const;

    dealii::FEValuesExtractors::Scalar
    get_component_extractor(Component component) const;

    dealii::FEValuesExtractors::Vector
    get_displacement_extractor(Displacement displacement) const;

    dealii::FEValuesExtractors::Scalar
    get_trace_component_extractor(Component component) const;

    std::shared_ptr<dealii::Function<dim>>
    extend_function_on_all_components(
      std::shared_ptr<const dealii::Function<dim>> f,
      Component                                    c) const;

    std::shared_ptr<dealii::Function<dim>>
    extend_function_on_all_components(
      std::shared_ptr<const dealii::Function<dim>> f,
      Displacement                                 d) const;

    std::shared_ptr<dealii::Function<dim>>
    extend_function_on_all_trace_components(
      std::shared_ptr<const dealii::Function<dim>> f,
      Component                                    c) const;

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

    typedef void (Solver<dim>::*assemble_system_one_cell_pointer)(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData &                                         scratch,
      PerTaskData &                                         task_data);

    assemble_system_one_cell_pointer
    get_assemble_system_one_cell_function();

    template <typename prm>
    void
    assemble_system_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData &                                         scratch,
      PerTaskData &                                         task_data);

    void
    prepare_data_on_cell_quadrature_points(ScratchData &scratch);

    template <typename prm>
    void
    add_cell_products_to_ll_matrix(ScratchData &scratch);

    template <typename prm>
    void
    add_cell_products_to_l_rhs(ScratchData &scratch);

    void
    prepare_data_on_face_quadrature_points(ScratchData &scratch);

    inline void
    copy_fe_values_on_scratch(ScratchData &scratch,
                              unsigned int face,
                              unsigned int q);

    inline void
    copy_fe_values_for_trace(ScratchData &scratch,
                             unsigned int face,
                             unsigned int q);

    template <typename prm>
    inline void
    assemble_lf_matrix(ScratchData &scratch, unsigned int face);

    template <typename prm>
    inline void
    add_lf_matrix_terms_to_l_rhs(ScratchData &scratch, unsigned int face);

    template <typename prm, Component c>
    inline void
    assemble_fl_matrix(ScratchData &scratch, unsigned int face);

    template <typename prm, Component c>
    inline void
    add_fl_matrix_terms_to_f_rhs(ScratchData &                    scratch,
                                 Ddhdg::Solver<dim>::PerTaskData &task_data,
                                 unsigned int                     face);

    template <typename prm, Component c>
    inline void
    assemble_cell_matrix(ScratchData &scratch,
                         PerTaskData &task_data,
                         unsigned int face);

    template <typename prm, Component c>
    inline void
    add_cell_matrix_terms_to_f_rhs(ScratchData &                    scratch,
                                   Ddhdg::Solver<dim>::PerTaskData &task_data,
                                   unsigned int                     face);

    template <Component c>
    inline void
    apply_dbc_on_face(ScratchData &                          scratch,
                      PerTaskData &                          task_data,
                      const DirichletBoundaryCondition<dim> &dbc,
                      unsigned int                           face);

    template <Component c>
    inline void
    apply_nbc_on_face(ScratchData &                        scratch,
                      PerTaskData &                        task_data,
                      const NeumannBoundaryCondition<dim> &nbc,
                      unsigned int                         face);

    template <typename prm, Component c>
    inline void
    assemble_flux_conditions(ScratchData &            scratch,
                             PerTaskData &            task_data,
                             bool                     has_dirichlet_conditions,
                             bool                     has_neumann_conditions,
                             const types::boundary_id face_boundary_id,
                             unsigned int             face);

    template <typename prm>
    inline void
    assemble_flux_conditions_wrapper(
      Component                               c,
      ScratchData &                           scratch,
      PerTaskData &                           task_data,
      const std::map<Ddhdg::Component, bool> &has_dirichlet_conditions,
      const std::map<Ddhdg::Component, bool> &has_neumann_conditions,
      const types::boundary_id                face_boundary_id,
      unsigned int                            face);

    template <typename prm>
    inline void
    add_border_products_to_ll_matrix(ScratchData &scratch, unsigned int face);

    template <typename prm>
    inline void
    add_border_products_to_l_rhs(ScratchData &scratch, unsigned int face);

    template <typename prm>
    inline void
    add_trace_terms_to_l_rhs(ScratchData &scratch, unsigned int face);

    void
    copy_local_to_global(const PerTaskData &data);

    const std::unique_ptr<Triangulation<dim>>           triangulation;
    const std::shared_ptr<const Permittivity<dim>>      permittivity;
    const std::shared_ptr<const ElectronMobility<dim>>  n_electron_mobility;
    const std::shared_ptr<const RecombinationTerm<dim>> n_recombination_term;
    const std::shared_ptr<const ElectronMobility<dim>>  p_electron_mobility;
    const std::shared_ptr<const RecombinationTerm<dim>> p_recombination_term;
    const std::shared_ptr<const dealii::Function<dim>>  temperature;
    const std::shared_ptr<const dealii::Function<dim>>  doping;
    const std::shared_ptr<const BoundaryConditionHandler<dim>> boundary_handler;

    EinsteinDiffusionModel einstein_diffusion_model;

    const std::unique_ptr<SolverParameters> parameters;

    FESystem<dim>   fe_local;
    DoFHandler<dim> dof_handler_local;
    Vector<double>  update_local;
    Vector<double>  current_solution_local;
    FESystem<dim>   fe;
    DoFHandler<dim> dof_handler;
    Vector<double>  update;
    Vector<double>  current_solution;
    Vector<double>  system_rhs;

    AffineConstraints<double> constraints;
    SparsityPattern           sparsity_pattern;
    SparseMatrix<double>      system_matrix;

    bool initialized = false;
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
    FEValues<dim>                                    fe_values_local;
    FEFaceValues<dim>                                fe_face_values_local;
    FEFaceValues<dim>                                fe_face_values;
    FullMatrix<double>                               ll_matrix;
    FullMatrix<double>                               lf_matrix;
    FullMatrix<double>                               fl_matrix;
    FullMatrix<double>                               tmp_matrix;
    Vector<double>                                   l_rhs;
    Vector<double>                                   tmp_rhs;
    std::vector<Point<dim>>                          cell_quadrature_points;
    std::vector<Point<dim>>                          face_quadrature_points;
    std::vector<Tensor<2, dim>>                      epsilon_cell;
    std::vector<Tensor<2, dim>>                      epsilon_face;
    std::vector<Tensor<2, dim>>                      mu_n_cell;
    std::vector<Tensor<2, dim>>                      mu_p_cell;
    std::vector<Tensor<2, dim>>                      mu_n_face;
    std::vector<Tensor<2, dim>>                      mu_p_face;
    std::vector<double>                              T_cell;
    std::vector<double>                              T_face;
    std::vector<double>                              doping_cell;
    std::vector<double>                              r_n_cell;
    std::vector<double>                              r_p_cell;
    std::map<Component, std::vector<double>>         dr_n_cell;
    std::map<Component, std::vector<double>>         dr_p_cell;
    std::map<Component, std::vector<double>>         previous_c_cell;
    std::map<Component, std::vector<double>>         previous_c_face;
    std::map<Component, std::vector<Tensor<1, dim>>> previous_f_cell;
    std::map<Component, std::vector<Tensor<1, dim>>> previous_f_face;
    std::map<Component, std::vector<double>>         previous_tr_c_face;
    std::map<Component, std::vector<Tensor<1, dim>>> f;
    std::map<Component, std::vector<double>>         f_div;
    std::map<Component, std::vector<double>>         c;
    std::map<Component, std::vector<Tensor<1, dim>>> c_grad;
    std::map<Component, std::vector<double>>         tr_c;
    std::map<Component, std::vector<double>>         tr_c_solution_values;
    std::vector<std::vector<unsigned int>>           fe_local_support_on_face;
    std::vector<std::vector<unsigned int>>           fe_support_on_face;

    static std::map<Component, std::vector<double>>
    initialize_double_map_on_components(unsigned int n);

    static std::map<Component, std::vector<Tensor<1, dim>>>
    initialize_tensor_map_on_components(unsigned int n);

    static std::map<Component, std::vector<double>>
    initialize_double_map_on_n_and_p(unsigned int k);

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
      , previous_f_cell(
          initialize_tensor_map_on_components(quadrature_formula.size()))
      , previous_f_face(
          initialize_tensor_map_on_components(face_quadrature_formula.size()))
      , previous_tr_c_face(
          initialize_double_map_on_components(face_quadrature_formula.size()))
      , f(initialize_tensor_map_on_components(fe_local.dofs_per_cell))
      , f_div(initialize_double_map_on_components(fe_local.dofs_per_cell))
      , c(initialize_double_map_on_components(fe_local.dofs_per_cell))
      , c_grad(initialize_tensor_map_on_components(fe_local.dofs_per_cell))
      , tr_c(initialize_double_map_on_components(fe.dofs_per_cell))
      , tr_c_solution_values(
          initialize_double_map_on_components(face_quadrature_formula.size()))
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
      , previous_f_cell(sd.previous_f_cell)
      , previous_f_face(sd.previous_f_face)
      , previous_tr_c_face(sd.previous_tr_c_face)
      , f(sd.f)
      , f_div(sd.f_div)
      , c(sd.c)
      , c_grad(sd.c_grad)
      , tr_c(sd.tr_c)
      , tr_c_solution_values(sd.tr_c_solution_values)
      , fe_local_support_on_face(sd.fe_local_support_on_face)
      , fe_support_on_face(sd.fe_support_on_face)
    {}


    template <EinsteinDiffusionModel m, Component cmp, bool on_face = true>
    inline dealii::Tensor<2, dim>
    compute_einstein_diffusion_coefficient(const unsigned int q) const
    {
      switch (m)
        {
          case EinsteinDiffusionModel::M0:
            return Tensor<2, dim>();
          case EinsteinDiffusionModel::M1:
            switch (cmp)
              {
                case n:
                  if (on_face)
                    return Constants::KB / Constants::Q * this->T_face[q] *
                           this->mu_n_face[q];
                  return Constants::KB / Constants::Q * this->T_cell[q] *
                         this->mu_n_cell[q];
                case p:
                  if (on_face)
                    return Constants::KB / Constants::Q * this->T_face[q] *
                           this->mu_p_face[q];
                  return Constants::KB / Constants::Q * this->T_cell[q] *
                         this->mu_p_cell[q];
                default:
                  Assert(false, UnknownComponent());
                  break;
              }
            break;
          default:
            Assert(false, UnknownEinsteinDiffusionModel());
            break;
        }
      return Tensor<2, dim>();
    }

    template <EinsteinDiffusionModel m, Component cmp>
    inline double
    compute_stabilized_tau(const double          c_tau,
                           const Tensor<1, dim> &normal,
                           const unsigned int    q) const
    {
      switch (m)
        {
          case EinsteinDiffusionModel::M0:
            return c_tau;
          case EinsteinDiffusionModel::M1:
            switch (cmp)
              {
                case Component::V:
                  return this->epsilon_face[q] * normal * normal * c_tau;
                case Component::n:
                  return this->template compute_einstein_diffusion_coefficient<
                           EinsteinDiffusionModel::M1,
                           Component::n>(q) *
                         normal * normal * c_tau;
                case Component::p:
                  return this->template compute_einstein_diffusion_coefficient<
                           EinsteinDiffusionModel::M1,
                           Component::p>(q) *
                         normal * normal * c_tau;
                default:
                  Assert(false, UnknownComponent());
              }
          default:
            AssertThrow(false, UnknownEinsteinDiffusionModel());
            break;
        }
      return 1.;
    }
  };
} // end of namespace Ddhdg
