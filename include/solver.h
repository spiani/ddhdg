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
#include "nonlinear_iteration_results.h"
#include "permittivity.h"
#include "problem.h"
#include "recombination_term.h"

namespace Ddhdg
{
  using namespace dealii;

  DeclExceptionMsg(NoTraceIn1D, "The trace can not be saved in 1D");

  struct SolverParameters
  {
    explicit SolverParameters(
      unsigned int V_degree                                  = 2,
      unsigned int n_degree                                  = 2,
      unsigned int p_degree                                  = 2,
      double       nonlinear_solver_absolute_tolerance       = 1e-10,
      double       nonlinear_solver_relative_tolerance       = 1e-10,
      int          nonlinear_solver_max_number_of_iterations = 100,
      double       V_tau                                     = 1.,
      double       n_tau                                     = 1.,
      double       p_tau                                     = 1.,
      bool         iterative_linear_solver                   = false,
      bool         multithreading                            = true);

    SolverParameters(const SolverParameters &solver) = default;

    const std::map<Component, unsigned int> degree;

    const double nonlinear_solver_absolute_tolerance;
    const double nonlinear_solver_relative_tolerance;
    const int    nonlinear_solver_max_number_of_iterations;

    const std::map<Component, double> tau;
    const bool                        iterative_linear_solver;
    bool                              multithreading;
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
    set_multithreading(bool multithreading = true);

    [[nodiscard]] bool
    is_enabled(Component c) const;

    void
    enable_component(Component c);

    void
    disable_component(Component c);

    void
    enable_components(const std::set<Component> &c);

    void
    disable_components(const std::set<Component> &c);

    void
    set_enabled_components(bool V_enabled, bool n_enabled, bool p_enabled);

    NonlinearIterationResults
    run(double absolute_tol,
        double relative_tol,
        int    max_number_of_iterations = -1);

    NonlinearIterationResults
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
                       bool on_trace = false);

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
    get_trace_component_extractor(Component component,
                                  bool      restricted = false) const;

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

    std::map<Component, unsigned int>
    restrict_degrees_on_enabled_component() const;

    unsigned int
    get_number_of_quadrature_points() const;

    void
    setup_overall_system();

    void
    setup_restricted_trace_system();

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
    add_cell_products_to_cc_matrix(ScratchData &scratch);

    template <typename prm>
    void
    add_cell_products_to_cc_rhs(ScratchData &scratch);

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
    assemble_ct_matrix(ScratchData &scratch, unsigned int face);

    template <typename prm>
    inline void
    add_ct_matrix_terms_to_cc_rhs(ScratchData &scratch, unsigned int face);

    template <typename prm, Component c>
    inline void
    assemble_tc_matrix(ScratchData &scratch, unsigned int face);

    template <typename prm, Component c>
    inline void
    add_tc_matrix_terms_to_tt_rhs(ScratchData &                    scratch,
                                  Ddhdg::Solver<dim>::PerTaskData &task_data,
                                  unsigned int                     face);

    template <typename prm, Component c>
    inline void
    assemble_tt_matrix(ScratchData &scratch,
                       PerTaskData &task_data,
                       unsigned int face);

    template <typename prm, Component c>
    inline void
    add_tt_matrix_terms_to_tt_rhs(ScratchData &                    scratch,
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
    add_border_products_to_cc_matrix(ScratchData &scratch, unsigned int face);

    template <typename prm>
    inline void
    add_border_products_to_cc_rhs(ScratchData &scratch, unsigned int face);

    template <typename prm>
    inline void
    add_trace_terms_to_cc_rhs(ScratchData &scratch, unsigned int face);

    void
    copy_local_to_global(const PerTaskData &data);

    void
    build_restricted_to_trace_dof_map();

    void
    copy_restricted_to_trace();

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

    std::unique_ptr<const FESystem<dim>> fe_cell;
    DoFHandler<dim>                      dof_handler_cell;
    Vector<double>                       update_cell;
    Vector<double>                       current_solution_cell;
    std::unique_ptr<const FESystem<dim>> fe_trace;
    std::unique_ptr<const FESystem<dim>> fe_trace_restricted;
    DoFHandler<dim>                      dof_handler_trace;
    DoFHandler<dim>                      dof_handler_trace_restricted;
    Vector<double>                       update_trace;
    Vector<double>                       current_solution_trace;
    Vector<double>                       system_rhs;
    Vector<double>                       system_solution;
    std::vector<unsigned int>            restricted_to_trace_dof_map;


    AffineConstraints<double> constraints;
    SparsityPattern           sparsity_pattern;
    SparseMatrix<double>      system_matrix;

    std::set<Component> enabled_components = {Component::V,
                                              Component::n,
                                              Component::p};

    bool initialized = false;
  };

  template <int dim>
  struct Solver<dim>::PerTaskData
  {
    FullMatrix<double>                   tt_matrix;
    Vector<double>                       tt_vector;
    std::vector<types::global_dof_index> dof_indices;
    bool                                 trace_reconstruct;

    PerTaskData(const unsigned int n_dofs, const bool trace_reconstruct)
      : tt_matrix(n_dofs, n_dofs)
      , tt_vector(n_dofs)
      , dof_indices(n_dofs)
      , trace_reconstruct(trace_reconstruct)
    {}
  };

  template <int dim>
  struct Solver<dim>::ScratchData
  {
    FEValues<dim>                   fe_values_cell;
    FEFaceValues<dim>               fe_face_values_cell;
    FEFaceValues<dim>               fe_face_values_trace;
    FEFaceValues<dim>               fe_face_values_trace_restricted;
    const std::vector<unsigned int> enabled_component_indices;
    const std::vector<std::vector<unsigned int>>     fe_cell_support_on_face;
    const std::vector<std::vector<unsigned int>>     fe_trace_support_on_face;
    const unsigned int                               dofs_on_enabled_components;
    FullMatrix<double>                               cc_matrix;
    FullMatrix<double>                               ct_matrix;
    FullMatrix<double>                               tc_matrix;
    FullMatrix<double>                               tmp_matrix;
    Vector<double>                                   cc_rhs;
    Vector<double>                                   tmp_rhs;
    Vector<double>                                   restricted_tmp_rhs;
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
    std::map<Component, std::vector<Tensor<1, dim>>> previous_d_cell;
    std::map<Component, std::vector<Tensor<1, dim>>> previous_d_face;
    std::map<Component, std::vector<double>>         previous_tr_c_face;
    std::map<Component, std::vector<Tensor<1, dim>>> d;
    std::map<Component, std::vector<double>>         d_div;
    std::map<Component, std::vector<double>>         c;
    std::map<Component, std::vector<Tensor<1, dim>>> c_grad;
    std::map<Component, std::vector<double>>         tr_c;
    std::map<Component, std::vector<double>>         tr_c_solution_values;

    static std::map<Component, std::vector<double>>
    initialize_double_map_on_components(unsigned int n);

    static std::map<Component, std::vector<Tensor<1, dim>>>
    initialize_tensor_map_on_components(unsigned int n);

    static std::map<Component, std::vector<double>>
    initialize_double_map_on_n_and_p(unsigned int k);

    static std::vector<unsigned int>
    check_dofs_on_enabled_components(
      const FiniteElement<dim> & fe_cell,
      const std::set<Component> &enabled_components);

    static std::vector<std::vector<unsigned int>>
    check_dofs_on_faces_for_cells(
      const FiniteElement<dim> &       fe_cell,
      const std::vector<unsigned int> &enabled_component_indices);

    static std::vector<std::vector<unsigned int>>
    check_dofs_on_faces_for_trace(
      const FiniteElement<dim> &fe_trace_restricted);

    ScratchData(const FiniteElement<dim> & fe_trace_restricted,
                const FiniteElement<dim> & fe_trace,
                const FiniteElement<dim> & fe_cell,
                const QGauss<dim> &        quadrature_formula,
                const QGauss<dim - 1> &    face_quadrature_formula,
                UpdateFlags                cell_flags,
                UpdateFlags                cell_face_flags,
                UpdateFlags                trace_flags,
                UpdateFlags                trace_restricted_flags,
                const std::set<Component> &enabled_components);

    ScratchData(const ScratchData &sd);

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
                  Assert(false, InvalidComponent());
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
                  Assert(false, InvalidComponent());
              }
          default:
            AssertThrow(false, UnknownEinsteinDiffusionModel());
            break;
        }
      return 1.;
    }
  };

} // namespace Ddhdg
