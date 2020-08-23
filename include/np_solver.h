#pragma once

#include <deal.II/numerics/vector_tools.h>

#include "dof_types.h"
#include "local_condenser.h"
#include "solver.h"

namespace pyddhdg
{
  template <int dim>
  class NPSolver;
}

namespace Ddhdg
{
  using namespace dealii;

  template <int dim, class Permittivity>
  class TemplatizedParametersInterface;

  template <int dim, class Permittivity, unsigned int parameter_mask>
  class TemplatizedParameters;

  enum TraceProjectionStrategy
  {
    l2_average,
    reconstruct_problem_solution,
  };

  DeclExceptionMsg(InvalidStrategy, "Invalid strategy for trace projection");

  struct NPSolverParameters
  {
    explicit NPSolverParameters(
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

    NPSolverParameters(const NPSolverParameters &solver) = default;

    const std::map<Component, unsigned int> degree;

    const double nonlinear_solver_absolute_tolerance;
    const double nonlinear_solver_relative_tolerance;
    const int    nonlinear_solver_max_number_of_iterations;

    const std::map<Component, double> tau;
    const bool                        iterative_linear_solver;
    bool                              multithreading;
  };

  template <int dim, class Permittivity>
  class NPSolver : public Solver<dim, Permittivity>
  {
  public:
    using Solver<dim, Permittivity>::output_results;
    using Solver<dim, Permittivity>::compute_thermodynamic_equilibrium;

    explicit NPSolver(std::shared_ptr<const Problem<dim, Permittivity>> problem,
                      std::shared_ptr<const NPSolverParameters> parameters =
                        std::make_shared<NPSolverParameters>(),
                      std::shared_ptr<const Adimensionalizer> adimensionalizer =
                        std::make_shared<Adimensionalizer>(),
                      bool verbose = true);

    template <class OtherPermittivity>
    void
    copy_triangulation_from(const NPSolver<dim, OtherPermittivity> &other)
    {
      this->triangulation = std::make_unique<dealii::Triangulation<dim>>();
      this->triangulation->copy_triangulation(*(other.triangulation));
      this->dof_handler_cell.initialize(*(this->triangulation),
                                        *(this->fe_cell));
      this->dof_handler_trace.initialize(*(this->triangulation),
                                         *(this->fe_trace));
      this->dof_handler_trace_restricted.initialize(
        *(this->triangulation), *(this->fe_trace_restricted));
      this->initialized = false;
    }

    template <class OtherPermittivity>
    void
    copy_solution_from(const NPSolver<dim, OtherPermittivity> &other)
    {
      if (!this->initialized)
        this->setup_overall_system();

      dealii::VectorTools::interpolate_to_different_mesh(
        other.dof_handler_cell,
        other.current_solution_cell,
        this->dof_handler_cell,
        this->current_solution_cell);
    }

    void
    refine_grid(unsigned int i, bool preserve_solution) override;

    void
    refine_and_coarsen_fixed_fraction(
      const Vector<float> &criteria,
      double               top_fraction,
      double               bottom_fraction,
      unsigned int max_n_cells = std::numeric_limits<unsigned int>::max());

    [[nodiscard]] unsigned int
    n_of_triangulation_levels() const
    {
      return this->triangulation->n_levels();
    }

    void
    interpolate_component(
      Component                                    c,
      std::shared_ptr<const dealii::Function<dim>> c_function) override;

    void
    project_component(
      Component                                    c,
      std::shared_ptr<const dealii::Function<dim>> c_function) override;

    void
    set_current_solution(
      std::shared_ptr<const dealii::Function<dim>> V_function,
      std::shared_ptr<const dealii::Function<dim>> n_function,
      std::shared_ptr<const dealii::Function<dim>> p_function,
      bool                                         use_projection) override;

    void
    set_multithreading(bool multithreading = true);

    [[nodiscard]] bool
    is_enabled(Component c) const override;

    void
    enable_component(Component c) override;

    void
    disable_component(Component c) override;

    void
    enable_components(const std::set<Component> &c) override;

    void
    disable_components(const std::set<Component> &c) override;

    void
    set_enabled_components(bool V_enabled,
                           bool n_enabled,
                           bool p_enabled) override;

    NonlinearIterationResults
    run(double absolute_tol,
        double relative_tol,
        int    max_number_of_iterations) override;

    NonlinearIterationResults
    run() override;

    void
    compute_local_charge_neutrality();

    void
    compute_local_charge_neutrality_on_trace(bool only_at_boundary = false);
    NonlinearIterationResults
    compute_thermodynamic_equilibrium(double absolute_tol,
                                      double relative_tol,
                                      int    max_number_of_iterations,
                                      bool   generate_first_guess) override;

    NonlinearIterationResults
    compute_thermodynamic_equilibrium(bool generate_first_guess) override;

    unsigned int
    get_n_dofs(bool for_trace) const override;

    unsigned int
    get_n_active_cells() const override;

    void
    estimate_error_per_cell(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c,
      dealii::VectorTools::NormType                norm,
      dealii::Vector<float> &                      error) const;

    void
    estimate_error_per_cell(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Displacement                                 d,
      dealii::VectorTools::NormType                norm,
      dealii::Vector<float> &                      error) const;

    void
    estimate_error_per_cell(const NPSolver<dim, Permittivity> &other,
                            Component                          c,
                            dealii::VectorTools::NormType      norm,
                            dealii::Vector<float> &            error) const;

    void
    estimate_error_per_cell(const NPSolver<dim, Permittivity> &other,
                            Displacement                       d,
                            dealii::VectorTools::NormType      norm,
                            dealii::Vector<float> &            error) const;

    void
    estimate_error_per_cell(
      dealii::Vector<float> &      error,
      const dealii::ComponentMask &cmp_mask =
        dealii::ComponentMask(all_primary_components().size() * (dim + 1),
                              true)) const;

    void
    estimate_error_per_cell(Component c, dealii::Vector<float> &error) const;

    void
    estimate_error_per_cell(Displacement d, dealii::Vector<float> &error) const;

    double
    estimate_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c,
      dealii::VectorTools::NormType                norm) const override;

    double
    estimate_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Displacement                                 d,
      dealii::VectorTools::NormType                norm) const override;

    double
    estimate_error_on_trace(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c,
      dealii::VectorTools::NormType                norm) const override;

    double
    estimate_l2_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c) const override;

    double
    estimate_l2_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Displacement                                 d) const override;

    double
    estimate_l2_error_on_trace(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c) const override;

    double
    estimate_h1_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c) const override;

    double
    estimate_h1_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Displacement                                 d) const override;

    double
    estimate_linfty_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c) const override;

    double
    estimate_linfty_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Displacement                                 d) const override;

    double
    estimate_linfty_error_on_trace(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c) const override;

    double
    estimate_error(const NPSolver<dim, Permittivity> &other,
                   Component                          c,
                   dealii::VectorTools::NormType      norm) const;

    double
    estimate_error(const NPSolver<dim, Permittivity> &other,
                   Displacement                       d,
                   dealii::VectorTools::NormType      norm) const;

    std::shared_ptr<dealii::Function<dim>>
    get_solution() const override;

    std::shared_ptr<dealii::Function<dim>>
    get_solution(Component c) const override;

    double
    get_solution_on_a_point(const dealii::Point<dim> &p,
                            Component                 c) const override;

    dealii::Vector<double>
    get_solution_on_a_point(const dealii::Point<dim> &p,
                            Displacement              d) const override;

    template <Component cmp>
    void
    compute_current(const dealii::DoFHandler<dim> &dof,
                    dealii::Vector<double> &       data) const;

    void
    output_results(const std::string &solution_filename,
                   bool               save_update) const override;

    void
    output_results(const std::string &solution_filename,
                   const std::string &trace_filename,
                   bool               save_update) const override;

    void
    print_convergence_table(
      std::shared_ptr<Ddhdg::ConvergenceTable>     error_table,
      std::shared_ptr<const dealii::Function<dim>> expected_V_solution,
      std::shared_ptr<const dealii::Function<dim>> expected_n_solution,
      std::shared_ptr<const dealii::Function<dim>> expected_p_solution,
      unsigned int                                 n_cycles,
      unsigned int initial_refinements) override;

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
      unsigned int initial_refinements) override;

  protected:
    static std::unique_ptr<dealii::Triangulation<dim>>
    copy_triangulation(
      std::shared_ptr<const dealii::Triangulation<dim>> triangulation);

    static dealii::FESystem<dim> const *
    generate_fe_system(const std::map<Component, unsigned int> &degree,
                       bool on_trace = false);

    dealii::ComponentMask
    get_component_mask(Component component) const;

    dealii::ComponentMask
    get_component_mask(Displacement displacement) const;

    dealii::ComponentMask
    get_trace_component_mask(Component component,
                             bool      restricted = false) const;

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
    generate_dof_to_component_map(std::vector<Component> &dof_to_component_map,
                                  std::vector<DofType> &  dof_to_dof_type,
                                  bool                    for_trace) const;

    void
    project_cell_function_on_trace(
      const std::set<Component> &components = all_primary_components(),
      TraceProjectionStrategy    strategy   = l2_average);

    template <typename PCScratchData, typename PCCopyData, Component c>
    void
    project_component_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      PCScratchData &                                       scratch,
      PCCopyData &                                          copy_data) const;

    template <typename PCCopyData>
    void
    project_component_copier(PCCopyData &copy_data);

    template <Component c>
    void
    project_component_private(
      std::shared_ptr<const dealii::Function<dim>> c_function);

    unsigned int
    get_dofs_constrained_by_dirichlet_conditions(
      std::vector<bool> &lines) const;

    void
    setup_overall_system();

    void
    setup_restricted_trace_system();

    template <bool multithreading>
    void
    assemble_system(bool reconstruct_trace                 = false,
                    bool compute_thermodynamic_equilibrium = false);

    void
    solve_linear_problem();

    struct PerTaskData;
    struct ScratchData;

    dealii::Threads::Mutex inversion_mutex;

    template <typename prm, bool has_boundary_conditions>
    void
    assemble_system_one_cell_internal(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData &                                         scratch,
      PerTaskData &                                         task_data);

    typedef void (
      NPSolver<dim, Permittivity>::*assemble_system_one_cell_pointer)(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData &                                         scratch,
      PerTaskData &                                         task_data);

    assemble_system_one_cell_pointer
    get_assemble_system_one_cell_function(
      bool compute_thermodynamic_equilibrium);

    template <typename prm>
    void
    assemble_system_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData &                                         scratch,
      PerTaskData &                                         task_data);

    template <Component cmp,
              bool      on_face      = true,
              class ScratchDataClass = ScratchData>
    inline dealii::Tensor<2, dim>
    compute_einstein_diffusion_coefficient(const ScratchDataClass &scratch,
                                           unsigned int            q) const;

    template <Component cmp, class ScratchDataClass = ScratchData>
    inline double
    compute_stabilized_tau(const ScratchDataClass &scratch,
                           double                  c_tau,
                           const Tensor<1, dim> &  normal,
                           unsigned int            q) const;

    template <class ScratchDataClass = ScratchData>
    inline double
    compute_stabilized_tau(const ScratchDataClass &scratch,
                           double                  c_tau,
                           const Tensor<1, dim> &  normal,
                           unsigned int            q,
                           Component               c) const;

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
    add_tc_matrix_terms_to_tt_rhs(
      ScratchData &                                    scratch,
      Ddhdg::NPSolver<dim, Permittivity>::PerTaskData &task_data,
      unsigned int                                     face);

    template <typename prm, Component c>
    inline void
    assemble_tt_matrix(ScratchData &scratch,
                       PerTaskData &task_data,
                       unsigned int face);

    template <typename prm, Component c>
    inline void
    add_tt_matrix_terms_to_tt_rhs(
      ScratchData &                                    scratch,
      Ddhdg::NPSolver<dim, Permittivity>::PerTaskData &task_data,
      unsigned int                                     face);

    template <typename prm, Component c>
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
    assemble_flux_conditions(ScratchData &      scratch,
                             PerTaskData &      task_data,
                             bool               has_dirichlet_conditions,
                             bool               has_neumann_conditions,
                             types::boundary_id face_boundary_id,
                             unsigned int       face);

    template <typename prm>
    inline void
    assemble_flux_conditions_wrapper(Component    c,
                                     ScratchData &scratch,
                                     PerTaskData &task_data,
                                     bool         has_dirichlet_conditions,
                                     bool         has_neumann_conditions,
                                     types::boundary_id face_boundary_id,
                                     unsigned int       face);

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

    inline const Vector<double> &
    get_solution_vector() const
    {
      return this->current_solution_cell;
    }

    NonlinearIterationResults
    private_run(double absolute_tol,
                double relative_tol,
                int    max_number_of_iterations,
                bool   compute_thermodynamic_equilibrium);

    inline void
    compute_local_charge_neutrality_on_a_point(
      const std::vector<double> &evaluated_doping,
      const std::vector<double> &evaluated_temperature,
      std::vector<double> &      evaluated_potentials);

    void
    compute_local_charge_neutrality_on_cells();

    template <typename CTScratchData>
    inline void
    copy_trace_compute_tau(CTScratchData &scratch) const;

    template <typename CTScratchData>
    inline void
    copy_trace_compute_D_n_and_D_p(CTScratchData &scratch) const;

    template <typename IteratorType1,
              typename IteratorType2,
              typename CTScratchData,
              typename CTCopyData>
    void
    copy_trace_face_worker(const IteratorType1 &   cell1,
                           unsigned int            face1,
                           const IteratorType2 &   cell2,
                           unsigned int            face2,
                           CTScratchData &         scratch,
                           CTCopyData &            copy_data,
                           TraceProjectionStrategy strategy) const;

    template <typename IteratorType,
              typename CTScratchData,
              typename CTCopyData>
    void
    copy_trace_boundary_worker(const IteratorType &    cell,
                               unsigned int            face,
                               CTScratchData &         scratch,
                               CTCopyData &            copy_data,
                               TraceProjectionStrategy strategy) const;

    template <typename CTCopyData>
    void
    copy_trace_copier(const CTCopyData &copy_data);

    template <typename IteratorType,
              typename CTScratchData,
              typename CTCopyData,
              TraceProjectionStrategy strategy>
    void
    copy_trace_cell_worker(const IteratorType &cell,
                           CTScratchData &     scratch,
                           CTCopyData &        copy_data);

    std::unique_ptr<Triangulation<dim>>       triangulation;
    const std::unique_ptr<NPSolverParameters> parameters;

    const std::shared_ptr<const dealii::Function<dim>> rescaled_doping;

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
    AffineConstraints<double> global_constraints;
    SparsityPattern           sparsity_pattern;
    SparseMatrix<double>      system_matrix;

    std::set<Component> enabled_components = {Component::V,
                                              Component::n,
                                              Component::p};

    bool initialized = false;

    unsigned int                                 n_dirichlet_constraints;
    std::vector<dealii::types::global_dof_index> constrained_dof_indices;
    std::vector<double>                          constrained_dof_values;

    friend class TemplatizedParametersInterface<dim, Permittivity>;

    template <int d, class p, unsigned int parameter_mask>
    friend class TemplatizedParameters;

    friend class pyddhdg::NPSolver<dim>;
  };

  template <int dim, class Permittivity>
  struct NPSolver<dim, Permittivity>::PerTaskData
  {
    FullMatrix<double>                   tt_matrix;
    Vector<double>                       tt_vector;
    std::vector<types::global_dof_index> dof_indices;

    unsigned int                                 n_dirichlet_constrained_dofs;
    std::vector<dealii::types::global_dof_index> dirichlet_trace_dof_indices;
    std::vector<double>                          dirichlet_trace_dof_values;

    bool trace_reconstruct;

    PerTaskData(const unsigned int n_dofs, const bool trace_reconstruct)
      : tt_matrix(n_dofs, n_dofs)
      , tt_vector(n_dofs)
      , dof_indices(n_dofs)
      , n_dirichlet_constrained_dofs(0)
      , dirichlet_trace_dof_indices(n_dofs)
      , dirichlet_trace_dof_values(n_dofs)
      , trace_reconstruct(trace_reconstruct)
    {}
  };

  template <int dim, class Permittivity>
  struct NPSolver<dim, Permittivity>::ScratchData
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
    Permittivity                                     permittivity;
    std::vector<Point<dim>>                          cell_quadrature_points;
    std::vector<Point<dim>>                          face_quadrature_points;
    std::vector<Tensor<2, dim>>                      mu_n_cell;
    std::vector<Tensor<2, dim>>                      mu_p_cell;
    std::vector<Tensor<2, dim>>                      mu_n_face;
    std::vector<Tensor<2, dim>>                      mu_p_face;
    std::vector<double>                              T_cell;
    std::vector<double>                              T_face;
    std::vector<double>                              U_T_cell;
    std::vector<double>                              U_T_face;
    std::vector<double>                              doping_cell;
    std::vector<double>                              r_cell;
    std::map<Component, std::vector<double>>         dr_cell;
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

    LocalCondenser<dim> local_condenser;

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

    ScratchData(
      const FiniteElement<dim> & fe_trace_restricted,
      const FiniteElement<dim> & fe_trace,
      const FiniteElement<dim> & fe_cell,
      const QGauss<dim> &        quadrature_formula,
      const QGauss<dim - 1> &    face_quadrature_formula,
      UpdateFlags                cell_flags,
      UpdateFlags                cell_face_flags,
      UpdateFlags                trace_flags,
      UpdateFlags                trace_restricted_flags,
      const Permittivity &       permittivity,
      const std::set<Component> &enabled_components,
      const std::map<Component, const dealii::FiniteElement<dim> &> &fe_map);

    ScratchData(const ScratchData &sd);
  };

  template <int dim, class Permittivity>
  template <Component cmp, bool on_face, class ScratchDataClass>
  dealii::Tensor<2, dim>
  NPSolver<dim, Permittivity>::compute_einstein_diffusion_coefficient(
    const ScratchDataClass &scratch,
    const unsigned int      q) const
  {
    switch (cmp)
      {
        case n:
          if (on_face)
            return scratch.U_T_face[q] * scratch.mu_n_face[q];
          return scratch.U_T_cell[q] * scratch.mu_n_cell[q];
        case p:
          if (on_face)
            return scratch.U_T_face[q] * scratch.mu_p_face[q];
          return scratch.U_T_cell[q] * scratch.mu_p_cell[q];
        default:
          Assert(false, InvalidComponent());
          return Tensor<2, dim>();
      }
  }

  template <int dim, class Permittivity>
  template <Component cmp, class ScratchDataClass>
  double
  NPSolver<dim, Permittivity>::compute_stabilized_tau(
    const ScratchDataClass &scratch,
    const double            c_tau,
    const Tensor<1, dim> &  normal,
    const unsigned int      q) const
  {
    switch (cmp)
      {
        case Component::V:
          return scratch.permittivity.compute_stabilized_v_tau(q,
                                                               c_tau,
                                                               normal);
        case Component::n:
          return this->template compute_einstein_diffusion_coefficient<
                   Component::n,
                   true,
                   ScratchDataClass>(scratch, q) *
                 normal * normal * c_tau;
        case Component::p:
          return this->template compute_einstein_diffusion_coefficient<
                   Component::p,
                   true,
                   ScratchDataClass>(scratch, q) *
                 normal * normal * c_tau;
        default:
          Assert(false, InvalidComponent());
          return 1.;
      }
  }

  template <int dim, class Permittivity>
  template <class ScratchDataClass>
  double
  NPSolver<dim, Permittivity>::compute_stabilized_tau(
    const ScratchDataClass &scratch,
    const double            c_tau,
    const Tensor<1, dim> &  normal,
    const unsigned int      q,
    const Component         c) const
  {
    switch (c)
      {
        case Component::V:
          return this->compute_stabilized_tau<Component::V>(scratch,
                                                            c_tau,
                                                            normal,
                                                            q);
        case Component::n:
          return this->compute_stabilized_tau<Component::n>(scratch,
                                                            c_tau,
                                                            normal,
                                                            q);
        case Component::p:
          return this->compute_stabilized_tau<Component::p>(scratch,
                                                            c_tau,
                                                            normal,
                                                            q);
        default:
          Assert(false, InvalidComponent());
          return 1.;
      }
  }
} // namespace Ddhdg
