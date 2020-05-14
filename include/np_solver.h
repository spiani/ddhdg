#include "dof_types.h"
#include "solver.h"

namespace Ddhdg
{
  using namespace dealii;

  DeclExceptionMsg(
    MissingConvergenceForChargeNeutrality,
    "On at least one cell, Newton algorithm has not converged while computing "
    "the local charge neutrality. Please ensure that the magnitude of the "
    "doping is set with a correct value in the adimensionalizer");

  template <int dim>
  class TemplatizedParametersInterface;

  template <int dim, unsigned int parameter_mask>
  class TemplatizedParameters;

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

  template <int dim>
  class NPSolver : public Solver<dim>
  {
  public:
    using Solver<dim>::output_results;

    explicit NPSolver(std::shared_ptr<const Problem<dim>>       problem,
                      std::shared_ptr<const NPSolverParameters> parameters =
                        std::make_shared<NPSolverParameters>(),
                      std::shared_ptr<const Adimensionalizer> adimensionalizer =
                        std::make_shared<Adimensionalizer>());

    void
    refine_grid(unsigned int i) override;

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

    NonlinearIterationResults
    compute_thermodynamic_equilibrium(double absolute_tol,
                                      double relative_tol,
                                      int    max_number_of_iterations) override;

    NonlinearIterationResults
    compute_thermodynamic_equilibrium() override;

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
    get_solution_on_a_point(const dealii::Point<dim> &p,
                            Component                 c) const override;

    dealii::Vector<double>
    get_solution_on_a_point(const dealii::Point<dim> &p,
                            Displacement              d) const override;

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
    generate_dof_to_component_map(std::vector<Component> &dof_to_component_map,
                                  std::vector<DofType> &  dof_to_dof_type,
                                  bool                    for_trace) const;

    void
    project_cell_function_on_trace();

    void
    setup_overall_system();

    void
    setup_restricted_trace_system();

    void
    assemble_system_multithreaded(
      bool reconstruct_trace                 = false,
      bool compute_thermodynamic_equilibrium = false);

    void
    assemble_system(bool reconstruct_trace                 = false,
                    bool compute_thermodynamic_equilibrium = false);

    void
    solve_linear_problem();

    struct PerTaskData;
    struct ScratchData;

    struct ChargeNeutralityScratchData;

    dealii::Threads::Mutex inversion_mutex;

    typedef void (NPSolver<dim>::*assemble_system_one_cell_pointer)(
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
    add_tc_matrix_terms_to_tt_rhs(ScratchData &                      scratch,
                                  Ddhdg::NPSolver<dim>::PerTaskData &task_data,
                                  unsigned int                       face);

    template <typename prm, Component c>
    inline void
    assemble_tt_matrix(ScratchData &scratch,
                       PerTaskData &task_data,
                       unsigned int face);

    template <typename prm, Component c>
    inline void
    add_tt_matrix_terms_to_tt_rhs(ScratchData &                      scratch,
                                  Ddhdg::NPSolver<dim>::PerTaskData &task_data,
                                  unsigned int                       face);

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
    assemble_flux_conditions_wrapper(
      Component                               c,
      ScratchData &                           scratch,
      PerTaskData &                           task_data,
      const std::map<Ddhdg::Component, bool> &has_dirichlet_conditions,
      const std::map<Ddhdg::Component, bool> &has_neumann_conditions,
      types::boundary_id                      face_boundary_id,
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

    NonlinearIterationResults
    private_run(double absolute_tol,
                double relative_tol,
                int    max_number_of_iterations,
                bool   compute_thermodynamic_equilibrium);

    inline void
    compute_local_charge_neutrality_first_guess(
      const std::vector<double> &evaluated_doping,
      const std::vector<double> &evaluated_temperature,
      std::vector<double> &      evaluated_potentials);

    void
    set_local_charge_neutrality_first_guess();

    void
    compute_local_charge_neutrality_copy_solution(
      const dealii::FiniteElement<dim> &V_fe,
      const dealii::DoFHandler<dim> &   V_dof_handler,
      Vector<double> &                  current_solution);

    void
    compute_local_charge_neutrality_single_cell_residual(
      ChargeNeutralityScratchData &scratch,
      const Vector<double> &       V0,
      Vector<double> &             local_residual);

    void
    compute_local_charge_neutrality_single_cell_solve_jacobian(
      ChargeNeutralityScratchData &scratch,
      const Vector<double> &       V0,
      const Vector<double> &       local_residual,
      Vector<double> &             update);

    void
    compute_local_charge_neutrality_for_single_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ChargeNeutralityScratchData &                         scratch,
      Vector<double> &                                      current_solution);

    void
    compute_local_charge_neutrality_nonlinear_solver(
      const dealii::FiniteElement<dim> &V_fe,
      const dealii::DoFHandler<dim> &   V_dof_handler,
      Vector<double> &                  current_solution);

    void
    compute_local_charge_neutrality_set_solution(
      const dealii::FiniteElement<dim> &V_fe,
      const dealii::DoFHandler<dim> &   V_dof_handler,
      Vector<double> &                  current_solution);

    void
    compute_local_charge_neutrality();

    const std::unique_ptr<Triangulation<dim>> triangulation;
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
    SparsityPattern           sparsity_pattern;
    SparseMatrix<double>      system_matrix;

    std::set<Component> enabled_components = {Component::V,
                                              Component::n,
                                              Component::p};

    bool initialized = false;

    friend class TemplatizedParametersInterface<dim>;

    template <int d, unsigned int parameter_mask>
    friend class TemplatizedParameters;
  };

  template <int dim>
  struct NPSolver<dim>::PerTaskData
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
  struct NPSolver<dim>::ScratchData
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
    std::vector<double>                              U_T_cell;
    std::vector<double>                              U_T_face;
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

    template <Component cmp, bool on_face = true>
    inline dealii::Tensor<2, dim>
    compute_einstein_diffusion_coefficient(const unsigned int q) const
    {
      switch (cmp)
        {
          case n:
            if (on_face)
              return this->U_T_face[q] * this->mu_n_face[q];
            return this->U_T_cell[q] * this->mu_n_cell[q];
          case p:
            if (on_face)
              return this->U_T_face[q] * this->mu_p_face[q];
            return this->U_T_cell[q] * this->mu_p_cell[q];
          default:
            Assert(false, InvalidComponent());
            return Tensor<2, dim>();
        }
    }

    template <Component cmp>
    inline double
    compute_stabilized_tau(const double          c_tau,
                           const Tensor<1, dim> &normal,
                           const unsigned int    q) const
    {
      switch (cmp)
        {
          case Component::V:
            return this->epsilon_face[q] * normal * normal * c_tau;
          case Component::n:
            return this->template compute_einstein_diffusion_coefficient<
                     Component::n>(q) *
                   normal * normal * c_tau;
          case Component::p:
            return this->template compute_einstein_diffusion_coefficient<
                     Component::p>(q) *
                   normal * normal * c_tau;
          default:
            Assert(false, InvalidComponent());
            return 1.;
        }
    }
  };

  template <int dim>
  struct NPSolver<dim>::ChargeNeutralityScratchData
  {
    const dealii::FiniteElement<dim> &   V_fe;
    const dealii::DoFHandler<dim> &      V_dof_handler;
    const QGauss<dim>                    quadrature_formula;
    const std::unique_ptr<FEValues<dim>> fe_values;

    LAPACKFullMatrix<double>             jacobian;
    Vector<double>                       V0;
    Vector<double>                       local_update;
    Vector<double>                       local_residual;
    std::vector<types::global_dof_index> dof_indices;

    std::vector<dealii::Point<dim>> quadrature_points;
    std::vector<double>             T;
    std::vector<double>             c;
    std::vector<double>             V0_q;

    std::vector<double> delta_V;

    const double V_rescale;
    const double c_rescale;
    const double Nc;
    const double Nv;
    const double Ec;
    const double Ev;

    ChargeNeutralityScratchData(const dealii::FiniteElement<dim> &V_fe,
                                const dealii::DoFHandler<dim> &   V_dof_handler,
                                const dealii::QGauss<dim> &quadrature_formula,
                                const double               V_rescale,
                                const double               c_rescale,
                                const double               Nc,
                                const double               Nv,
                                const double               Ec,
                                const double               Ev)
      : V_fe(V_fe)
      , V_dof_handler(V_dof_handler)
      , quadrature_formula(quadrature_formula)
      , fe_values(generate_fe_values(V_fe, quadrature_formula))
      , jacobian(V_fe.n_dofs_per_cell(), V_fe.n_dofs_per_cell())
      , V0(V_fe.n_dofs_per_cell())
      , local_update(V_fe.n_dofs_per_cell())
      , local_residual(V_fe.n_dofs_per_cell())
      , dof_indices(V_fe.n_dofs_per_cell())
      , quadrature_points(quadrature_formula.size())
      , T(quadrature_formula.size())
      , c(quadrature_formula.size())
      , V0_q(quadrature_formula.size())
      , delta_V(V_fe.n_dofs_per_cell())
      , V_rescale(V_rescale)
      , c_rescale(c_rescale)
      , Nc(Nc)
      , Nv(Nv)
      , Ec(Ec)
      , Ev(Ev)
    {}

    static std::unique_ptr<FEValues<dim>>
    generate_fe_values(const dealii::FiniteElement<dim> &V_fe,
                       const QGauss<dim> &               quadrature_formula)
    {
      const dealii::UpdateFlags flags_cell(update_values | update_gradients |
                                           update_JxW_values |
                                           update_quadrature_points);

      std::unique_ptr<FEValues<dim>> fe_values =
        std::make_unique<FEValues<dim>>(V_fe, quadrature_formula, flags_cell);

      return fe_values;
    }
  };
} // namespace Ddhdg
