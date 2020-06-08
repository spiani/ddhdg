#pragma once

#include <deal.II/grid/grid_generator.h>

#include "boundary_conditions.h"
#include "ddhdg.h"

namespace pyddhdg
{
  template <int dim>
  class HomogeneousPermittivity
  {
  public:
    explicit HomogeneousPermittivity(double epsilon);

    std::shared_ptr<Ddhdg::HomogeneousPermittivity<dim>>
    generate_ddhdg_permittivity();

    const double epsilon;
  };

  template <int dim>
  class ElectronMobility
  {
  public:
    virtual ~ElectronMobility() = default;

    virtual std::shared_ptr<Ddhdg::ElectronMobility<dim>>
    generate_ddhdg_electron_mobility() = 0;
  };

  template <int dim>
  class HomogeneousElectronMobility : public ElectronMobility<dim>
  {
  public:
    explicit HomogeneousElectronMobility(double mu);

    virtual std::shared_ptr<Ddhdg::ElectronMobility<dim>>
    generate_ddhdg_electron_mobility();

    const double mu;
  };

  template <int dim>
  class DealIIFunction
  {
  public:
    explicit DealIIFunction(std::shared_ptr<dealii::Function<dim>> f);

    explicit DealIIFunction(double f_const);

    std::shared_ptr<dealii::Function<dim>>
    get_dealii_function() const;

  private:
    const std::shared_ptr<dealii::Function<dim>> f;
  };

  template <int dim>
  class AnalyticFunction : public DealIIFunction<dim>
  {
  public:
    explicit AnalyticFunction(std::string f_expr);

    [[nodiscard]] std::string
    get_expression() const;

  private:
    static std::shared_ptr<dealii::FunctionParser<dim>>
    get_function_from_string(const std::string &f_expr);

    const std::string f_expr;
  };

  template <int dim>
  class PiecewiseFunction : public DealIIFunction<dim>
  {
  public:
    PiecewiseFunction(const DealIIFunction<dim> &condition,
                      const DealIIFunction<dim> &f1,
                      const DealIIFunction<dim> &f2);

    PiecewiseFunction(const std::string &condition,
                      const std::string &f1,
                      const std::string &f2);

    PiecewiseFunction(const std::string &condition,
                      const std::string &f1,
                      double             f2);

    PiecewiseFunction(const std::string &condition,
                      double             f1,
                      const std::string &f2);

    PiecewiseFunction(const std::string &condition, double f1, double f2);
  };

  template <int dim>
  class RecombinationTerm
  {
  public:
    virtual ~RecombinationTerm() = default;

    virtual std::shared_ptr<Ddhdg::RecombinationTerm<dim>>
    generate_ddhdg_recombination_term() = 0;
  };

  template <int dim>
  class LinearRecombinationTerm : public RecombinationTerm<dim>
  {
  public:
    LinearRecombinationTerm(const DealIIFunction<dim> &zero_term,
                            const DealIIFunction<dim> &n_linear_coefficient,
                            const DealIIFunction<dim> &p_linear_coefficient);

    std::shared_ptr<Ddhdg::RecombinationTerm<dim>>
    generate_ddhdg_recombination_term() override;

    [[nodiscard]] DealIIFunction<dim>
    get_constant_term() const;

    [[nodiscard]] DealIIFunction<dim>
    get_n_linear_coefficient() const;

    [[nodiscard]] DealIIFunction<dim>
    get_p_linear_coefficient() const;

  private:
    const DealIIFunction<dim> zero_term;
    const DealIIFunction<dim> n_linear_coefficient;
    const DealIIFunction<dim> p_linear_coefficient;
  };

  template <int dim>
  class BoundaryConditionHandler
  {
  public:
    BoundaryConditionHandler();

    std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>>
    get_ddhdg_boundary_condition_handler();

    void
    add_boundary_condition(dealii::types::boundary_id   id,
                           Ddhdg::BoundaryConditionType bc_type,
                           Ddhdg::Component             c,
                           const DealIIFunction<dim> &  f);

    void
    add_boundary_condition(dealii::types::boundary_id   id,
                           Ddhdg::BoundaryConditionType bc_type,
                           Ddhdg::Component             c,
                           const std::string &          f);

    [[nodiscard]] bool
    has_dirichlet_boundary_conditions() const;

    [[nodiscard]] bool
    has_neumann_boundary_conditions() const;

  private:
    const std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>> bc_handler;
  };

  template <int dim>
  class Problem
  {
  public:
    Problem(double                         left,
            double                         right,
            HomogeneousPermittivity<dim> & permittivity,
            ElectronMobility<dim> &        n_electron_mobility,
            ElectronMobility<dim> &        p_electron_mobility,
            RecombinationTerm<dim> &       recombination_term,
            DealIIFunction<dim> &          temperature,
            DealIIFunction<dim> &          doping,
            BoundaryConditionHandler<dim> &bc_handler,
            double                         conduction_band_density,
            double                         valence_band_density,
            double                         conduction_band_edge_energy,
            double                         valence_band_edge_energy);

    Problem(const Problem<dim> &problem);

    std::shared_ptr<
      const Ddhdg::Problem<dim, Ddhdg::HomogeneousPermittivity<dim>>>
    get_ddhdg_problem() const;

  private:
    static std::shared_ptr<dealii::Triangulation<dim>>
    generate_triangulation(double left = 0., double right = 1.);

    const std::shared_ptr<
      const Ddhdg::Problem<dim, Ddhdg::HomogeneousPermittivity<dim>>>
      ddhdg_problem;
  };

  template <int dim>
  class NPSolver
  {
  public:
    NPSolver(const Problem<dim> &             problem,
             const Ddhdg::NPSolverParameters &parameters,
             const Ddhdg::Adimensionalizer &  adimensionalizer);

    void
    refine_grid(unsigned int i = 1);

    [[nodiscard]] unsigned int
    n_of_triangulation_levels() const;

    [[nodiscard]] unsigned int
    get_n_dofs(bool for_trace) const;

    [[nodiscard]] unsigned int
    get_n_active_cells() const;

    void
    set_component(Ddhdg::Component   c,
                  const std::string &f,
                  bool               use_projection);

    void
    set_component(Ddhdg::Component    c,
                  DealIIFunction<dim> f,
                  bool                use_projection);

    void
    set_current_solution(const std::string &v_f,
                         const std::string &n_f,
                         const std::string &p_f,
                         bool               use_projection = false);

    void
    set_multithreading(bool multithreading = true);

    [[nodiscard]] bool
    is_enabled(Ddhdg::Component c) const;

    void
    enable_component(Ddhdg::Component c);

    void
    disable_component(Ddhdg::Component c);

    void
    set_enabled_components(bool V_enabled, bool n_enabled, bool p_enabled);

    Ddhdg::NonlinearIterationResults
    run();

    Ddhdg::NonlinearIterationResults
    compute_thermodynamic_equilibrium(bool generate_first_guess);

    Ddhdg::NonlinearIterationResults
    compute_thermodynamic_equilibrium(double absolute_tol,
                                      double relative_tol,
                                      int    max_number_of_iterations,
                                      bool   generate_first_guess);

    [[nodiscard]] double
    estimate_l2_error(DealIIFunction<dim> expected_solution,
                      Ddhdg::Component    c) const;

    [[nodiscard]] double
    estimate_l2_error(DealIIFunction<dim> expected_solution,
                      Ddhdg::Displacement d) const;

    [[nodiscard]] double
    estimate_l2_error(NPSolver<dim> solver, Ddhdg::Component c) const;

    [[nodiscard]] double
    estimate_l2_error(NPSolver<dim> solver, Ddhdg::Displacement d) const;

    [[nodiscard]] double
    estimate_h1_error(DealIIFunction<dim> expected_solution,
                      Ddhdg::Component    c) const;

    [[nodiscard]] double
    estimate_h1_error(DealIIFunction<dim> expected_solution,
                      Ddhdg::Displacement d) const;

    [[nodiscard]] double
    estimate_h1_error(NPSolver<dim> solver, Ddhdg::Component c) const;

    [[nodiscard]] double
    estimate_h1_error(NPSolver<dim> solver, Ddhdg::Displacement d) const;

    [[nodiscard]] double
    estimate_linfty_error(DealIIFunction<dim> expected_solution,
                          Ddhdg::Component    c) const;

    [[nodiscard]] double
    estimate_linfty_error(DealIIFunction<dim> expected_solution,
                          Ddhdg::Displacement d) const;

    [[nodiscard]] double
    estimate_linfty_error(NPSolver<dim> solver, Ddhdg::Component c) const;

    [[nodiscard]] double
    estimate_linfty_error(NPSolver<dim> solver, Ddhdg::Displacement d) const;

    [[nodiscard]] double
    estimate_l2_error_on_trace(const std::string &expected_solution,
                               Ddhdg::Component   c) const;

    [[nodiscard]] double
    estimate_linfty_error_on_trace(const std::string &expected_solution,
                                   Ddhdg::Component   c) const;

    [[nodiscard]] DealIIFunction<dim>
    get_solution(Ddhdg::Component c) const;

    [[nodiscard]] double
    get_solution_on_a_point(dealii::Point<dim> p, Ddhdg::Component c) const;

    void
    output_results(const std::string &solution_filename,
                   bool               save_update = false) const;

    void
    output_results(const std::string &solution_filename,
                   const std::string &trace_filename,
                   bool               save_update = false) const;

    void
    print_convergence_table(const std::string &expected_V_solution,
                            const std::string &expected_n_solution,
                            const std::string &expected_p_solution,
                            unsigned int       n_cycles,
                            unsigned int       initial_refinements = 0);

    void
    print_convergence_table(const std::string &expected_V_solution,
                            const std::string &expected_n_solution,
                            const std::string &expected_p_solution,
                            const std::string &initial_V_function,
                            const std::string &initial_n_function,
                            const std::string &initial_p_function,
                            unsigned int       n_cycles,
                            unsigned int       initial_refinements = 0);

  private:
    const std::shared_ptr<
      Ddhdg::NPSolver<dim, Ddhdg::HomogeneousPermittivity<dim>>>
      ddhdg_solver;
  };

} // namespace pyddhdg
