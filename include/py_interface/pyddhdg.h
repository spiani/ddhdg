#pragma once

#include <deal.II/grid/grid_generator.h>

#include "boundary_conditions.h"
#include "ddhdg.h"

namespace pyddhdg
{
  template <int dim>
  class Permittivity
  {
  public:
    virtual ~Permittivity() = default;

    virtual std::shared_ptr<Ddhdg::Permittivity<dim>>
    generate_ddhdg_permittivity() = 0;
  };

  template <int dim>
  class HomogeneousPermittivity : public Permittivity<dim>
  {
  public:
    explicit HomogeneousPermittivity(double epsilon);

    virtual std::shared_ptr<Ddhdg::Permittivity<dim>>
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
  class PythonFunction
  {
  public:
    explicit PythonFunction(std::string f_exp);

    std::shared_ptr<dealii::Function<dim>>
    get_dealii_function() const;

    [[nodiscard]] std::string
    get_expression() const;

  private:
    std::string                                        f_expr;
    const std::shared_ptr<dealii::FunctionParser<dim>> f;
  };

  template <int dim>
  class Temperature : public PythonFunction<dim>
  {
  public:
    explicit Temperature(const std::string &f_expr);
  };

  template <int dim>
  class Doping : public PythonFunction<dim>
  {
  public:
    explicit Doping(const std::string &f_expr);
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
    LinearRecombinationTerm(const PythonFunction<dim> &zero_term,
                            const PythonFunction<dim> &n_linear_coefficient,
                            const PythonFunction<dim> &p_linear_coefficient);

    std::shared_ptr<Ddhdg::RecombinationTerm<dim>>
    generate_ddhdg_recombination_term() override;

    [[nodiscard]] std::string
    get_constant_term() const;

    [[nodiscard]] std::string
    get_n_linear_coefficient() const;

    [[nodiscard]] std::string
    get_p_linear_coefficient() const;

  private:
    const PythonFunction<dim> zero_term;
    const PythonFunction<dim> n_linear_coefficient;
    const PythonFunction<dim> p_linear_coefficient;
  };

  template <int dim>
  class BoundaryConditionHandler
  {
  public:
    BoundaryConditionHandler();

    std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>>
    get_ddhdg_boundary_condition_handler();

    void
    add_boundary_condition_from_function(dealii::types::boundary_id   id,
                                         Ddhdg::BoundaryConditionType bc_type,
                                         Ddhdg::Component             c,
                                         const PythonFunction<dim> &  f);

    void
    add_boundary_condition_from_string(dealii::types::boundary_id   id,
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
    Problem(Permittivity<dim> &            permittivity,
            ElectronMobility<dim> &        n_electron_mobility,
            RecombinationTerm<dim> &       n_recombination_term,
            ElectronMobility<dim> &        p_electron_mobility,
            RecombinationTerm<dim> &       p_recombination_term,
            Temperature<dim> &             temperature,
            Doping<dim> &                  doping,
            BoundaryConditionHandler<dim> &bc_handler,
            double                         conduction_band_density,
            double                         valence_band_density,
            double                         conduction_band_edge_energy,
            double                         valence_band_edge_energy);

    Problem(const Problem<dim> &problem);

    std::shared_ptr<const Ddhdg::Problem<dim>>
    get_ddhdg_problem() const;

  private:
    static std::shared_ptr<dealii::Triangulation<dim>>
    generate_triangulation();

    const std::shared_ptr<const Ddhdg::Problem<dim>> ddhdg_problem;
  };

  template <int dim>
  class NPSolver
  {
  public:
    NPSolver(const Problem<dim> &             problem,
             const Ddhdg::NPSolverParameters &parameters);

    void
    refine_grid(unsigned int i = 1);

    void
    set_component(Ddhdg::Component   c,
                  const std::string &f,
                  bool               use_projection);

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

    [[nodiscard]] double
    estimate_l2_error(const std::string &expected_solution,
                      Ddhdg::Component   c) const;

    [[nodiscard]] double
    estimate_l2_error(const std::string & expected_solution,
                      Ddhdg::Displacement d) const;

    [[nodiscard]] double
    estimate_h1_error(const std::string &expected_solution,
                      Ddhdg::Component   c) const;

    [[nodiscard]] double
    estimate_h1_error(const std::string & expected_solution,
                      Ddhdg::Displacement d) const;

    [[nodiscard]] double
    estimate_linfty_error(const std::string &expected_solution,
                          Ddhdg::Component   c) const;

    [[nodiscard]] double
    estimate_linfty_error(const std::string & expected_solution,
                          Ddhdg::Displacement d) const;

    [[nodiscard]] double
    estimate_l2_error_on_trace(const std::string &expected_solution,
                               Ddhdg::Component   c) const;

    [[nodiscard]] double
    estimate_linfty_error_on_trace(const std::string &expected_solution,
                                   Ddhdg::Component   c) const;

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
    const std::shared_ptr<Ddhdg::NPSolver<dim>> ddhdg_solver;
  };

} // namespace pyddhdg
