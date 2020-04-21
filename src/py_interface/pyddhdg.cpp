#include "py_interface/pyddhdg.h"

#include <utility>

#include "function_tools.h"

namespace pyddhdg
{
  template <int dim>
  HomogeneousPermittivity<dim>::HomogeneousPermittivity(const double epsilon)
    : epsilon(epsilon)
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::Permittivity<dim>>
  HomogeneousPermittivity<dim>::generate_ddhdg_permittivity()
  {
    return std::make_shared<Ddhdg::HomogeneousPermittivity<dim>>(this->epsilon);
  }



  template <int dim>
  HomogeneousElectronMobility<dim>::HomogeneousElectronMobility(const double mu)
    : mu(mu)
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::ElectronMobility<dim>>
  HomogeneousElectronMobility<dim>::generate_ddhdg_electron_mobility()
  {
    return std::make_shared<Ddhdg::HomogeneousElectronMobility<dim>>(this->mu);
  }



  template <int dim>
  std::shared_ptr<dealii::FunctionParser<dim>>
  PythonFunction<dim>::get_function_from_string(const std::string &f_expr)
  {
    auto f = std::make_shared<dealii::FunctionParser<dim>>();
    f->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                  f_expr,
                  Ddhdg::Constants::constants);
    return f;
  }



  template <int dim>
  PythonFunction<dim>::PythonFunction(std::string f_expr)
    : f_expr(f_expr)
    , f(get_function_from_string(f_expr))
  {}



  template <int dim>
  PythonFunction<dim>::PythonFunction(const double f_const)
    : f_expr(std::to_string(f_const))
    , f(std::make_shared<dealii::Functions::ConstantFunction<dim>>(f_const))
  {}



  template <int dim>
  PythonFunction<dim>::PythonFunction(std::string f_expr,
                                      std::shared_ptr<dealii::Function<dim>> f)
    : f_expr(f_expr)
    , f(f)
  {}



  template <int dim>
  std::shared_ptr<dealii::Function<dim>>
  PythonFunction<dim>::get_dealii_function() const
  {
    return this->f;
  }



  template <int dim>
  std::string
  PythonFunction<dim>::get_expression() const
  {
    return this->f_expr;
  }



  template <int dim>
  PiecewiseFunction<dim>::PiecewiseFunction(
    const PythonFunction<dim> &condition,
    const PythonFunction<dim> &f1,
    const PythonFunction<dim> &f2)
    : PythonFunction<dim>("(" + condition.get_expression() + ") ? " +
                            f1.get_expression() + " : " + f2.get_expression(),
                          std::make_shared<Ddhdg::PiecewiseFunction<dim>>(
                            condition.get_dealii_function(),
                            f1.get_dealii_function(),
                            f2.get_dealii_function()))
  {}



  template <int dim>
  PiecewiseFunction<dim>::PiecewiseFunction(const std::string &condition,
                                            const std::string &f1,
                                            const std::string &f2)
    : PiecewiseFunction(PythonFunction<dim>(condition),
                        PythonFunction<dim>(f1),
                        PythonFunction<dim>(f2))
  {}



  template <int dim>
  PiecewiseFunction<dim>::PiecewiseFunction(const std::string &condition,
                                            const std::string &f1,
                                            double             f2)
    : PiecewiseFunction(PythonFunction<dim>(condition),
                        PythonFunction<dim>(f1),
                        PythonFunction<dim>(f2))
  {}



  template <int dim>
  PiecewiseFunction<dim>::PiecewiseFunction(const std::string &condition,
                                            double             f1,
                                            const std::string &f2)
    : PiecewiseFunction(PythonFunction<dim>(condition),
                        PythonFunction<dim>(f1),
                        PythonFunction<dim>(f2))
  {}



  template <int dim>
  PiecewiseFunction<dim>::PiecewiseFunction(const std::string &condition,
                                            double             f1,
                                            double             f2)
    : PiecewiseFunction(PythonFunction<dim>(condition),
                        PythonFunction<dim>(f1),
                        PythonFunction<dim>(f2))
  {}



  template <int dim>
  LinearRecombinationTerm<dim>::LinearRecombinationTerm(
    const PythonFunction<dim> &zero_term,
    const PythonFunction<dim> &n_linear_coefficient,
    const PythonFunction<dim> &p_linear_coefficient)
    : zero_term(zero_term.get_expression())
    , n_linear_coefficient(n_linear_coefficient.get_expression())
    , p_linear_coefficient(p_linear_coefficient.get_expression())
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::RecombinationTerm<dim>>
  LinearRecombinationTerm<dim>::generate_ddhdg_recombination_term()
  {
    return std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>(
      this->zero_term.get_expression(),
      this->n_linear_coefficient.get_expression(),
      this->p_linear_coefficient.get_expression());
  }



  template <int dim>
  std::string
  LinearRecombinationTerm<dim>::get_constant_term() const
  {
    return this->zero_term.get_expression();
  }



  template <int dim>
  std::string
  LinearRecombinationTerm<dim>::get_n_linear_coefficient() const
  {
    return this->p_linear_coefficient.get_expression();
  }



  template <int dim>
  std::string
  LinearRecombinationTerm<dim>::get_p_linear_coefficient() const
  {
    return this->n_linear_coefficient.get_expression();
  }



  template <int dim>
  BoundaryConditionHandler<dim>::BoundaryConditionHandler()
    : bc_handler(std::make_shared<Ddhdg::BoundaryConditionHandler<dim>>())
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>>
  BoundaryConditionHandler<dim>::get_ddhdg_boundary_condition_handler()
  {
    return this->bc_handler;
  }



  template <int dim>
  void
  BoundaryConditionHandler<dim>::add_boundary_condition_from_function(
    const dealii::types::boundary_id   id,
    const Ddhdg::BoundaryConditionType bc_type,
    const Ddhdg::Component             c,
    const PythonFunction<dim> &        f)
  {
    this->bc_handler->add_boundary_condition(id,
                                             bc_type,
                                             c,
                                             f.get_dealii_function());
  }



  template <int dim>
  void
  BoundaryConditionHandler<dim>::add_boundary_condition_from_string(
    const dealii::types::boundary_id   id,
    const Ddhdg::BoundaryConditionType bc_type,
    const Ddhdg::Component             c,
    const std::string &                f)
  {
    this->add_boundary_condition_from_function(id,
                                               bc_type,
                                               c,
                                               PythonFunction<dim>(f));
  }



  template <int dim>
  bool
  BoundaryConditionHandler<dim>::has_dirichlet_boundary_conditions() const
  {
    return this->bc_handler->has_dirichlet_boundary_conditions();
  }



  template <int dim>
  bool
  BoundaryConditionHandler<dim>::has_neumann_boundary_conditions() const
  {
    return this->bc_handler->has_neumann_boundary_conditions();
  }



  template <int dim>
  Problem<dim>::Problem(const double                   left,
                        const double                   right,
                        Permittivity<dim> &            permittivity,
                        ElectronMobility<dim> &        n_electron_mobility,
                        RecombinationTerm<dim> &       n_recombination_term,
                        ElectronMobility<dim> &        p_electron_mobility,
                        RecombinationTerm<dim> &       p_recombination_term,
                        PythonFunction<dim> &          temperature,
                        PythonFunction<dim> &          doping,
                        BoundaryConditionHandler<dim> &bc_handler,
                        const double                   conduction_band_density,
                        const double                   valence_band_density,
                        const double conduction_band_edge_energy,
                        const double valence_band_edge_energy)
    : ddhdg_problem(std::make_shared<Ddhdg::Problem<dim>>(
        generate_triangulation(left, right),
        permittivity.generate_ddhdg_permittivity(),
        n_electron_mobility.generate_ddhdg_electron_mobility(),
        n_recombination_term.generate_ddhdg_recombination_term(),
        p_electron_mobility.generate_ddhdg_electron_mobility(),
        p_recombination_term.generate_ddhdg_recombination_term(),
        temperature.get_dealii_function(),
        doping.get_dealii_function(),
        bc_handler.get_ddhdg_boundary_condition_handler(),
        conduction_band_density,
        valence_band_density,
        conduction_band_edge_energy,
        valence_band_edge_energy))
  {}



  template <int dim>
  Problem<dim>::Problem(const Problem<dim> &problem)
    : ddhdg_problem(problem.ddhdg_problem)
  {}



  template <int dim>
  std::shared_ptr<const Ddhdg::Problem<dim>>
  Problem<dim>::get_ddhdg_problem() const
  {
    return this->ddhdg_problem;
  }



  template <int dim>
  std::shared_ptr<dealii::Triangulation<dim>>
  Problem<dim>::generate_triangulation(const double left, const double right)
  {
    std::shared_ptr<dealii::Triangulation<dim>> triangulation =
      std::make_shared<dealii::Triangulation<dim>>();

    dealii::GridGenerator::hyper_cube(*triangulation, left, right, true);

    return triangulation;
  }



  template <int dim>
  NPSolver<dim>::NPSolver(const Problem<dim> &             problem,
                          const Ddhdg::NPSolverParameters &parameters,
                          const Ddhdg::Adimensionalizer &  adimensionalizer)
    : ddhdg_solver(std::make_shared<Ddhdg::NPSolver<dim>>(
        problem.get_ddhdg_problem(),
        std::make_shared<const Ddhdg::NPSolverParameters>(parameters),
        std::make_shared<const Ddhdg::Adimensionalizer>(adimensionalizer)))
  {}



  template <int dim>
  void
  NPSolver<dim>::refine_grid(const unsigned int i)
  {
    this->ddhdg_solver->refine_grid(i);
  }



  template <int dim>
  void
  NPSolver<dim>::set_component(const Ddhdg::Component c,
                               const std::string &    f,
                               const bool             use_projection)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> c_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    c_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      f,
      Ddhdg::Constants::constants);
    this->ddhdg_solver->set_component(c, c_function, use_projection);
  }



  template <int dim>
  void
  NPSolver<dim>::set_current_solution(const std::string &v_f,
                                      const std::string &n_f,
                                      const std::string &p_f,
                                      const bool         use_projection)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> v_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> n_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> p_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    v_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      v_f,
      Ddhdg::Constants::constants);
    n_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      n_f,
      Ddhdg::Constants::constants);
    p_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      p_f,
      Ddhdg::Constants::constants);
    this->ddhdg_solver->set_current_solution(v_function,
                                             n_function,
                                             p_function,
                                             use_projection);
  }



  template <int dim>
  void
  NPSolver<dim>::set_multithreading(const bool multithreading)
  {
    this->ddhdg_solver->set_multithreading(multithreading);
  }



  template <int dim>
  bool
  NPSolver<dim>::is_enabled(Ddhdg::Component c) const
  {
    return this->ddhdg_solver->is_enabled(c);
  }



  template <int dim>
  void
  NPSolver<dim>::enable_component(Ddhdg::Component c)
  {
    this->ddhdg_solver->enable_component(c);
  }



  template <int dim>
  void
  NPSolver<dim>::disable_component(Ddhdg::Component c)
  {
    this->ddhdg_solver->disable_component(c);
  }



  template <int dim>
  void
  NPSolver<dim>::set_enabled_components(const bool V_enabled,
                                        const bool n_enabled,
                                        const bool p_enabled)
  {
    this->ddhdg_solver->set_enabled_components(V_enabled, n_enabled, p_enabled);
  }



  template <int dim>
  Ddhdg::NonlinearIterationResults
  NPSolver<dim>::run()
  {
    return this->ddhdg_solver->run();
  }



  template <int dim>
  Ddhdg::NonlinearIterationResults
  NPSolver<dim>::compute_thermodynamic_equilibrium()
  {
    return this->ddhdg_solver->compute_thermodynamic_equilibrium();
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(const std::string &    expected_solution,
                                   const Ddhdg::Component c) const
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_solution,
      Ddhdg::Constants::constants);
    return this->ddhdg_solver->estimate_l2_error(expected_solution_f, c);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(const std::string &       expected_solution,
                                   const Ddhdg::Displacement d) const
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>(dim);
    expected_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_solution,
      Ddhdg::Constants::constants);
    return this->ddhdg_solver->estimate_l2_error(expected_solution_f, d);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(const std::string &    expected_solution,
                                   const Ddhdg::Component c) const
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_solution,
      Ddhdg::Constants::constants);
    return this->ddhdg_solver->estimate_h1_error(expected_solution_f, c);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(const std::string &       expected_solution,
                                   const Ddhdg::Displacement d) const
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>(dim);
    expected_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_solution,
      Ddhdg::Constants::constants);
    return this->ddhdg_solver->estimate_h1_error(expected_solution_f, d);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(const std::string &    expected_solution,
                                       const Ddhdg::Component c) const
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_solution,
      Ddhdg::Constants::constants);
    return this->ddhdg_solver->estimate_linfty_error(expected_solution_f, c);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(const std::string &expected_solution,
                                       const Ddhdg::Displacement d) const
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>(dim);
    expected_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_solution,
      Ddhdg::Constants::constants);
    return this->ddhdg_solver->estimate_linfty_error(expected_solution_f, d);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error_on_trace(
    const std::string &    expected_solution,
    const Ddhdg::Component c) const
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_solution,
      Ddhdg::Constants::constants);
    return this->ddhdg_solver->estimate_l2_error_on_trace(expected_solution_f,
                                                          c);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error_on_trace(
    const std::string &    expected_solution,
    const Ddhdg::Component c) const
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_solution,
      Ddhdg::Constants::constants);
    return this->ddhdg_solver->estimate_linfty_error_on_trace(
      expected_solution_f, c);
  }


  template <int dim>
  void
  NPSolver<dim>::output_results(const std::string &solution_filename,
                                const bool         save_update) const
  {
    this->ddhdg_solver->output_results(solution_filename, save_update);
  }



  template <int dim>
  void
  NPSolver<dim>::output_results(const std::string &solution_filename,
                                const std::string &trace_filename,
                                const bool         save_update) const
  {
    this->ddhdg_solver->output_results(solution_filename,
                                       trace_filename,
                                       save_update);
  }



  template <int dim>
  void
  NPSolver<dim>::print_convergence_table(const std::string &expected_V_solution,
                                         const std::string &expected_n_solution,
                                         const std::string &expected_p_solution,
                                         const unsigned int n_cycles,
                                         const unsigned int initial_refinements)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_V_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> expected_n_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> expected_p_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_V_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_V_solution,
      Ddhdg::Constants::constants);
    expected_n_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_n_solution,
      Ddhdg::Constants::constants);
    expected_p_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_p_solution,
      Ddhdg::Constants::constants);
    this->ddhdg_solver->print_convergence_table(
      std::make_shared<Ddhdg::ConvergenceTable>(dim),
      expected_V_solution_f,
      expected_n_solution_f,
      expected_p_solution_f,
      n_cycles,
      initial_refinements);
  }



  template <int dim>
  void
  NPSolver<dim>::print_convergence_table(const std::string &expected_V_solution,
                                         const std::string &expected_n_solution,
                                         const std::string &expected_p_solution,
                                         const std::string &initial_V_function,
                                         const std::string &initial_n_function,
                                         const std::string &initial_p_function,
                                         const unsigned int n_cycles,
                                         const unsigned int initial_refinements)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_V_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> expected_n_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> expected_p_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> initial_V_function_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> initial_n_function_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> initial_p_function_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_V_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_V_solution,
      Ddhdg::Constants::constants);
    expected_n_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_n_solution,
      Ddhdg::Constants::constants);
    expected_p_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_p_solution,
      Ddhdg::Constants::constants);
    initial_V_function_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      initial_V_function,
      Ddhdg::Constants::constants);
    initial_n_function_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      initial_n_function,
      Ddhdg::Constants::constants);
    initial_p_function_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      initial_p_function,
      Ddhdg::Constants::constants);
    this->ddhdg_solver->print_convergence_table(
      std::make_shared<Ddhdg::ConvergenceTable>(dim),
      expected_V_solution_f,
      expected_n_solution_f,
      expected_p_solution_f,
      initial_V_function_f,
      initial_n_function_f,
      initial_p_function_f,
      n_cycles,
      initial_refinements);
  }



  template class HomogeneousPermittivity<1>;
  template class HomogeneousPermittivity<2>;
  template class HomogeneousPermittivity<3>;

  template class HomogeneousElectronMobility<1>;
  template class HomogeneousElectronMobility<2>;
  template class HomogeneousElectronMobility<3>;

  template class PythonFunction<1>;
  template class PythonFunction<2>;
  template class PythonFunction<3>;

  template class PiecewiseFunction<1>;
  template class PiecewiseFunction<2>;
  template class PiecewiseFunction<3>;

  template class LinearRecombinationTerm<1>;
  template class LinearRecombinationTerm<2>;
  template class LinearRecombinationTerm<3>;

  template class BoundaryConditionHandler<1>;
  template class BoundaryConditionHandler<2>;
  template class BoundaryConditionHandler<3>;

  template class Problem<1>;
  template class Problem<2>;
  template class Problem<3>;

  template class NPSolver<1>;
  template class NPSolver<2>;
  template class NPSolver<3>;
} // namespace pyddhdg
