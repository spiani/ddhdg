#include "py_interface/pyddhdg.h"

#include <utility>

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
  PythonFunction<dim>::PythonFunction(std::string f_expr)
    : f_expr(std::move(f_expr))
    , f(std::make_shared<dealii::FunctionParser<dim>>(1))
  {
    f->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                  this->f_expr,
                  Ddhdg::Constants::constants);
  }



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
  Temperature<dim>::Temperature(const std::string &f_expr)
    : PythonFunction<dim>(f_expr)
  {}



  template <int dim>
  Doping<dim>::Doping(const std::string &f_expr)
    : PythonFunction<dim>(f_expr)
  {}



  template <int dim>
  LinearRecombinationTerm<dim>::LinearRecombinationTerm(
    const PythonFunction<dim> &zero_term,
    const PythonFunction<dim> &first_term)
    : zero_term(zero_term.get_expression())
    , first_term(first_term.get_expression())
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::RecombinationTerm<dim>>
  LinearRecombinationTerm<dim>::generate_ddhdg_recombination_term()
  {
    return std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>(
      this->zero_term.get_expression(), this->first_term.get_expression());
  }



  template <int dim>
  std::string
  LinearRecombinationTerm<dim>::get_constant_term() const
  {
    return this->zero_term.get_expression();
  }


  template <int dim>
  std::string
  LinearRecombinationTerm<dim>::get_linear_coefficient() const
  {
    return this->first_term.get_expression();
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
  Problem<dim>::Problem(Permittivity<dim> &            permittivity,
                        ElectronMobility<dim> &        electron_mobility,
                        RecombinationTerm<dim> &       recombination_term,
                        Temperature<dim> &             temperature,
                        Doping<dim> &                  doping,
                        BoundaryConditionHandler<dim> &bc_handler)
    : ddhdg_problem(std::make_shared<Ddhdg::Problem<dim>>(
        generate_triangulation(),
        permittivity.generate_ddhdg_permittivity(),
        electron_mobility.generate_ddhdg_electron_mobility(),
        recombination_term.generate_ddhdg_recombination_term(),
        temperature.get_dealii_function(),
        doping.get_dealii_function(),
        bc_handler.get_ddhdg_boundary_condition_handler()))
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
  Problem<dim>::generate_triangulation()
  {
    std::shared_ptr<dealii::Triangulation<dim>> triangulation =
      std::make_shared<dealii::Triangulation<dim>>();

    dealii::GridGenerator::hyper_cube(*triangulation, 0., 1., true);

    return triangulation;
  }



  template <int dim>
  Solver<dim>::Solver(const Problem<dim> &           problem,
                      const Ddhdg::SolverParameters &parameters)
    : ddhdg_solver(std::make_shared<Ddhdg::Solver<dim>>(
        problem.get_ddhdg_problem(),
        std::make_shared<const Ddhdg::SolverParameters>(parameters)))
  {}



  template <int dim>
  void
  Solver<dim>::refine_grid(const unsigned int i)
  {
    this->ddhdg_solver->refine_grid(i);
  }



  template <int dim>
  void
  Solver<dim>::set_component(const Ddhdg::Component c, const std::string &f)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> c_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    c_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      f,
      Ddhdg::Constants::constants);
    this->ddhdg_solver->set_component(c, c_function);
  }



  template <int dim>
  void
  Solver<dim>::set_current_solution(const std::string &v_f,
                                    const std::string &n_f,
                                    const bool         use_projection)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> v_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> n_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    v_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      v_f,
      Ddhdg::Constants::constants);
    n_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      n_f,
      Ddhdg::Constants::constants);
    this->ddhdg_solver->set_current_solution(v_function,
                                             n_function,
                                             use_projection);
  }



  template <int dim>
  void
  Solver<dim>::set_multithreading(const bool multithreading)
  {
    this->ddhdg_solver->set_multithreading(multithreading);
  }



  template <int dim>
  Ddhdg::NonlinearIteratorStatus
  Solver<dim>::run()
  {
    return this->ddhdg_solver->run();
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

  template class Temperature<1>;
  template class Temperature<2>;
  template class Temperature<3>;

  template class Doping<1>;
  template class Doping<2>;
  template class Doping<3>;

  template class LinearRecombinationTerm<1>;
  template class LinearRecombinationTerm<2>;
  template class LinearRecombinationTerm<3>;

  template class BoundaryConditionHandler<1>;
  template class BoundaryConditionHandler<2>;
  template class BoundaryConditionHandler<3>;

  template class Problem<1>;
  template class Problem<2>;
  template class Problem<3>;

  template class Solver<1>;
  template class Solver<2>;
  template class Solver<3>;
} // namespace pyddhdg
