#include <deal.II/grid/grid_generator.h>

#include <ddhdg.h>
#include <gtest/gtest.h>

class DisablingComponentsTest
  : public Ddhdg::NPSolver<2, Ddhdg::HomogeneousProblem<2>>,
    public ::testing::Test
{
public:
  DisablingComponentsTest()
    : Ddhdg::NPSolver<2, Ddhdg::HomogeneousProblem<2>>(
        get_problem(),
        std::make_shared<Ddhdg::NPSolverParameters>(1, 2, 3),
        std::make_shared<Ddhdg::Adimensionalizer>(1,
                                                  Ddhdg::Constants::Q /
                                                    Ddhdg::Constants::KB,
                                                  1 / Ddhdg::Constants::Q),
        false){};

protected:
  constexpr static const unsigned int dim = 2;

  constexpr static const double V_tolerance = 0.2;
  constexpr static const double n_tolerance = 0.1;
  constexpr static const double p_tolerance = 0.1;

  static std::shared_ptr<dealii::FunctionParser<2>>
  get_function(const std::string &s)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> custom_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    custom_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      s,
      Ddhdg::Constants::constants);
    return custom_function;
  }

  static std::shared_ptr<dealii::FunctionParser<2>>
  get_expected_solution(Ddhdg::Component c)
  {
    switch (c)
      {
        case Ddhdg::Component::V:
          return get_function("sin(pi*x)*sin(pi*y)");
        case Ddhdg::Component::n:
          return get_function("1/q * x");
        case Ddhdg::Component::p:
          return get_function("- 1/q * x");
        default:
          AssertThrow(false, Ddhdg::InvalidComponent());
          break;
      }
    return get_function("0");
  }

  static std::shared_ptr<dealii::Triangulation<2>>
  get_triangulation()
  {
    std::shared_ptr<dealii::Triangulation<dim>> triangulation =
      std::make_shared<dealii::Triangulation<dim>>();

    dealii::GridGenerator::hyper_cube(*triangulation, -1, 1, true);
    return triangulation;
  }

  static std::shared_ptr<dealii::FunctionParser<2>>
  get_temperature()
  {
    return get_function("q / kb");
  }

  static std::shared_ptr<dealii::FunctionParser<2>>
  get_doping()
  {
    return get_function("2*(pi^2*sin(pi*x)*sin(pi*y) + x)/q");
  }

  static std::shared_ptr<Ddhdg::RecombinationTerm<2>>
  get_recombination_term()
  {
    std::string recombination_constant_term =
      "(2*pi^2*x*sin(pi*x) - pi*cos(pi*x))*sin(pi*y)";

    std::shared_ptr<Ddhdg::LinearRecombinationTerm<dim>> recombination_term =
      std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>(
        recombination_constant_term, "0", "0");
    return recombination_term;
  }

  static std::shared_ptr<Ddhdg::BoundaryConditionHandler<2>>
  get_boundary_conditions()
  {
    std::set<Ddhdg::Component> components{Ddhdg::Component::V,
                                          Ddhdg::Component::n,
                                          Ddhdg::Component::p};

    std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>> boundary_handler =
      std::make_shared<Ddhdg::BoundaryConditionHandler<dim>>();

    for (unsigned int i = 0; i < 4; i++)
      {
        boundary_handler->add_boundary_condition(i,
                                                 Ddhdg::dirichlet,
                                                 Ddhdg::Component::V,
                                                 get_function("0"));

        boundary_handler->add_boundary_condition(i,
                                                 Ddhdg::dirichlet,
                                                 Ddhdg::Component::n,
                                                 get_function("1/q * x"));

        boundary_handler->add_boundary_condition(i,
                                                 Ddhdg::dirichlet,
                                                 Ddhdg::Component::p,
                                                 get_function("- 1/q * x"));
      }

    return boundary_handler;
  }

  static std::shared_ptr<Ddhdg::HomogeneousProblem<2>>
  get_problem()
  {
    std::shared_ptr<Ddhdg::HomogeneousProblem<dim>> problem =
      std::make_shared<Ddhdg::HomogeneousProblem<dim>>(
        get_triangulation(),
        std::make_shared<const Ddhdg::HomogeneousPermittivity<dim>>(1.),
        std::make_shared<const Ddhdg::HomogeneousElectronMobility<dim>>(1.),
        std::make_shared<const Ddhdg::HomogeneousElectronMobility<dim>>(1.),
        get_recombination_term(),
        get_temperature(),
        get_doping(),
        get_boundary_conditions(),
        1.,
        1.,
        0.,
        0.);
    return problem;
  }
};



TEST_F(DisablingComponentsTest, RunWithAll) // NOLINT
{
  const auto V_expected_solution = get_expected_solution(Ddhdg::Component::V);
  const auto n_expected_solution = get_expected_solution(Ddhdg::Component::n);
  const auto p_expected_solution = get_expected_solution(Ddhdg::Component::p);

  this->set_multithreading(false);
  this->refine_grid(3, false);

  const Ddhdg::NonlinearIterationResults status = this->run();

  ASSERT_TRUE(status.converged);

  const double V_l2_error =
    this->estimate_l2_error(V_expected_solution, Ddhdg::Component::V);
  const double n_l2_error =
    this->estimate_l2_error(n_expected_solution, Ddhdg::Component::n);
  const double p_l2_error =
    this->estimate_l2_error(p_expected_solution, Ddhdg::Component::p);

  EXPECT_LT(V_l2_error, V_tolerance);
  EXPECT_LT(n_l2_error, n_tolerance);
  EXPECT_LT(p_l2_error, p_tolerance);
}



TEST_F(DisablingComponentsTest, RunForV) // NOLINT
{
  const auto V_expected_solution = get_expected_solution(Ddhdg::Component::V);
  const auto n_expected_solution = get_expected_solution(Ddhdg::Component::n);
  const auto p_expected_solution = get_expected_solution(Ddhdg::Component::p);

  this->set_multithreading(false);
  this->refine_grid(3, false);
  this->disable_component(Ddhdg::Component::n);
  this->disable_component(Ddhdg::Component::p);

  this->set_component(Ddhdg::Component::n, n_expected_solution, true);
  this->set_component(Ddhdg::Component::p, p_expected_solution, true);

  const Ddhdg::NonlinearIterationResults status = this->run();

  ASSERT_TRUE(status.converged);

  const double V_l2_error =
    this->estimate_l2_error(V_expected_solution, Ddhdg::Component::V);

  EXPECT_LT(V_l2_error, V_tolerance);
}



TEST_F(DisablingComponentsTest, RunForN) // NOLINT
{
  const auto V_expected_solution = get_expected_solution(Ddhdg::Component::V);
  const auto n_expected_solution = get_expected_solution(Ddhdg::Component::n);
  const auto p_expected_solution = get_expected_solution(Ddhdg::Component::p);

  this->set_multithreading(false);
  this->refine_grid(3, false);
  this->disable_component(Ddhdg::Component::V);
  this->disable_component(Ddhdg::Component::p);

  this->set_component(Ddhdg::Component::V, V_expected_solution, true);
  this->set_component(Ddhdg::Component::p, p_expected_solution, true);

  const Ddhdg::NonlinearIterationResults status = this->run();

  ASSERT_TRUE(status.converged);

  const double n_l2_error =
    this->estimate_l2_error(n_expected_solution, Ddhdg::Component::n);

  EXPECT_LT(n_l2_error, n_tolerance);
}



TEST_F(DisablingComponentsTest, RunForP) // NOLINT
{
  const auto V_expected_solution = get_expected_solution(Ddhdg::Component::V);
  const auto n_expected_solution = get_expected_solution(Ddhdg::Component::n);
  const auto p_expected_solution = get_expected_solution(Ddhdg::Component::p);

  this->set_multithreading(false);
  this->refine_grid(3, false);
  this->disable_component(Ddhdg::Component::V);
  this->disable_component(Ddhdg::Component::n);

  this->set_component(Ddhdg::Component::V, V_expected_solution, true);
  this->set_component(Ddhdg::Component::n, n_expected_solution, true);

  const Ddhdg::NonlinearIterationResults status = this->run();

  ASSERT_TRUE(status.converged);

  const double p_l2_error =
    this->estimate_l2_error(p_expected_solution, Ddhdg::Component::p);

  EXPECT_LT(p_l2_error, p_tolerance);
}



TEST_F(DisablingComponentsTest, RunForVAndP) // NOLINT
{
  const auto V_expected_solution = get_expected_solution(Ddhdg::Component::V);
  const auto n_expected_solution = get_expected_solution(Ddhdg::Component::n);
  const auto p_expected_solution = get_expected_solution(Ddhdg::Component::p);

  this->set_multithreading(false);
  this->refine_grid(3, false);
  this->disable_component(Ddhdg::Component::n);

  this->set_component(Ddhdg::Component::n, n_expected_solution, true);

  const Ddhdg::NonlinearIterationResults status = this->run();

  ASSERT_TRUE(status.converged);

  const double V_l2_error =
    this->estimate_l2_error(V_expected_solution, Ddhdg::Component::V);
  const double p_l2_error =
    this->estimate_l2_error(p_expected_solution, Ddhdg::Component::p);

  EXPECT_LT(V_l2_error, V_tolerance);
  EXPECT_LT(p_l2_error, p_tolerance);
}
