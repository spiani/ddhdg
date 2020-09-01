#include <deal.II/grid/grid_generator.h>

#include <ddhdg.h>
#include <function_tools.h>
#include <gtest/gtest.h>

template <typename D>
class NeumannBCLinearTest
  : public Ddhdg::NPSolver<D::value, Ddhdg::HomogeneousProblem<D::value>>,
    public ::testing::Test
{
public:
  NeumannBCLinearTest()
    : Ddhdg::NPSolver<D::value, Ddhdg::HomogeneousProblem<D::value>>(
        get_problem())
  {
    this->log_standard_level = Ddhdg::Logging::severity_level::debug;
  };

protected:
  static std::shared_ptr<dealii::FunctionParser<D::value>>
  get_zero_function()
  {
    const unsigned int dim = D::value;

    std::shared_ptr<dealii::FunctionParser<dim>> zero_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    zero_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      "0",
      Ddhdg::Constants::constants);
    return zero_function;
  }

  static std::shared_ptr<dealii::FunctionParser<D::value>>
  get_expected_solution()
  {
    const unsigned int dim = D::value;

    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      "x",
      Ddhdg::Constants::constants);
    return expected_solution;
  }

  static std::shared_ptr<dealii::Triangulation<D::value>>
  get_triangulation()
  {
    const unsigned int dim = D::value;

    std::shared_ptr<dealii::Triangulation<dim>> triangulation =
      std::make_shared<dealii::Triangulation<dim>>();

    dealii::GridGenerator::hyper_cube(*triangulation, -1, 1, true);
    return triangulation;
  }

  static std::shared_ptr<dealii::FunctionParser<D::value>>
  get_temperature()
  {
    const unsigned int dim = D::value;

    std::shared_ptr<dealii::FunctionParser<dim>> temperature =
      std::make_shared<dealii::FunctionParser<dim>>();
    temperature->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      "q / kb",
      Ddhdg::Constants::constants);
    return temperature;
  }

  static std::shared_ptr<dealii::FunctionParser<D::value>>
  get_doping()
  {
    const unsigned int dim = D::value;

    std::shared_ptr<dealii::FunctionParser<dim>> doping =
      std::make_shared<dealii::FunctionParser<dim>>();
    doping->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                       "0",
                       Ddhdg::Constants::constants);
    return doping;
  }

  static std::shared_ptr<Ddhdg::BoundaryConditionHandler<D::value>>
  get_boundary_conditions()
  {
    const unsigned int dim = D::value;

    std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>> boundary_handler =
      std::make_shared<Ddhdg::BoundaryConditionHandler<dim>>();

    std::shared_ptr<dealii::FunctionParser<dim>> minus_one_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    minus_one_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      "-1.",
      Ddhdg::Constants::constants);

    // Boundary conditions for n and p: 0 everywhere (Dirichlet)
    for (unsigned int i = 0; i < 6; i++)
      {
        boundary_handler->add_boundary_condition(i,
                                                 Ddhdg::dirichlet,
                                                 Ddhdg::n,
                                                 get_zero_function());
        boundary_handler->add_boundary_condition(i,
                                                 Ddhdg::dirichlet,
                                                 Ddhdg::p,
                                                 get_zero_function());
      }

    // Boundary conditions for V
    boundary_handler->add_boundary_condition(0,
                                             Ddhdg::dirichlet,
                                             Ddhdg::V,
                                             minus_one_function);
    boundary_handler->add_boundary_condition(1,
                                             Ddhdg::neumann,
                                             Ddhdg::V,
                                             minus_one_function);
    for (unsigned int i = 2; i < 6; i++)
      {
        boundary_handler->add_boundary_condition(i,
                                                 Ddhdg::neumann,
                                                 Ddhdg::V,
                                                 get_zero_function());
      }


    return boundary_handler;
  }

  static std::shared_ptr<Ddhdg::HomogeneousProblem<D::value>>
  get_problem()
  {
    const unsigned int                              dim = D::value;
    std::shared_ptr<Ddhdg::HomogeneousProblem<dim>> problem =
      std::make_shared<Ddhdg::HomogeneousProblem<dim>>(
        get_triangulation(),
        std::make_shared<const Ddhdg::HomogeneousPermittivity<dim>>(1.),
        std::make_shared<const Ddhdg::HomogeneousElectronMobility<dim>>(1.),
        std::make_shared<const Ddhdg::HomogeneousElectronMobility<dim>>(1.),
        std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>("0", "0", "0"),
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


using dimensions = ::testing::Types<std::integral_constant<unsigned int, 1>,
                                    std::integral_constant<unsigned int, 2>,
                                    std::integral_constant<unsigned int, 3>>;


TYPED_TEST_SUITE(NeumannBCLinearTest, dimensions, );


TYPED_TEST(NeumannBCLinearTest, NeumannBCLinearTest) // NOLINT
{
  const unsigned int dim = TypeParam::value;

  const auto zero_function     = TestFixture::get_zero_function();
  const auto expected_solution = TestFixture::get_expected_solution();

  this->set_multithreading(false);
  this->refine_grid(3 - dim, false);
  this->set_component(Ddhdg::Component::n, zero_function, false);
  this->set_component(Ddhdg::Component::p, zero_function, false);

  const Ddhdg::NonlinearIterationResults status = this->run();

  const unsigned int number_of_iterations = status.iterations;

  EXPECT_LE(number_of_iterations, 3);

  const double V_l2_error =
    this->estimate_l2_error(expected_solution, Ddhdg::Component::V);
  const double n_l2_error =
    this->estimate_l2_error(zero_function, Ddhdg::Component::n);
  const double p_l2_error =
    this->estimate_l2_error(zero_function, Ddhdg::Component::p);

  EXPECT_LT(V_l2_error, 1e-10);
  EXPECT_LT(n_l2_error, 1e-10);
  EXPECT_LT(p_l2_error, 1e-10);
}



class NeumannBCTrigonometricTest
  : public Ddhdg::NPSolver<2, Ddhdg::HomogeneousProblem<2>>,
    public ::testing::Test
{
public:
  NeumannBCTrigonometricTest()
    : Ddhdg::NPSolver<2, Ddhdg::HomogeneousProblem<2>>(get_problem())
  {
    this->log_standard_level = Ddhdg::Logging::severity_level::debug;
  };

protected:
  static std::shared_ptr<dealii::FunctionParser<2>>
  get_function(const std::string &s)
  {
    const unsigned int dim = 2;

    std::shared_ptr<dealii::FunctionParser<dim>> custom_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    custom_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      s,
      Ddhdg::Constants::constants);
    return custom_function;
  }

  static std::shared_ptr<dealii::FunctionParser<2>>
  get_V_expected_solution()
  {
    const unsigned int dim = 2;

    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      "sin(pi * x) * sin(pi * y)",
      Ddhdg::Constants::constants);
    return expected_solution;
  }

  static std::shared_ptr<dealii::FunctionParser<2>>
  get_n_expected_solution()
  {
    const unsigned int dim = 2;

    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      "cos(pi * x) * cos(pi * y)",
      Ddhdg::Constants::constants);
    return expected_solution;
  }

  static std::shared_ptr<dealii::Triangulation<2>>
  get_triangulation()
  {
    const unsigned int dim = 2;

    std::shared_ptr<dealii::Triangulation<dim>> triangulation =
      std::make_shared<dealii::Triangulation<dim>>();

    dealii::GridGenerator::hyper_cube(*triangulation, -1, 1, true);
    return triangulation;
  }

  static std::shared_ptr<dealii::FunctionParser<2>>
  get_temperature()
  {
    const unsigned int dim = 2;

    std::shared_ptr<dealii::FunctionParser<dim>> temperature =
      std::make_shared<dealii::FunctionParser<dim>>();
    temperature->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      "q / kb",
      Ddhdg::Constants::constants);
    return temperature;
  }

  static std::shared_ptr<dealii::FunctionParser<2>>
  get_doping()
  {
    const unsigned int dim = 2;

    std::shared_ptr<dealii::FunctionParser<dim>> doping =
      std::make_shared<dealii::FunctionParser<dim>>();
    doping->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                       "(2*pi^2*sin(pi*x)*sin(pi*y) + q*cos(pi*x)*cos(pi*y))/q",
                       Ddhdg::Constants::constants);
    return doping;
  }

  static std::shared_ptr<Ddhdg::RecombinationTerm<2>>
  get_recombination_term()
  {
    const unsigned int dim = 2;

    std::shared_ptr<Ddhdg::LinearRecombinationTerm<dim>> recombination_term =
      std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>(
        "(4*pi^2*cos(pi*x)*cos(pi*y)*sin(pi*x)*sin(pi*y) - 2*pi^2*cos(pi*x)*cos(pi*y))*q",
        "0",
        "0");
    return recombination_term;
  }

  static std::shared_ptr<Ddhdg::BoundaryConditionHandler<2>>
  get_boundary_conditions()
  {
    const unsigned int dim = 2;

    std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>> boundary_handler =
      std::make_shared<Ddhdg::BoundaryConditionHandler<dim>>();

    boundary_handler->add_boundary_condition(0,
                                             Ddhdg::dirichlet,
                                             Ddhdg::V,
                                             get_function("0"));
    boundary_handler->add_boundary_condition(1,
                                             Ddhdg::dirichlet,
                                             Ddhdg::V,
                                             get_function("0"));
    boundary_handler->add_boundary_condition(2,
                                             Ddhdg::neumann,
                                             Ddhdg::V,
                                             get_function("-pi*sin(pi*x)"));
    boundary_handler->add_boundary_condition(3,
                                             Ddhdg::neumann,
                                             Ddhdg::V,
                                             get_function("pi*sin(pi*x)"));

    boundary_handler->add_boundary_condition(
      0, Ddhdg::neumann, Ddhdg::n, get_function("pi*cos(pi*y)*sin(pi*y)*q"));
    boundary_handler->add_boundary_condition(
      1, Ddhdg::neumann, Ddhdg::n, get_function("-pi*cos(pi*y)*sin(pi*y)*q"));
    boundary_handler->add_boundary_condition(2,
                                             Ddhdg::dirichlet,
                                             Ddhdg::n,
                                             get_function("-cos(pi * x)"));
    boundary_handler->add_boundary_condition(3,
                                             Ddhdg::dirichlet,
                                             Ddhdg::n,
                                             get_function("-cos(pi * x)"));

    for (unsigned int i = 0; i < 4; i++)
      {
        boundary_handler->add_boundary_condition(i,
                                                 Ddhdg::dirichlet,
                                                 Ddhdg::p,
                                                 get_function("0"));
      }

    return boundary_handler;
  }

  static std::shared_ptr<Ddhdg::HomogeneousProblem<2>>
  get_problem()
  {
    const unsigned int dim = 2;

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


TEST_F(NeumannBCTrigonometricTest, NeumannBCTrigonometricTest) // NOLINT
{
  const auto V_expected_solution = get_V_expected_solution();
  const auto n_expected_solution = get_n_expected_solution();

  this->set_multithreading(false);
  this->refine_grid(3, false);

  const Ddhdg::NonlinearIterationResults status = this->run();

  ASSERT_TRUE(status.converged);

  const double V_l2_error =
    this->estimate_l2_error(V_expected_solution, Ddhdg::Component::V);
  const double n_l2_error =
    this->estimate_l2_error(n_expected_solution, Ddhdg::Component::n);

  EXPECT_LT(V_l2_error, 1e-2);
  EXPECT_LT(n_l2_error, 1e-2);
}
