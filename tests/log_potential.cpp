#include <deal.II/grid/grid_generator.h>

#include <ddhdg.h>
#include <gtest/gtest.h>

template <typename D>
class LogPotentialTest
  : public Ddhdg::NPSolver<D::value, Ddhdg::HomogeneousPermittivity<D::value>>,
    public ::testing::Test
{
public:
  LogPotentialTest()
    : Ddhdg::NPSolver<D::value, Ddhdg::HomogeneousPermittivity<D::value>>(
        get_problem(),
        std::make_shared<Ddhdg::NPSolverParameters>(1, 2, 1),
        std::make_shared<Ddhdg::Adimensionalizer>(1,
                                                  Ddhdg::Constants::Q /
                                                    Ddhdg::Constants::KB,
                                                  1 / Ddhdg::Constants::Q,
                                                  1)){};

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
  get_expected_solution(const Ddhdg::Component c)
  {
    const unsigned int dim = D::value;

    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution =
      std::make_shared<dealii::FunctionParser<dim>>();
    if (c == Ddhdg::Component::V)
      expected_solution->initialize(
        dealii::FunctionParser<dim>::default_variable_names(),
        "-2 * log(x)",
        Ddhdg::Constants::constants);
    if (c == Ddhdg::Component::n)
      expected_solution->initialize(
        dealii::FunctionParser<dim>::default_variable_names(),
        "2 / (x^2 * q)",
        Ddhdg::Constants::constants);
    return expected_solution;
  }

  static std::shared_ptr<dealii::Triangulation<D::value>>
  get_triangulation()
  {
    const unsigned int dim = D::value;

    std::shared_ptr<dealii::Triangulation<dim>> triangulation =
      std::make_shared<dealii::Triangulation<dim>>();

    dealii::GridGenerator::hyper_cube(*triangulation, 1, 2, false);
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

    for (unsigned int i = 0; i < 2; i++)
      {
        boundary_handler->add_boundary_condition(i,
                                                 Ddhdg::dirichlet,
                                                 Ddhdg::Component::V,
                                                 get_expected_solution(
                                                   Ddhdg::Component::V));
        boundary_handler->add_boundary_condition(i,
                                                 Ddhdg::dirichlet,
                                                 Ddhdg::Component::n,
                                                 get_expected_solution(
                                                   Ddhdg::Component::n));
        boundary_handler->add_boundary_condition(i,
                                                 Ddhdg::dirichlet,
                                                 Ddhdg::Component::p,
                                                 get_zero_function());
      }

    return boundary_handler;
  }

  static std::shared_ptr<
    Ddhdg::Problem<D::value, Ddhdg::HomogeneousPermittivity<D::value>>>
  get_problem()
  {
    const unsigned int dim = D::value;

    std::shared_ptr<Ddhdg::Problem<dim, Ddhdg::HomogeneousPermittivity<dim>>>
      problem = std::make_shared<
        Ddhdg::Problem<dim, Ddhdg::HomogeneousPermittivity<dim>>>(
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



TYPED_TEST_CASE(LogPotentialTest, dimensions);



TYPED_TEST(LogPotentialTest, LogPotentialTest) // NOLINT
{
  const unsigned int dim = TypeParam::value;

  const double V_tolerance = (dim == 3) ? 2e-2 : 1e-2;
  const double n_tolerance = 1e-2;

  const auto zero_function       = TestFixture::get_zero_function();
  const auto V_expected_solution = TestFixture::get_expected_solution(Ddhdg::V);
  const auto n_expected_solution = TestFixture::get_expected_solution(Ddhdg::n);

  this->set_multithreading(false);
  this->refine_grid(4 - dim, false);
  this->set_current_solution(zero_function,
                             zero_function,
                             zero_function,
                             false);

  const Ddhdg::NonlinearIterationResults status = this->run();

  const unsigned int number_of_iterations = status.iterations;

  EXPECT_LE(number_of_iterations, 25);

  const double V_l2_error =
    this->estimate_l2_error(V_expected_solution, Ddhdg::Component::V);
  const double n_l2_error =
    this->estimate_l2_error(n_expected_solution, Ddhdg::Component::n);

  EXPECT_LT(V_l2_error, V_tolerance);
  EXPECT_LT(n_l2_error, n_tolerance);
}
