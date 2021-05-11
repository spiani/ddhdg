#include <deal.II/grid/grid_generator.h>

#include <ddhdg.h>
#include <gtest/gtest.h>


template <unsigned int     dimension,
          unsigned int     direction,
          Ddhdg::Component component,
          unsigned int     v_degree,
          unsigned int     n_degree,
          unsigned int     p_degree>
struct TestParameters
{
  static constexpr unsigned int     D     = dimension;
  static constexpr unsigned int     S     = direction;
  static constexpr Ddhdg::Component c     = component;
  static constexpr unsigned int     v_deg = v_degree;
  static constexpr unsigned int     n_deg = n_degree;
  static constexpr unsigned int     p_deg = p_degree;
};


template <typename TestParameters>
class LaplacianTest
  : public Ddhdg::NPSolver<TestParameters::D,
                           Ddhdg::HomogeneousProblem<TestParameters::D>>,
    public ::testing::Test
{
public:
  LaplacianTest()
    : Ddhdg::NPSolver<TestParameters::D,
                      Ddhdg::HomogeneousProblem<TestParameters::D>>(
        get_problem(),
        std::make_shared<Ddhdg::NPSolverParameters>(TestParameters::v_deg,
                                                    TestParameters::n_deg,
                                                    TestParameters::p_deg),
        std::make_shared<Ddhdg::Adimensionalizer>(
          1,
          Ddhdg::Constants::Q / Ddhdg::Constants::KB,
          (TestParameters::c == Ddhdg::Component::V) ? 1 / Ddhdg::Constants::Q :
                                                       1))
  {
    this->log_standard_level = Ddhdg::Logging::severity_level::debug;
  };

protected:
  static std::shared_ptr<dealii::FunctionParser<TestParameters::D>>
  get_expected_solution()
  {
    const unsigned int     dim = TestParameters::D;
    const Ddhdg::Component c   = TestParameters::c;
    const unsigned int     s   = TestParameters::S;

    std::string f_description;
    switch (c)
      {
        case Ddhdg::Component::V:
          f_description = (s == 0) ? "x^3 - x" :
                          (s == 1) ? "y^3 - y" :
                                     "z^3 - z";
          break;
        case Ddhdg::Component::n:
          f_description = (s == 0) ? "-1/20*x^5 + 7/6*x^3 - 67/60*x" :
                          (s == 1) ? "-1/20*y^5 + 7/6*y^3 - 67/60*y" :
                                     "-1/20*z^5 + 7/6*z^3 - 67/60*z";
          break;
        case Ddhdg::Component::p:
          f_description = (s == 0) ? "x^3 - x" :
                          (s == 1) ? "y^3 - y" :
                                     "z^3 - z";
          break;
        default:
          Assert(false, Ddhdg::InvalidComponent())
      }

    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      f_description,
      Ddhdg::Constants::constants);
    return expected_solution;
  }

  static std::shared_ptr<dealii::Triangulation<TestParameters::D>>
  get_triangulation()
  {
    const unsigned int dim = TestParameters::D;

    std::shared_ptr<dealii::Triangulation<dim>> triangulation =
      std::make_shared<dealii::Triangulation<dim>>();

    dealii::GridGenerator::hyper_cube(*triangulation, -1, 1, true);
    return triangulation;
  }

  static std::shared_ptr<dealii::FunctionParser<TestParameters::D>>
  get_temperature()
  {
    const unsigned int dim = TestParameters::D;

    std::shared_ptr<dealii::FunctionParser<dim>> temperature =
      std::make_shared<dealii::FunctionParser<dim>>();
    temperature->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      "q / kb",
      Ddhdg::Constants::constants);
    return temperature;
  }

  static std::shared_ptr<dealii::FunctionParser<TestParameters::D>>
  get_doping()
  {
    const unsigned int dim = TestParameters::D;
    const unsigned int s   = TestParameters::S;

    std::string f_description = (s == 0) ? "-6 * x / q" :
                                (s == 1) ? "-6 * y / q" :
                                           "-6 * z / q";

    std::shared_ptr<dealii::FunctionParser<dim>> doping =
      std::make_shared<dealii::FunctionParser<dim>>();
    doping->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                       f_description,
                       Ddhdg::Constants::constants);
    return doping;
  }

  static std::shared_ptr<Ddhdg::BoundaryConditionHandler<TestParameters::D>>
  get_boundary_conditions()
  {
    const unsigned int dim = TestParameters::D;
    const unsigned int s   = TestParameters::S;

    std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>> boundary_handler =
      std::make_shared<Ddhdg::BoundaryConditionHandler<dim>>();

    for (const auto c : Ddhdg::all_primary_components())
      for (unsigned int i = 0; i < 6; i++)
        {
          boundary_handler->add_boundary_condition(
            i,
            (i / 2 == s) ? Ddhdg::dirichlet : Ddhdg::neumann,
            c,
            std::make_shared<dealii::Functions::ZeroFunction<dim>>());
        }

    return boundary_handler;
  }

  static std::shared_ptr<Ddhdg::HomogeneousProblem<TestParameters::D>>
  get_problem()
  {
    const unsigned int dim = TestParameters::D;
    const unsigned int s   = TestParameters::S;

    std::string r0_string = (s == 0) ? "(7 * x - x^3)" :
                            (s == 1) ? "(7 * y - y^3)" :
                                       "(7 * z - z^3)";

    std::shared_ptr<Ddhdg::HomogeneousProblem<dim>> problem =
      std::make_shared<Ddhdg::HomogeneousProblem<dim>>(
        get_triangulation(),
        std::make_shared<const Ddhdg::HomogeneousPermittivity<dim>>(1.),
        std::make_shared<const Ddhdg::HomogeneousMobility<dim>>(
          1., Ddhdg::Component::n),
        std::make_shared<const Ddhdg::HomogeneousMobility<dim>>(
          1., Ddhdg::Component::p),
        std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>(r0_string,
                                                              "0",
                                                              "1"),
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



using parameters =
  ::testing::Types<TestParameters<1, 0, Ddhdg::Component::V, 1, 0, 0>,
                   TestParameters<1, 0, Ddhdg::Component::V, 1, 4, 4>,
                   TestParameters<2, 0, Ddhdg::Component::V, 1, 2, 0>,
                   TestParameters<2, 1, Ddhdg::Component::V, 1, 1, 1>,
                   TestParameters<3, 0, Ddhdg::Component::V, 1, 1, 1>,
                   TestParameters<3, 1, Ddhdg::Component::V, 1, 2, 1>,
                   TestParameters<3, 2, Ddhdg::Component::V, 1, 1, 1>,
                   TestParameters<1, 0, Ddhdg::Component::V, 2, 2, 2>,
                   TestParameters<2, 1, Ddhdg::Component::V, 2, 2, 2>,
                   TestParameters<2, 0, Ddhdg::Component::V, 3, 3, 3>,
                   // Tests for n
                   TestParameters<1, 0, Ddhdg::Component::n, 0, 1, 0>,
                   TestParameters<1, 0, Ddhdg::Component::n, 4, 1, 4>,
                   TestParameters<2, 0, Ddhdg::Component::n, 2, 1, 0>,
                   TestParameters<2, 1, Ddhdg::Component::n, 1, 1, 1>,
                   TestParameters<3, 0, Ddhdg::Component::n, 1, 1, 1>,
                   TestParameters<3, 1, Ddhdg::Component::n, 2, 1, 0>,
                   TestParameters<3, 2, Ddhdg::Component::n, 1, 1, 1>,
                   TestParameters<1, 0, Ddhdg::Component::n, 2, 2, 2>,
                   TestParameters<2, 1, Ddhdg::Component::n, 2, 2, 2>,
                   TestParameters<2, 0, Ddhdg::Component::n, 3, 5, 3>,
                   // Tests for p
                   TestParameters<1, 0, Ddhdg::Component::p, 0, 0, 1>,
                   TestParameters<1, 0, Ddhdg::Component::p, 4, 4, 1>,
                   TestParameters<2, 0, Ddhdg::Component::p, 0, 2, 1>,
                   TestParameters<2, 1, Ddhdg::Component::p, 1, 1, 1>,
                   TestParameters<3, 0, Ddhdg::Component::p, 1, 1, 1>,
                   TestParameters<3, 1, Ddhdg::Component::p, 2, 0, 1>,
                   TestParameters<3, 2, Ddhdg::Component::p, 1, 1, 1>,
                   TestParameters<1, 0, Ddhdg::Component::p, 2, 2, 2>,
                   TestParameters<2, 1, Ddhdg::Component::p, 2, 2, 2>,
                   TestParameters<2, 0, Ddhdg::Component::p, 3, 3, 3>>;



TYPED_TEST_SUITE(LaplacianTest, parameters, );



TYPED_TEST(LaplacianTest, StartingFromZero) // NOLINT
{
  const unsigned int     dim = TypeParam::D;
  const Ddhdg::Component c   = TypeParam::c;

  constexpr double TOLERANCE   = (dim < 3) ? 2e-2 : 5e-1;
  constexpr int    REFINEMENTS = (dim < 3) ? 4 : 3;

  const auto expected_solution = TestFixture::get_expected_solution();

  this->set_multithreading(false);
  this->refine_grid(REFINEMENTS, false);

  auto zero_function = std::make_shared<dealii::Functions::ZeroFunction<dim>>();

  for (const auto cmp : Ddhdg::all_primary_components())
    this->set_component(cmp, zero_function, false);

  if constexpr (c == Ddhdg::Component::V)
    this->set_enabled_components(true, false, false);
  else if constexpr (c == Ddhdg::Component::n)
    this->set_enabled_components(false, true, false);
  else
    this->set_enabled_components(false, false, true);

  const Ddhdg::NonlinearIterationResults status = this->run();

  const unsigned int number_of_iterations = status.iterations;

  EXPECT_LE(number_of_iterations, 3);

  const double l2_error = this->estimate_l2_error(expected_solution, c);

  EXPECT_LT(l2_error, TOLERANCE);
}



TYPED_TEST(LaplacianTest, StartingFromNoise) // NOLINT
{
  const unsigned int     dim = TypeParam::D;
  const Ddhdg::Component c   = TypeParam::c;

  constexpr double TOLERANCE   = (dim < 3) ? 2e-2 : 5e-1;
  constexpr int    REFINEMENTS = (dim < 3) ? 4 : 3;

  const auto expected_solution = TestFixture::get_expected_solution();

  this->set_multithreading(false);
  this->refine_grid(REFINEMENTS, false);

  auto zero_function = std::make_shared<dealii::Functions::ZeroFunction<dim>>();
  auto noise_function = std::make_shared<dealii::FunctionParser<dim>>();
  std::string noise_function_str =
    (dim == 1) ? "sin(100 * x)" :
    (dim == 2) ? "sin (100 * x) + sin(101 * y + 0.05)" :
                 "sin(100 * x) + sin(101 * y + 0.05) + sin(111 * z + 0.1)";
  noise_function->initialize(
    dealii::FunctionParser<dim>::default_variable_names(),
    noise_function_str,
    Ddhdg::Constants::constants);

  for (const auto cmp : Ddhdg::all_primary_components())
    if (cmp == c)
      this->set_component(c, noise_function, false);
    else
      this->set_component(c, zero_function, false);

  if constexpr (c == Ddhdg::Component::V)
    this->set_enabled_components(true, false, false);
  else if constexpr (c == Ddhdg::Component::n)
    this->set_enabled_components(false, true, false);
  else
    this->set_enabled_components(false, false, true);

  // Force only one iteration
  this->run(0, 0, 1);

  const double l2_error = this->estimate_l2_error(expected_solution, c);

  EXPECT_LT(l2_error, TOLERANCE);
}
