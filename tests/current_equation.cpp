#include <deal.II/grid/grid_generator.h>

#include <ddhdg.h>
#include <gtest/gtest.h>


template <unsigned int     dimension,
          unsigned int     jump_width,
          Ddhdg::Component component,
          unsigned int     v_degree,
          unsigned int     n_degree,
          unsigned int     p_degree>
struct TestParameters
{
  static constexpr unsigned int     D     = dimension;
  static constexpr double           k     = 1. / jump_width;
  static constexpr Ddhdg::Component c     = component;
  static constexpr unsigned int     v_deg = v_degree;
  static constexpr unsigned int     n_deg = n_degree;
  static constexpr unsigned int     p_deg = p_degree;
};


template <typename TestParameters>
class CurrentEquationTest
  : public Ddhdg::NPSolver<TestParameters::D,
                           Ddhdg::HomogeneousProblem<TestParameters::D>>,
    public ::testing::Test
{
public:
  CurrentEquationTest()
    : Ddhdg::NPSolver<TestParameters::D,
                      Ddhdg::HomogeneousProblem<TestParameters::D>>(
        get_problem(),
        std::make_shared<Ddhdg::NPSolverParameters>(TestParameters::v_deg,
                                                    TestParameters::n_deg,
                                                    TestParameters::p_deg),
        std::make_shared<Ddhdg::Adimensionalizer>(1,
                                                  Ddhdg::Constants::Q /
                                                    Ddhdg::Constants::KB,
                                                  1))
  {
    this->log_standard_level = Ddhdg::Logging::severity_level::debug;
  };

protected:
  static std::shared_ptr<dealii::Function<TestParameters::D>>
  get_expected_solution()
  {
    const unsigned int dim = TestParameters::D;
    constexpr double   k   = TestParameters::k;

    std::map<std::string, double> f_constants;
    for (auto const &[name, value] : Ddhdg::Constants::constants)
      f_constants.insert({name, value});
    f_constants.insert({"k", k});

    const std::shared_ptr<const dealii::Function<dim>> zero_function =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>();
    const std::shared_ptr<const dealii::Function<dim>> one_function =
      std::make_shared<dealii::Functions::ConstantFunction<dim>>(1.);

    std::shared_ptr<dealii::FunctionParser<dim>> smaller_than_k1 =
      std::make_shared<dealii::FunctionParser<dim>>();
    smaller_than_k1->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      "(0.5 - k/2) - x",
      f_constants);

    std::shared_ptr<dealii::FunctionParser<dim>> bigger_than_k2 =
      std::make_shared<dealii::FunctionParser<dim>>();
    bigger_than_k2->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      "x - (0.5 + k/2)",
      f_constants);

    std::string f_description = "1/32*(16*k^7 - 640*x^7 - 35*k^6 + "
                                "672*(k^2 - 5)*x^5 + 2240*x^6 - "
                                "560*(3*k^2 - 5)*x^4 + 35*k^4 - "
                                "280*(k^4 - 6*k^2 + 5)*x^3 + "
                                "420*(k^4 - 2*k^2 + 1)*x^2 - "
                                "21*k^2 + 70*(k^6 - 3*k^4 + 3*k^2 - 1)*x + "
                                "5)/k^7";

    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution_not_zero =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution_not_zero->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      f_description,
      f_constants);

    auto f_temp = std::make_shared<Ddhdg::PiecewiseFunction<dim>>(
      bigger_than_k2, one_function, expected_solution_not_zero);

    auto expected_solution =
      std::make_shared<Ddhdg::PiecewiseFunction<dim>>(smaller_than_k1,
                                                      zero_function,
                                                      f_temp);

    return expected_solution;
  }

  static std::shared_ptr<dealii::Triangulation<TestParameters::D>>
  get_triangulation()
  {
    const unsigned int dim = TestParameters::D;

    std::shared_ptr<dealii::Triangulation<dim>> triangulation =
      std::make_shared<dealii::Triangulation<dim>>();

    dealii::GridGenerator::hyper_cube(*triangulation, 0, 1, true);
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

    std::shared_ptr<dealii::FunctionParser<dim>> doping =
      std::make_shared<dealii::FunctionParser<dim>>();
    doping->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                       "0",
                       Ddhdg::Constants::constants);
    return doping;
  }

  static std::shared_ptr<Ddhdg::BoundaryConditionHandler<TestParameters::D>>
  get_boundary_conditions()
  {
    const unsigned int dim = TestParameters::D;

    std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>> boundary_handler =
      std::make_shared<Ddhdg::BoundaryConditionHandler<dim>>();

    const auto one_function =
      std::make_shared<dealii::Functions::ConstantFunction<dim>>(1.);
    const auto zero_function =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>();

    for (const auto c : Ddhdg::all_primary_components())
      for (unsigned int i = 0; i < 6; i++)
        {
          boundary_handler->add_boundary_condition(
            i,
            (i / 2 == 0) ? Ddhdg::dirichlet : Ddhdg::neumann,
            c,
            (i / 2 == 0) ? ((i == 1) ? one_function : zero_function) :
                           zero_function);
        }

    return boundary_handler;
  }

  static std::shared_ptr<Ddhdg::HomogeneousProblem<TestParameters::D>>
  get_problem()
  {
    constexpr unsigned int dim = TestParameters::D;
    constexpr double       k   = TestParameters::k;

    std::map<std::string, double> f_constants;
    for (auto const &[name, value] : Ddhdg::Constants::constants)
      f_constants.insert({name, value});
    f_constants.insert({"k", k});

    std::shared_ptr<dealii::Function<dim>> zero_function =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>();

    std::shared_ptr<dealii::FunctionParser<dim>> smaller_than_k1 =
      std::make_shared<dealii::FunctionParser<dim>>();
    smaller_than_k1->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      "(0.5 - k/2) - x",
      f_constants);

    std::shared_ptr<dealii::FunctionParser<dim>> bigger_than_k2 =
      std::make_shared<dealii::FunctionParser<dim>>();
    bigger_than_k2->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      "x - (0.5 + k/2)",
      f_constants);

    std::string r0_string = "-105/4 * (32*x^5 - k^4 - 16*(k^2 - 5)*x^3 - "
                            "80*x^4 + 8*(3*k^2 - 5)*x^2 + 2*k^2 + "
                            "2*(k^4 - 6*k^2 + 5)*x - 1)/k^7";

    std::shared_ptr<dealii::FunctionParser<dim>> r0_middle =
      std::make_shared<dealii::FunctionParser<dim>>();
    r0_middle->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                          r0_string,
                          f_constants);

    auto r0_temp =
      std::make_shared<Ddhdg::PiecewiseFunction<dim>>(bigger_than_k2,
                                                      zero_function,
                                                      r0_middle);

    auto r0 = std::make_shared<Ddhdg::PiecewiseFunction<dim>>(smaller_than_k1,
                                                              zero_function,
                                                              r0_temp);

    std::shared_ptr<Ddhdg::HomogeneousProblem<dim>> problem =
      std::make_shared<Ddhdg::HomogeneousProblem<dim>>(
        get_triangulation(),
        std::make_shared<const Ddhdg::HomogeneousPermittivity<dim>>(1.),
        std::make_shared<const Ddhdg::HomogeneousElectronMobility<dim>>(1.),
        std::make_shared<const Ddhdg::HomogeneousElectronMobility<dim>>(1.),
        std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>(r0,
                                                              zero_function,
                                                              zero_function),
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
  ::testing::Types<TestParameters<1, 2, Ddhdg::Component::n, 0, 1, 0>,
                   TestParameters<1, 8, Ddhdg::Component::n, 0, 1, 0>,
                   TestParameters<1, 32, Ddhdg::Component::n, 0, 1, 0>,
                   TestParameters<1, 64, Ddhdg::Component::n, 0, 1, 0>,
                   TestParameters<1, 2, Ddhdg::Component::n, 0, 2, 0>,
                   TestParameters<1, 8, Ddhdg::Component::n, 0, 2, 0>,
                   TestParameters<1, 32, Ddhdg::Component::n, 0, 2, 0>,
                   TestParameters<1, 64, Ddhdg::Component::n, 0, 2, 0>,
                   TestParameters<1, 2, Ddhdg::Component::p, 0, 0, 1>,
                   TestParameters<1, 8, Ddhdg::Component::p, 0, 0, 1>,
                   TestParameters<1, 32, Ddhdg::Component::p, 0, 0, 1>,
                   TestParameters<1, 64, Ddhdg::Component::p, 0, 0, 1>,
                   TestParameters<1, 2, Ddhdg::Component::p, 0, 0, 2>,
                   TestParameters<1, 8, Ddhdg::Component::p, 0, 0, 2>,
                   TestParameters<1, 32, Ddhdg::Component::p, 0, 0, 2>,
                   TestParameters<1, 64, Ddhdg::Component::p, 0, 0, 2>,
                   TestParameters<2, 2, Ddhdg::Component::n, 0, 1, 0>,
                   TestParameters<2, 4, Ddhdg::Component::n, 0, 2, 0>,
                   TestParameters<2, 6, Ddhdg::Component::n, 0, 3, 0>,
                   TestParameters<2, 2, Ddhdg::Component::p, 0, 0, 1>,
                   TestParameters<2, 4, Ddhdg::Component::p, 0, 0, 2>,
                   TestParameters<2, 6, Ddhdg::Component::p, 0, 0, 3>,
                   TestParameters<3, 2, Ddhdg::Component::n, 0, 1, 0>,
                   TestParameters<3, 2, Ddhdg::Component::p, 0, 0, 1>>;



TYPED_TEST_SUITE(CurrentEquationTest, parameters, );



TYPED_TEST(CurrentEquationTest, StartingFromZero) // NOLINT
{
  constexpr unsigned int     dim        = TypeParam::D;
  constexpr Ddhdg::Component c          = TypeParam::c;
  constexpr double           jump_width = TypeParam::k;

  constexpr double TOLERANCE = (dim == 1) ? 2e-2 : (dim == 2) ? 3e-2 : 5e-1;
  constexpr int REFINEMENTS  = (dim == 1) ? ((jump_width < 1. / 24) ? 11 : 8) :
                               (dim == 2) ? 5 :
                                            3;

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



TYPED_TEST(CurrentEquationTest, StartingFromNoise) // NOLINT
{
  constexpr unsigned int     dim        = TypeParam::D;
  constexpr Ddhdg::Component c          = TypeParam::c;
  constexpr double           jump_width = TypeParam::k;

  constexpr double TOLERANCE = (dim == 1) ? 2e-2 : (dim == 2) ? 3e-2 : 5e-1;
  constexpr int REFINEMENTS  = (dim == 1) ? ((jump_width < 1. / 24) ? 11 : 8) :
                               (dim == 2) ? 5 :
                                            3;

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
