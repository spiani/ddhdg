#include <deal.II/grid/grid_generator.h>

#include <ddhdg.h>
#include <function_tools.h>
#include <gtest/gtest.h>

template <typename D>
class SetComponentMethod
  : public Ddhdg::NPSolver<D::value, Ddhdg::HomogeneousPermittivity<D::value>>,
    public ::testing::Test
{
public:
  SetComponentMethod()
    : Ddhdg::NPSolver<D::value, Ddhdg::HomogeneousPermittivity<D::value>>(
        get_problem(),
        std::make_shared<Ddhdg::NPSolverParameters>(0, 1, 2)){};

protected:
  static std::shared_ptr<dealii::FunctionParser<D::value>>
  get_expected_solution(const Ddhdg::Component c)
  {
    const unsigned int dim = D::value;

    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution =
      std::make_shared<dealii::FunctionParser<dim>>();

    switch (c)
      {
        case Ddhdg::Component::V:
          switch (dim)
            {
              case 1:
                expected_solution->initialize(
                  dealii::FunctionParser<dim>::default_variable_names(),
                  "atan(10 * x^2)",
                  Ddhdg::Constants::constants);
                break;
              case 2:
                expected_solution->initialize(
                  dealii::FunctionParser<dim>::default_variable_names(),
                  "atan(10 * (x^2 + y^2))",
                  Ddhdg::Constants::constants);
                break;
              default:
                expected_solution->initialize(
                  dealii::FunctionParser<dim>::default_variable_names(),
                  "atan(10 * (x^2 + y^2 + z^2))",
                  Ddhdg::Constants::constants);
                break;
            }
          break;
        case Ddhdg::Component::n:
          switch (dim)
            {
              case 1:
                expected_solution->initialize(
                  dealii::FunctionParser<dim>::default_variable_names(),
                  "sin(x)",
                  Ddhdg::Constants::constants);
                break;
              case 2:
                expected_solution->initialize(
                  dealii::FunctionParser<dim>::default_variable_names(),
                  "sin(x) * sin(y)",
                  Ddhdg::Constants::constants);
                break;
              default:
                expected_solution->initialize(
                  dealii::FunctionParser<dim>::default_variable_names(),
                  "sin(x) * sin(y) * sin(z)",
                  Ddhdg::Constants::constants);
                break;
            }
          break;
        default:
          switch (dim)
            {
              case 1:
                expected_solution->initialize(
                  dealii::FunctionParser<dim>::default_variable_names(),
                  "sin(pi * x)",
                  Ddhdg::Constants::constants);
                break;
              case 2:
                expected_solution->initialize(
                  dealii::FunctionParser<dim>::default_variable_names(),
                  "sin(pi * x) * sin(pi * y)",
                  Ddhdg::Constants::constants);
                break;
              default:
                expected_solution->initialize(
                  dealii::FunctionParser<dim>::default_variable_names(),
                  "sin(pi * x) * sin(pi * y) * sin(pi * z)",
                  Ddhdg::Constants::constants);
                break;
            }
          break;
      }

    return expected_solution;
  }

  static std::shared_ptr<dealii::Triangulation<D::value>>
  get_triangulation()
  {
    const unsigned int dim = D::value;

    std::shared_ptr<dealii::Triangulation<dim>> triangulation =
      std::make_shared<dealii::Triangulation<dim>>();

    dealii::GridGenerator::hyper_cube(*triangulation, -1, 1, false);
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


TYPED_TEST_CASE(SetComponentMethod, dimensions);



TYPED_TEST(SetComponentMethod, interpolation) // NOLINT
{
  const unsigned int dim = TypeParam::value;

  constexpr double V_TOLERANCE = 5e-1;
  constexpr double N_TOLERANCE = 1e-2;
  constexpr double P_TOLERANCE = 1e-2;

  constexpr double WN_TOLERANCE = 5e-2;
  constexpr double WP_TOLERANCE = 5e-2;

  constexpr double V_TRACE_TOLERANCE = 5e-1;
  constexpr double N_TRACE_TOLERANCE = 5e-2;
  constexpr double P_TRACE_TOLERANCE = 5e-2;

  const auto V_expected_solution =
    TestFixture::get_expected_solution(Ddhdg::Component::V);
  const auto n_expected_solution =
    TestFixture::get_expected_solution(Ddhdg::Component::n);
  const auto p_expected_solution =
    TestFixture::get_expected_solution(Ddhdg::Component::p);

  const auto Wn_expected_solution = std::make_shared<Ddhdg::Opposite<dim>>(
    std::make_shared<Ddhdg::Gradient<dim>>(n_expected_solution));
  const auto Wp_expected_solution = std::make_shared<Ddhdg::Opposite<dim>>(
    std::make_shared<Ddhdg::Gradient<dim>>(p_expected_solution));

  this->set_multithreading(false);
  this->refine_grid(3, false);
  this->set_component(Ddhdg::Component::V, V_expected_solution, false);
  this->set_component(Ddhdg::Component::p, p_expected_solution, false);
  this->set_component(Ddhdg::Component::n, n_expected_solution, false);

  const double V_l2_error =
    this->estimate_l2_error(V_expected_solution, Ddhdg::Component::V);
  const double n_l2_error =
    this->estimate_l2_error(n_expected_solution, Ddhdg::Component::n);
  const double p_l2_error =
    this->estimate_l2_error(p_expected_solution, Ddhdg::Component::p);

  EXPECT_LT(V_l2_error, V_TOLERANCE);
  EXPECT_LT(n_l2_error, N_TOLERANCE);
  EXPECT_LT(p_l2_error, P_TOLERANCE);

  const double Wn_l2_error =
    this->estimate_l2_error(Wn_expected_solution, Ddhdg::Displacement::Wn);
  const double Wp_l2_error =
    this->estimate_l2_error(Wp_expected_solution, Ddhdg::Displacement::Wp);

  EXPECT_LT(Wn_l2_error, WN_TOLERANCE);
  EXPECT_LT(Wp_l2_error, WP_TOLERANCE);

  const double V_l2_trace_error =
    this->estimate_l2_error_on_trace(V_expected_solution, Ddhdg::Component::V);
  const double n_l2_trace_error =
    this->estimate_l2_error_on_trace(n_expected_solution, Ddhdg::Component::n);
  const double p_l2_trace_error =
    this->estimate_l2_error_on_trace(p_expected_solution, Ddhdg::Component::p);

  EXPECT_LT(V_l2_trace_error, V_TRACE_TOLERANCE);
  EXPECT_LT(n_l2_trace_error, N_TRACE_TOLERANCE);
  EXPECT_LT(p_l2_trace_error, P_TRACE_TOLERANCE);
}


TYPED_TEST(SetComponentMethod, projection) // NOLINT
{
  const unsigned int dim = TypeParam::value;

  constexpr double V_TOLERANCE = 5e-1;
  constexpr double N_TOLERANCE = 1e-2;
  constexpr double P_TOLERANCE = 3e-3;

  constexpr double WN_TOLERANCE = 1e-2;
  constexpr double WP_TOLERANCE = 2e-2;

  constexpr double V_TRACE_TOLERANCE = 5e-1;
  constexpr double N_TRACE_TOLERANCE = 1e-2;
  constexpr double P_TRACE_TOLERANCE = 1e-2;

  const auto V_expected_solution =
    TestFixture::get_expected_solution(Ddhdg::Component::V);
  const auto n_expected_solution =
    TestFixture::get_expected_solution(Ddhdg::Component::n);
  const auto p_expected_solution =
    TestFixture::get_expected_solution(Ddhdg::Component::p);

  const auto Wn_expected_solution = std::make_shared<Ddhdg::Opposite<dim>>(
    std::make_shared<Ddhdg::Gradient<dim>>(n_expected_solution));
  const auto Wp_expected_solution = std::make_shared<Ddhdg::Opposite<dim>>(
    std::make_shared<Ddhdg::Gradient<dim>>(p_expected_solution));

  this->set_multithreading(false);
  this->refine_grid(3, false);
  this->set_component(Ddhdg::Component::V, V_expected_solution, true);
  this->set_component(Ddhdg::Component::p, p_expected_solution, true);
  this->set_component(Ddhdg::Component::n, n_expected_solution, true);

  const double V_l2_error =
    this->estimate_l2_error(V_expected_solution, Ddhdg::Component::V);
  const double n_l2_error =
    this->estimate_l2_error(n_expected_solution, Ddhdg::Component::n);
  const double p_l2_error =
    this->estimate_l2_error(p_expected_solution, Ddhdg::Component::p);

  EXPECT_LT(V_l2_error, V_TOLERANCE);
  EXPECT_LT(n_l2_error, N_TOLERANCE);
  EXPECT_LT(p_l2_error, P_TOLERANCE);

  const double Wn_l2_error =
    this->estimate_l2_error(Wn_expected_solution, Ddhdg::Displacement::Wn);
  const double Wp_l2_error =
    this->estimate_l2_error(Wp_expected_solution, Ddhdg::Displacement::Wp);

  EXPECT_LT(Wn_l2_error, WN_TOLERANCE);
  EXPECT_LT(Wp_l2_error, WP_TOLERANCE);

  const double V_l2_trace_error =
    this->estimate_l2_error_on_trace(V_expected_solution, Ddhdg::Component::V);
  const double n_l2_trace_error =
    this->estimate_l2_error_on_trace(n_expected_solution, Ddhdg::Component::n);
  const double p_l2_trace_error =
    this->estimate_l2_error_on_trace(p_expected_solution, Ddhdg::Component::p);

  EXPECT_LT(V_l2_trace_error, V_TRACE_TOLERANCE);
  EXPECT_LT(n_l2_trace_error, N_TRACE_TOLERANCE);
  EXPECT_LT(p_l2_trace_error, P_TRACE_TOLERANCE);
}
