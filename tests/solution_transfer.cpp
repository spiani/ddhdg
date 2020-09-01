#include <deal.II/grid/grid_generator.h>

#include <ddhdg.h>
#include <gtest/gtest.h>

template <typename D>
class SolutionTransferTest
  : public Ddhdg::NPSolver<D::value, Ddhdg::HomogeneousProblem<D::value>>,
    public ::testing::Test
{
public:
  SolutionTransferTest()
    : Ddhdg::NPSolver<D::value, Ddhdg::HomogeneousProblem<D::value>>(
        get_problem(),
        std::make_shared<Ddhdg::NPSolverParameters>(),
        std::make_shared<Ddhdg::Adimensionalizer>(1,
                                                  Ddhdg::Constants::Q /
                                                    Ddhdg::Constants::KB,
                                                  1 / Ddhdg::Constants::Q,
                                                  1),
        false){};

protected:
  static std::shared_ptr<dealii::FunctionParser<D::value>>
  get_expected_solution()
  {
    const unsigned int dim = D::value;

    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      "cos(pi/2 * x)",
      Ddhdg::Constants::constants);
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
                       "pi^2 / (4 * q) * cos(pi/2 * x) ",
                       Ddhdg::Constants::constants);
    return doping;
  }

  static std::shared_ptr<Ddhdg::BoundaryConditionHandler<D::value>>
  get_boundary_conditions()
  {
    const unsigned int dim = D::value;

    std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>> boundary_handler =
      std::make_shared<Ddhdg::BoundaryConditionHandler<dim>>();

    for (unsigned int i = 0; i < 6; i++)
      {
        boundary_handler->add_boundary_condition(i,
                                                 Ddhdg::dirichlet,
                                                 Ddhdg::V,
                                                 get_expected_solution());
        boundary_handler->add_boundary_condition(
          i,
          Ddhdg::dirichlet,
          Ddhdg::n,
          std::make_shared<dealii::Functions::ZeroFunction<dim>>(1));
        boundary_handler->add_boundary_condition(
          i,
          Ddhdg::dirichlet,
          Ddhdg::p,
          std::make_shared<dealii::Functions::ZeroFunction<dim>>(1));
      }

    return boundary_handler;
  }

  static std::shared_ptr<Ddhdg::HomogeneousProblem<D::value>>
  get_problem()
  {
    const unsigned int dim = D::value;

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



TYPED_TEST_SUITE(SolutionTransferTest, dimensions, );



TYPED_TEST(SolutionTransferTest, SolutionTransferTest) // NOLINT
{
  constexpr unsigned int dim       = TypeParam::value;
  const double           TOLERANCE = pow(1e-1, 4 - dim);

  const auto expected_solution = TestFixture::get_expected_solution();

  this->set_multithreading(false);
  this->refine_grid(4 - dim, false);
  this->set_component(Ddhdg::Component::V,
                      std::make_shared<dealii::Functions::ZeroFunction<dim>>(1),
                      false);
  this->set_component(Ddhdg::Component::n,
                      std::make_shared<dealii::Functions::ZeroFunction<dim>>(1),
                      false);
  this->set_component(Ddhdg::Component::p,
                      std::make_shared<dealii::Functions::ZeroFunction<dim>>(1),
                      false);

  this->set_enabled_components(true, false, false);

  const Ddhdg::NonlinearIterationResults status = this->run();

  const unsigned int number_of_iterations = status.iterations;

  EXPECT_LE(number_of_iterations, 3);

  this->refine_grid(2, true);

  const double V_l2_error =
    this->estimate_l2_error(expected_solution, Ddhdg::Component::V);

  EXPECT_LT(V_l2_error, TOLERANCE);
}
