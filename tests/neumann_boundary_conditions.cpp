#include <deal.II/grid/grid_generator.h>

#include <ddhdg.h>
#include <function_tools.h>
#include <gtest/gtest.h>

template <typename D>
class NeumannBCLinearTest : public Ddhdg::Solver<D::value>,
                            public ::testing::Test
{
public:
  NeumannBCLinearTest()
    : Ddhdg::Solver<D::value>(get_problem()){};

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

    std::shared_ptr<dealii::FunctionParser<dim>> one_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    one_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      "1.",
      Ddhdg::Constants::constants);

    std::shared_ptr<dealii::Function<dim>> minus_one_function =
      std::make_shared<Ddhdg::Opposite<dim>>(one_function);

    // Boundary conditions for n: 0 everywhere
    for (unsigned int i = 0; i < 6; i++)
      {
        boundary_handler->add_boundary_condition(i,
                                                 Ddhdg::dirichlet,
                                                 Ddhdg::n,
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
                                             one_function);
    for (unsigned int i = 2; i < 6; i++)
      {
        boundary_handler->add_boundary_condition(i,
                                                 Ddhdg::neumann,
                                                 Ddhdg::V,
                                                 get_zero_function());
      }


    return boundary_handler;
  }

  static std::shared_ptr<Ddhdg::Problem<D::value>>
  get_problem()
  {
    const unsigned int                   dim = D::value;
    std::shared_ptr<Ddhdg::Problem<dim>> problem =
      std::make_shared<Ddhdg::Problem<dim>>(
        get_triangulation(),
        std::make_shared<const Ddhdg::HomogeneousPermittivity<dim>>(1.),
        std::make_shared<const Ddhdg::HomogeneousElectronMobility<dim>>(1.),
        std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>("0", "0"),
        get_temperature(),
        get_doping(),
        get_boundary_conditions());
    return problem;
  }
};



using dimensions = ::testing::Types<std::integral_constant<unsigned int, 1>,
                                    std::integral_constant<unsigned int, 2>,
                                    std::integral_constant<unsigned int, 3>>;



TYPED_TEST_CASE(NeumannBCLinearTest, dimensions);



TYPED_TEST(NeumannBCLinearTest, NeumannBCLinearTest) // NOLINT
{
  const unsigned int dim = TypeParam::value;

  const auto zero_function     = this->get_zero_function();
  const auto expected_solution = this->get_expected_solution();

  this->set_multithreading(false);
  this->refine_grid(3 - dim);
  this->set_component(Ddhdg::Component::n, zero_function);

  const Ddhdg::NonlinearIteratorStatus status = this->run();

  const unsigned int number_of_iterations = status.iterations;

  EXPECT_LE(number_of_iterations, 3);

  const double V_l2_error =
    this->estimate_l2_error(expected_solution, Ddhdg::Component::V);
  const double n_l2_error =
    this->estimate_l2_error(zero_function, Ddhdg::Component::n);

  EXPECT_LT(V_l2_error, 1e-10);
  EXPECT_LT(n_l2_error, 1e-10);
}
