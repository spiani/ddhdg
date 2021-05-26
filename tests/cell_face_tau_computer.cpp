#include <deal.II/grid/grid_generator.h>

#include <ddhdg.h>
#include <gtest/gtest.h>

template <int dim>
class SolverForLogPotentialProblem
  : public Ddhdg::NPSolver<dim, Ddhdg::HomogeneousProblem<dim>>
{
public:
  SolverForLogPotentialProblem(
    const std::shared_ptr<Ddhdg::NPSolverParameters> parameters)
    : Ddhdg::NPSolver<dim, Ddhdg::HomogeneousProblem<dim>>(
        get_problem(),
        parameters,
        std::make_shared<Ddhdg::Adimensionalizer>(1,
                                                  Ddhdg::Constants::Q /
                                                    Ddhdg::Constants::KB,
                                                  1 / Ddhdg::Constants::Q,
                                                  1),
        false){};

protected:
  static std::shared_ptr<dealii::FunctionParser<dim>>
  get_expected_solution(const Ddhdg::Component c)
  {
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

  static std::shared_ptr<dealii::Triangulation<dim>>
  get_triangulation()
  {
    std::shared_ptr<dealii::Triangulation<dim>> triangulation =
      std::make_shared<dealii::Triangulation<dim>>();

    dealii::GridGenerator::hyper_cube(*triangulation, 1, 2, false);
    return triangulation;
  }

  static std::shared_ptr<dealii::FunctionParser<dim>>
  get_temperature()
  {
    std::shared_ptr<dealii::FunctionParser<dim>> temperature =
      std::make_shared<dealii::FunctionParser<dim>>();
    temperature->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      "q / kb",
      Ddhdg::Constants::constants);
    return temperature;
  }

  static std::shared_ptr<dealii::FunctionParser<dim>>
  get_doping()
  {
    std::shared_ptr<dealii::FunctionParser<dim>> doping =
      std::make_shared<dealii::FunctionParser<dim>>();
    doping->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                       "0",
                       Ddhdg::Constants::constants);
    return doping;
  }

  static std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>>
  get_boundary_conditions()
  {
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
        boundary_handler->add_boundary_condition(
          i,
          Ddhdg::dirichlet,
          Ddhdg::Component::p,
          std::make_shared<dealii::Functions::ZeroFunction<dim>>());
      }

    return boundary_handler;
  }

  static std::shared_ptr<Ddhdg::HomogeneousProblem<dim>>
  get_problem()
  {
    std::shared_ptr<Ddhdg::HomogeneousProblem<dim>> problem =
      std::make_shared<Ddhdg::HomogeneousProblem<dim>>(
        get_triangulation(),
        std::make_shared<const Ddhdg::HomogeneousPermittivity<dim>>(1.),
        std::make_shared<const Ddhdg::HomogeneousMobility<dim>>(
          1., Ddhdg::Component::n),
        std::make_shared<const Ddhdg::HomogeneousMobility<dim>>(
          1., Ddhdg::Component::p),
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



template <typename D>
class CellFaceTauComputerTest : public testing::Test
{};



using dimensions = ::testing::Types<std::integral_constant<unsigned int, 1>,
                                    std::integral_constant<unsigned int, 2>,
                                    std::integral_constant<unsigned int, 3>>;



TYPED_TEST_SUITE(CellFaceTauComputerTest, dimensions, );



TYPED_TEST(CellFaceTauComputerTest, CompareWithFixedTau)
{
  constexpr unsigned int dim       = TypeParam::value;
  constexpr double       tolerance = 1e-9;

#ifdef DEBUG
  constexpr bool multithreading = false;
#else
  constexpr bool multithreading = true;
#endif

  const std::shared_ptr<Ddhdg::NonlinearSolverParameters>
    nonlinear_solver_parameters =
      std::make_shared<Ddhdg::NonlinearSolverParameters>();

  std::shared_ptr<Ddhdg::FixedTauNPSolverParameters> fixed_tau_parameters =
    std::make_shared<Ddhdg::FixedTauNPSolverParameters>(
      1,
      1,
      1,
      nonlinear_solver_parameters,
      1,
      100,
      10000,
      true,
      multithreading);

  std::shared_ptr<Ddhdg::CellFaceTauNPSolverParameters>
    cell_face_tau_parameters =
      std::make_shared<Ddhdg::CellFaceTauNPSolverParameters>(
        1, 1, 1, nonlinear_solver_parameters, true, multithreading);

  std::unique_ptr<SolverForLogPotentialProblem<dim>> fixed_tau_solver =
    std::make_unique<SolverForLogPotentialProblem<dim>>(fixed_tau_parameters);
  std::unique_ptr<SolverForLogPotentialProblem<dim>> cell_face_tau_solver =
    std::make_unique<SolverForLogPotentialProblem<dim>>(
      cell_face_tau_parameters);

  fixed_tau_solver->refine_grid(4 - dim, false);
  cell_face_tau_solver->refine_grid(4 - dim, false);

  const unsigned int faces_per_cell = dealii::GeometryInfo<dim>::faces_per_cell;
  for (const auto cell : cell_face_tau_solver->get_active_cell_iterator())
    for (unsigned int face = 0; face < faces_per_cell; ++face)
      cell_face_tau_parameters->set_face<dim>(*cell, face, 1., 100., 10000.);

  fixed_tau_solver->run();
  cell_face_tau_solver->run();

  const double error_on_V = fixed_tau_solver->estimate_error(
    *cell_face_tau_solver,
    Ddhdg::Component::V,
    dealii::VectorTools::NormType::Linfty_norm);
  const double error_on_n = fixed_tau_solver->estimate_error(
    *cell_face_tau_solver,
    Ddhdg::Component::n,
    dealii::VectorTools::NormType::Linfty_norm);
  const double error_on_p = fixed_tau_solver->estimate_error(
    *cell_face_tau_solver,
    Ddhdg::Component::p,
    dealii::VectorTools::NormType::Linfty_norm);

  EXPECT_LT(error_on_V, tolerance);
  EXPECT_LT(error_on_n, tolerance);
  EXPECT_LT(error_on_p, tolerance);
}