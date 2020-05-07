#include <deal.II/grid/grid_generator.h>

#include <ddhdg.h>
#include <gtest/gtest.h>


constexpr unsigned int dim = 1;


class LaplacianEqualCosineProblem : public Ddhdg::Problem<dim>
{
public:
  LaplacianEqualCosineProblem(const double grid_size)
    : Ddhdg::Problem<dim>(
        get_triangulation(grid_size),
        std::make_shared<const Ddhdg::HomogeneousPermittivity<dim>>(
          Ddhdg::Constants::Q),
        std::make_shared<const Ddhdg::HomogeneousElectronMobility<dim>>(1.),
        std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>("0", "0", "0"),
        std::make_shared<const Ddhdg::HomogeneousElectronMobility<dim>>(1.),
        std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>("0", "0", "0"),
        std::make_shared<dealii::Functions::ConstantFunction<dim>>(
          Ddhdg::Constants::Q / Ddhdg::Constants::KB),
        get_doping(grid_size),
        get_boundary_conditions(),
        1.,
        1.){};

protected:
  static std::shared_ptr<dealii::Triangulation<dim>>
  get_triangulation(const double grid_size)
  {
    std::shared_ptr<dealii::Triangulation<dim>> triangulation =
      std::make_shared<dealii::Triangulation<dim>>();

    dealii::GridGenerator::hyper_cube(*triangulation,
                                      -grid_size / 2.,
                                      grid_size / 2.,
                                      true);

    triangulation->refine_global(8);

    return triangulation;
  }

  static std::shared_ptr<dealii::Function<dim>>
  get_doping(const double grid_size)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> doping =
      std::make_shared<dealii::FunctionParser<dim>>();
    doping->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                       "cos(pi * x / " + std::to_string(grid_size) + ")",
                       Ddhdg::Constants::constants);
    return doping;
  }

  static std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>>
  get_boundary_conditions()
  {
    std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>> boundary_handler =
      std::make_shared<Ddhdg::BoundaryConditionHandler<dim>>();

    for (const Ddhdg::Component c : Ddhdg::all_components())
      {
        for (unsigned int i = 0; i < 2; i++)
          {
            boundary_handler->add_boundary_condition(
              i,
              Ddhdg::dirichlet,
              c,
              std::make_shared<dealii::Functions::ZeroFunction<dim>>());
          }
        for (unsigned int i = 2; i < 6; i++)
          {
            boundary_handler->add_boundary_condition(
              i,
              Ddhdg::neumann,
              c,
              std::make_shared<dealii::Functions::ZeroFunction<dim>>());
          }
      }

    return boundary_handler;
  }
};



TEST(AdimensionalizerTest, scale_length)
{
  constexpr double EXPECTED_VALUE =
    4 / (Ddhdg::Constants::PI * Ddhdg::Constants::PI);
  const double     EXPECTED_GRADIENT_VALUE = sqrt(2) / Ddhdg::Constants::PI;
  constexpr double TOLERANCE               = 0.1;

  std::shared_ptr<LaplacianEqualCosineProblem> problem_in_m =
    std::make_shared<LaplacianEqualCosineProblem>(2.);
  std::shared_ptr<LaplacianEqualCosineProblem> problem_in_mm =
    std::make_shared<LaplacianEqualCosineProblem>(2e3);
  std::shared_ptr<LaplacianEqualCosineProblem> problem_in_km =
    std::make_shared<LaplacianEqualCosineProblem>(2e-3);

  std::shared_ptr<Ddhdg::Adimensionalizer> adimensionalizer_for_m =
    std::make_shared<Ddhdg::Adimensionalizer>(
      1, Ddhdg::Constants::Q / Ddhdg::Constants::KB, 1);
  std::shared_ptr<Ddhdg::Adimensionalizer> adimensionalizer_for_mm =
    std::make_shared<Ddhdg::Adimensionalizer>(
      1e-3, Ddhdg::Constants::Q / Ddhdg::Constants::KB, 1);
  std::shared_ptr<Ddhdg::Adimensionalizer> adimensionalizer_for_km =
    std::make_shared<Ddhdg::Adimensionalizer>(
      1e3, Ddhdg::Constants::Q / Ddhdg::Constants::KB, 1);

  std::shared_ptr<Ddhdg::NPSolverParameters> parameters =
    std::make_shared<Ddhdg::NPSolverParameters>();

  Ddhdg::NPSolver<dim> solver_m(problem_in_m,
                                parameters,
                                adimensionalizer_for_m);
  Ddhdg::NPSolver<dim> solver_mm(problem_in_mm,
                                 parameters,
                                 adimensionalizer_for_mm);
  Ddhdg::NPSolver<dim> solver_km(problem_in_km,
                                 parameters,
                                 adimensionalizer_for_km);

  solver_m.set_enabled_components(true, false, false);
  solver_mm.set_enabled_components(true, false, false);
  solver_km.set_enabled_components(true, false, false);

  const dealii::Point<dim> p_value(0.);

  solver_m.run();
  solver_mm.run();
  solver_km.run();

  const double m_value =
    solver_m.get_solution_on_a_point(p_value, Ddhdg::Component::V);
  const double mm_value =
    solver_mm.get_solution_on_a_point(p_value, Ddhdg::Component::V);
  const double km_value =
    solver_km.get_solution_on_a_point(p_value, Ddhdg::Component::V);

  EXPECT_NEAR(m_value, EXPECTED_VALUE, TOLERANCE);
  EXPECT_NEAR(mm_value, EXPECTED_VALUE, TOLERANCE);
  EXPECT_NEAR(km_value, EXPECTED_VALUE, TOLERANCE);

  const dealii::Vector<double> m_gradient_value =
    solver_m.get_solution_on_a_point(dealii::Point<dim>(0.5),
                                     Ddhdg::Displacement::E);
  const dealii::Vector<double> mm_gradient_value =
    solver_mm.get_solution_on_a_point(dealii::Point<dim>(500),
                                      Ddhdg::Displacement::E);
  const dealii::Vector<double> km_gradient_value =
    solver_km.get_solution_on_a_point(dealii::Point<dim>(5e-4),
                                      Ddhdg::Displacement::E);

  EXPECT_NEAR(m_gradient_value[0], EXPECTED_GRADIENT_VALUE, TOLERANCE);
  EXPECT_NEAR(mm_gradient_value[0], EXPECTED_GRADIENT_VALUE, TOLERANCE);
  EXPECT_NEAR(km_gradient_value[0], EXPECTED_GRADIENT_VALUE, TOLERANCE);
}
