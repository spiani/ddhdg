#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/fe_field_function.h>

#include <ddhdg.h>
#include <gtest/gtest.h>

constexpr unsigned int dim         = 2;
constexpr double       domain_size = 1;

constexpr double ND = 1e22;
constexpr double NA = 1e23;


class LocalChargeNeutralityTest
  : public Ddhdg::NPSolver<dim, Ddhdg::HomogeneousPermittivity<dim>>,
    public ::testing::Test
{
public:
  LocalChargeNeutralityTest()
    : Ddhdg::NPSolver<dim, Ddhdg::HomogeneousPermittivity<dim>>(
        get_problem(),
        std::make_shared<Ddhdg::NPSolverParameters>(3, 1, 1),
        std::make_shared<Ddhdg::Adimensionalizer>(1,
                                                  Ddhdg::Constants::Q /
                                                    Ddhdg::Constants::KB,
                                                  1e23,
                                                  1)){};

protected:
  static std::shared_ptr<dealii::FunctionParser<dim>>
  get_function(const std::string &f_str)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> f =
      std::make_shared<dealii::FunctionParser<dim>>(1);

    f->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                  f_str,
                  Ddhdg::Constants::constants);

    return f;
  }

  static std::shared_ptr<dealii::Triangulation<dim>>
  get_triangulation()
  {
    std::shared_ptr<dealii::Triangulation<dim>> triangulation =
      std::make_shared<dealii::Triangulation<dim>>();

    dealii::GridGenerator::hyper_cube(*triangulation, 0, domain_size, true);
    return triangulation;
  }

  static std::shared_ptr<dealii::Function<dim>>
  get_temperature()
  {
    return std::make_shared<dealii::Functions::ConstantFunction<dim>>(300.);
  }

  static std::shared_ptr<dealii::Function<dim>>
  get_doping()
  {
    constexpr double region1 = domain_size * 0.1;
    constexpr double region2 = domain_size * 0.9;

    const std::string r1_str = std::to_string(region1);
    const std::string r2_str = std::to_string(region2);

    // f1 is positive if x < region1; f2 is positive if x > region2
    std::shared_ptr<dealii::Function<dim>> f1 = get_function(r1_str + " - x");
    std::shared_ptr<dealii::Function<dim>> f2 = get_function("x - " + r2_str);

    const std::shared_ptr<dealii::Function<dim>> r1_doping =
      std::make_shared<dealii::Functions::ConstantFunction<dim>>(ND);
    const std::shared_ptr<dealii::Function<dim>> r2_doping =
      std::make_shared<dealii::Functions::ConstantFunction<dim>>(-NA);
    const std::shared_ptr<dealii::Function<dim>> zero_function =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>(1);

    const std::shared_ptr<dealii::Function<dim>> temp =
      std::make_shared<Ddhdg::PiecewiseFunction<dim>>(f2,
                                                      r2_doping,
                                                      zero_function);

    return std::make_shared<Ddhdg::PiecewiseFunction<dim>>(f1, r1_doping, temp);
  }

  static std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>>
  get_boundary_conditions()
  {
    std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>> boundary_handler =
      std::make_shared<Ddhdg::BoundaryConditionHandler<dim>>();

    return boundary_handler;
  }

  static std::shared_ptr<
    Ddhdg::Problem<dim, Ddhdg::HomogeneousPermittivity<dim>>>
  get_problem()
  {
    std::shared_ptr<Ddhdg::Problem<dim, Ddhdg::HomogeneousPermittivity<dim>>>
      problem = std::make_shared<
        Ddhdg::Problem<dim, Ddhdg::HomogeneousPermittivity<dim>>>(
        get_triangulation(),
        std::make_shared<const Ddhdg::HomogeneousPermittivity<dim>>(
          12.9 * Ddhdg::Constants::EPSILON0),
        std::make_shared<const Ddhdg::HomogeneousElectronMobility<dim>>(1.),
        std::make_shared<const Ddhdg::HomogeneousElectronMobility<dim>>(1.),
        std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>("0", "0", "0"),
        get_temperature(),
        get_doping(),
        get_boundary_conditions(),
        4.7e23,
        9.0e24,
        1.424,
        0.);
    return problem;
  }
};



TEST_F(LocalChargeNeutralityTest, charge_neutrality) // NOLINT
{
  constexpr double TOLERANCE = 0.1;

  this->set_multithreading(false);
  this->refine_grid(4);
  this->set_enabled_components(true, false, false);

  if (!this->initialized)
    this->setup_overall_system();

  this->compute_local_charge_neutrality();

  dealii::Functions::FEFieldFunction<dim> solution(this->dof_handler_cell,
                                                   this->current_solution_cell);

  const unsigned int V_component =
    this->get_component_mask(Ddhdg::Component::V).first_selected_component();

  const std::vector<std::pair<dealii::Point<dim>, double>> expected_values{
    {dealii::Point<dim>(0.05, 0.5), 1.4},
    {dealii::Point<dim>(0.5, 0.5), 0.75},
    {dealii::Point<dim>(.95, 1), 0.1}};

  for (const auto &k : expected_values)
    {
      const double current_value  = solution.value(k.first, V_component);
      const double expected_value = k.second;
      EXPECT_NEAR(current_value, expected_value, TOLERANCE);
    }
}
