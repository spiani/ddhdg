#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/fe_field_function.h>

#include <ddhdg.h>
#include <gtest/gtest.h>

constexpr unsigned int dim         = 2;
constexpr double       domain_size = 1;

constexpr double ND = 1e22;
constexpr double NA = 1e23;


class LocalChargeNeutralityTest : public Ddhdg::NPSolver<dim>,
                                  public ::testing::Test
{
public:
  LocalChargeNeutralityTest()
    : Ddhdg::NPSolver<dim>(get_problem(),
                           std::make_shared<Ddhdg::NPSolverParameters>(3, 1, 1),
                           std::make_shared<Ddhdg::Adimensionalizer>(
                             1,
                             Ddhdg::Constants::Q / Ddhdg::Constants::KB,
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

  static std::shared_ptr<Ddhdg::Problem<dim>>
  get_problem()
  {
    std::shared_ptr<Ddhdg::Problem<dim>> problem =
      std::make_shared<Ddhdg::Problem<dim>>(
        get_triangulation(),
        std::make_shared<const Ddhdg::HomogeneousPermittivity<dim>>(
          12.9 * Ddhdg::Constants::EPSILON0),
        std::make_shared<const Ddhdg::HomogeneousElectronMobility<dim>>(1.),
        std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>("0", "0", "0"),
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



TEST_F(LocalChargeNeutralityTest, charge_neutrality_first_guess) // NOLINT
{
  this->set_multithreading(false);
  this->refine_grid(3);
  this->set_enabled_components(true, false, false);

  if (!this->initialized)
    this->setup_overall_system();

  this->set_local_charge_neutrality_first_guess();

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
      EXPECT_NEAR(current_value, expected_value, 0.1);
    }
}



TEST_F(LocalChargeNeutralityTest, charge_neutrality_copy_solution) // NOLINT
{
  constexpr double L2_TOLERANCE     = 1e-12;
  constexpr double LINFTY_TOLERANCE = 1e-12;

  this->set_multithreading(false);
  this->refine_grid(3);
  this->set_enabled_components(true, false, false);

  if (!this->initialized)
    this->setup_overall_system();

  // Create a DoFHandler just for V
  const dealii::ComponentMask V_component_mask =
    this->get_component_mask(Ddhdg::Component::V);

  const unsigned int fsc = V_component_mask.first_selected_component();
  const unsigned int nsc = V_component_mask.n_selected_components();

  const dealii::FiniteElement<dim> &V_fe = this->fe_cell->get_sub_fe(fsc, nsc);

  dealii::DoFHandler<dim> V_dof_handler(*(this->triangulation));
  V_dof_handler.distribute_dofs(V_fe);

  // Create an array where the copy of the potential could be stored
  dealii::Vector<double> potential_copy(V_dof_handler.n_dofs());

  dealii::Vector<double> difference_per_cell(triangulation->n_active_cells());

  std::shared_ptr<dealii::Function<dim>> f1 = get_function("x^2 + y^3");
  std::shared_ptr<dealii::Function<dim>> f2 = get_function("sin(x)");
  std::shared_ptr<dealii::Function<dim>> f3 = get_function("x + cos(y)");

  const std::vector<std::shared_ptr<dealii::Function<dim>>> functions{f1,
                                                                      f2,
                                                                      f3};

  for (const auto &f : functions)
    {
      this->set_component(Ddhdg::Component::V, f, true);
      const double V_l2_norm = this->estimate_l2_error(
        std::make_shared<dealii::Functions::ZeroFunction<dim>>(),
        Ddhdg::Component::V);
      const double V_linfty_norm = this->estimate_linfty_error(
        std::make_shared<dealii::Functions::ZeroFunction<dim>>(),
        Ddhdg::Component::V);

      // Copy the solution on potential_copy
      this->compute_local_charge_neutrality_copy_solution(V_fe,
                                                          V_dof_handler,
                                                          potential_copy);
      // Compute the norm of the copy
      dealii::VectorTools::integrate_difference(
        V_dof_handler,
        potential_copy,
        dealii::Functions::ZeroFunction<dim>(),
        difference_per_cell,
        dealii::QGauss<dim>(this->get_number_of_quadrature_points()),
        dealii::VectorTools::L2_norm);
      const double V_copy_l2_norm =
        dealii::VectorTools::compute_global_error(*(this->triangulation),
                                                  difference_per_cell,
                                                  dealii::VectorTools::L2_norm);
      dealii::VectorTools::integrate_difference(
        V_dof_handler,
        potential_copy,
        dealii::Functions::ZeroFunction<dim>(),
        difference_per_cell,
        dealii::QGauss<dim>(this->get_number_of_quadrature_points()),
        dealii::VectorTools::Linfty_norm);
      const double V_copy_linfty_norm =
        dealii::VectorTools::compute_global_error(
          *(this->triangulation),
          difference_per_cell,
          dealii::VectorTools::Linfty_norm);

      EXPECT_NEAR(V_l2_norm, V_copy_l2_norm, L2_TOLERANCE);
      EXPECT_NEAR(V_linfty_norm, V_copy_linfty_norm, LINFTY_TOLERANCE);
    }
}



TEST_F(LocalChargeNeutralityTest, charge_neutrality_set_solution) // NOLINT
{
  // This test check if charge_neutrality_set_solution method is able to
  // rebuild the overall system, going in the opposite direction than
  // charge_neutrality_copy_solution method
  constexpr double TOLERANCE = 1e-12;

  this->set_multithreading(false);
  this->refine_grid(3);
  this->set_enabled_components(true, false, false);

  if (!this->initialized)
    this->setup_overall_system();

  // Create a DoFHandler just for V
  const dealii::ComponentMask V_component_mask =
    this->get_component_mask(Ddhdg::Component::V);

  const unsigned int fsc = V_component_mask.first_selected_component();
  const unsigned int nsc = V_component_mask.n_selected_components();

  const dealii::FiniteElement<dim> &V_fe = this->fe_cell->get_sub_fe(fsc, nsc);

  dealii::DoFHandler<dim> V_dof_handler(*(this->triangulation));
  V_dof_handler.distribute_dofs(V_fe);

  // Create an array where the copy of the potential could be stored
  dealii::Vector<double> potential_copy(V_dof_handler.n_dofs());

  // Create an array to store a copy of all the components
  dealii::Vector<double> all_system_copy(this->dof_handler_cell.n_dofs());

  // Here you must choose function that can be represented without error inside
  // the space we are using (polynomial of degree 3). Indeed, if you choose a
  // function outside this space, when you use the "set_component" method, E
  // will be the projection of the gradient of this function. When you call
  // the charge_neutrality_copy_solution you will forget this information and,
  // when you try to go back using the charge_neutrality_set_solution, E will be
  // just the gradient of the approximate function. In other worlds, E will
  // start as the projection of the gradient of f and will then become the
  // gradient of the projection of f!
  std::shared_ptr<dealii::Function<dim>> f1 = get_function("x^2 + y^2");
  std::shared_ptr<dealii::Function<dim>> f2 = get_function("x*y");
  std::shared_ptr<dealii::Function<dim>> f3 = get_function("x^2*y^3");
  const std::vector<std::shared_ptr<dealii::Function<dim>>> functions{f1,
                                                                      f2,
                                                                      f3};

  const std::vector<double> constants{0., 2., 9.};

  for (const double c : constants)
    for (const auto &f : functions)
      {
        // set the overall system to a fixed useless constant
        for (unsigned int i = 0; i < this->dof_handler_cell.n_dofs(); ++i)
          this->current_solution_cell[i] = c;

        // Now set the components related to V (and E) to a specific function
        this->set_component(Ddhdg::Component::V, f, true);

        // Save the current values
        for (unsigned int i = 0; i < this->dof_handler_cell.n_dofs(); ++i)
          all_system_copy[i] = this->current_solution_cell[i];

        // Copy them also in the potential copy
        this->compute_local_charge_neutrality_copy_solution(V_fe,
                                                            V_dof_handler,
                                                            potential_copy);

        // Destroy again the values inside the overall system
        for (unsigned int i = 0; i < this->dof_handler_cell.n_dofs(); ++i)
          this->current_solution_cell[i] = c;

        // Copy back the values from the potential copy
        this->compute_local_charge_neutrality_set_solution(V_fe,
                                                           V_dof_handler,
                                                           potential_copy);

        // Ensure that we get again the same results
        this->current_solution_cell -= all_system_copy;

        EXPECT_LT(this->current_solution_cell.linfty_norm(), TOLERANCE);
      }
}



TEST_F(LocalChargeNeutralityTest, charge_neutrality) // NOLINT
{
  constexpr double TOLERANCE = 0.1;

  this->set_multithreading(false);
  this->refine_grid(4);
  this->set_enabled_components(true, false, false);

  if (!this->initialized)
    this->setup_overall_system();

  this->set_local_charge_neutrality_first_guess();

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
