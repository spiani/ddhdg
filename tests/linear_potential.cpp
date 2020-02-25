#include <deal.II/grid/grid_generator.h>

#include <ddhdg.h>
#include <gtest/gtest.h>

TEST(LinearPotential, case1)
{
  const unsigned int dim = 2;

  const std::shared_ptr<dealii::FunctionParser<dim>> zero_function =
    std::make_shared<dealii::FunctionParser<dim>>();
  zero_function->initialize(
    dealii::FunctionParser<dim>::default_variable_names(),
    "0",
    Ddhdg::Constants::constants);

  const std::shared_ptr<dealii::FunctionParser<dim>> expected_solution =
    std::make_shared<dealii::FunctionParser<dim>>();
  expected_solution->initialize(
    dealii::FunctionParser<dim>::default_variable_names(),
    "x",
    Ddhdg::Constants::constants);

  // Prepare the grid
  std::shared_ptr<dealii::Triangulation<dim>> triangulation =
    std::make_shared<dealii::Triangulation<dim>>();

  dealii::GridGenerator::hyper_cube(*triangulation, -1, 1, false);

  // Set the boundary conditions
  std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>> boundary_handler =
    std::make_shared<Ddhdg::BoundaryConditionHandler<dim>>();

  std::shared_ptr<dealii::FunctionParser<dim>> V_boundary_condition =
    std::make_shared<dealii::FunctionParser<dim>>();
  V_boundary_condition->initialize(
    dealii::FunctionParser<dim>::default_variable_names(),
    "x",
    Ddhdg::Constants::constants);

  for (unsigned int i = 0; i < 2; i++)
    {
      boundary_handler->add_boundary_condition(i,
                                               Ddhdg::dirichlet,
                                               Ddhdg::V,
                                               V_boundary_condition);
      boundary_handler->add_boundary_condition(i,
                                               Ddhdg::dirichlet,
                                               Ddhdg::n,
                                               zero_function);
    }

  const std::shared_ptr<const Ddhdg::Permittivity<dim>> permittivity =
    std::make_shared<const Ddhdg::HomogeneousPermittivity<dim>>(1.);
  const std::shared_ptr<const Ddhdg::ElectronMobility<dim>> electron_mobility =
    std::make_shared<const Ddhdg::HomogeneousElectronMobility<dim>>(1.);
  const std::shared_ptr<const Ddhdg::RecombinationTerm<dim>>
    recombination_term =
      std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>("0", "0");
  const std::shared_ptr<dealii::FunctionParser<dim>> temperature =
    std::make_shared<dealii::FunctionParser<dim>>();
  temperature->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                          "q / kb",
                          Ddhdg::Constants::constants);
  const std::shared_ptr<dealii::FunctionParser<dim>> &doping = zero_function;

  std::shared_ptr<Ddhdg::Problem<dim>> problem =
    std::make_shared<Ddhdg::Problem<dim>>(triangulation,
                                          permittivity,
                                          electron_mobility,
                                          recombination_term,
                                          temperature,
                                          doping,
                                          boundary_handler);

  Ddhdg::Solver<dim> solver(problem);
  solver.set_multithreading(false);

  solver.refine_grid(2);
  solver.set_n_component(zero_function);

  const Ddhdg::NonlinearIteratorStatus status               = solver.run();
  const unsigned int                   number_of_iterations = status.iterations;

  EXPECT_LE(number_of_iterations, 3);

  const double V_l2_error =
    solver.estimate_l2_error(expected_solution, Ddhdg::Component::V);
  const double n_l2_error =
    solver.estimate_l2_error(zero_function, Ddhdg::Component::n);

  EXPECT_LT(V_l2_error, 1e-10);
  EXPECT_LT(n_l2_error, 1e-10);
}
