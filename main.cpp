#define DEFAULT_PARAMETER_FILE "parameters.prm"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <recombination_term.h>
#include <sys/stat.h>

#include <iostream>

#include "constants.h"
#include "ddhdg.h"

const unsigned int dim = 2;


DeclExceptionMsg(
  TooManyCommandLineArguments,
  "Only one argument is allowed: the path of the parameters file");


bool
file_exists(const std::string &file_path)
{
  struct stat buffer;
  return (stat(file_path.c_str(), &buffer) == 0);
}


class ProblemParameters : public dealii::ParameterHandler
{
public:
  ProblemParameters()
    : error_table(std::make_shared<Ddhdg::ConvergenceTable>(dim))
  {
    enter_subsection("Error table");
    error_table->add_parameters(*this);
    leave_subsection();

    add_parameter("initial refinements",
                  initial_refinements,
                  "How many time the square grid will be refined");

    add_parameter("number of refinement cycles", n_cycles);

    add_parameter(
      "V degree",
      V_degree,
      "The degree of the polynomials used to approximate the potential");
    add_parameter(
      "n degree",
      n_degree,
      "The degree of the polynomials used to approximate the electron density");
    add_parameter(
      "p degree",
      p_degree,
      "The degree of the polynomials used to approximate the hole density");
    add_parameter(
      "V tau",
      V_tau,
      "The value of the stabilization constant for the electric potential");
    add_parameter(
      "n tau",
      n_tau,
      "The value of the stabilization constant for the electron density");
    add_parameter(
      "p tau",
      p_tau,
      "The value of the stabilization constant for the hole density");
    add_parameter("use iterative linear solver",
                  iterative_linear_solver,
                  "Shall the code use an iterative linear solver (GMRES)?");
    add_parameter("multithreading",
                  multithreading,
                  "Shall the code run in multithreading mode?");

    enter_subsection("physical quantities");
    add_parameter("temperature",
                  temperature_str,
                  "A function that defines the temperature on the domain");
    add_parameter("doping",
                  doping_str,
                  "A function that defines the temperature on the domain");
    add_parameter("conduction band density",
                  conduction_band_density,
                  "The conduction band density (Nc) in m^-3",
                  dealii::Patterns::Double());
    add_parameter("valence band density",
                  valence_band_density,
                  "The valence band density (Nv) in m^-3",
                  dealii::Patterns::Double());
    add_parameter("conduction band edge energy",
                  conduction_band_edge_energy,
                  "The conduction band edge energy (Ec) in eV",
                  dealii::Patterns::Double());
    add_parameter("valence band edge energy",
                  valence_band_edge_energy,
                  "The valence band edge energy (Ev) in eV",
                  dealii::Patterns::Double());
    enter_subsection("recombination term");
    add_parameter(
      "zero order term",
      recombination_term_constant_term,
      "A function of the space that represent the value of the recombination"
      "term when n = 0 and p = 0");
    add_parameter(
      "n coefficient",
      recombination_term_n_coefficient,
      "Let the recombination term be R = a + b * n + c * p where "
      "a, b, c are space functions; then this field is the value of b");
    add_parameter(
      "p coefficient",
      recombination_term_p_coefficient,
      "Let the recombination term be R = a + b * n + c * p where "
      "a, b, c are space functions; this this field is the value of c");
    leave_subsection();
    leave_subsection();

    enter_subsection("dimensionality");
    add_parameter("length scale",
                  length_scale,
                  "If length scale is l, two points that on the mesh have "
                  "distance 1 will represent two points that have distance "
                  "l in reality. In this way, it is possible to create "
                  "grids with reasonable size (so, for example, a square with "
                  "side 1) even if we are modelling an object with microscopic "
                  "size (like 10^(-9) meters).",
                  dealii::Patterns::Double());
    add_parameter("doping magnitude",
                  doping_magnitude,
                  "A number with is approximately maximum value that the "
                  "doping function will reach. This is used internally to "
                  "normalize vectors and improve the quality of the results",
                  dealii::Patterns::Double());
    add_parameter("electron mobility magnitude",
                  electron_mobility_magnitude,
                  "A number with is approximately maximum eigenvalue that the "
                  "electron mobility matrix will have. This is used internally "
                  "to normalize vectors and improve the quality of the results",
                  dealii::Patterns::Double());
    leave_subsection();

    enter_subsection("boundary conditions");
    {
      add_parameter(
        "V boundary function",
        V_boundary_function_str,
        "The function that will be used to specify the Dirichlet boundary "
        "conditions for V");
      add_parameter(
        "n boundary function",
        n_boundary_function_str,
        "The function that will be used to specify the Dirichlet boundary "
        "conditions for n");
      add_parameter(
        "p boundary function",
        p_boundary_function_str,
        "The function that will be used to specify the Dirichlet boundary "
        "conditions for p");
    }
    leave_subsection();
    enter_subsection("domain geometry");
    {
      add_parameter("left border", left);
      add_parameter("right border", right);
    }
    leave_subsection();
    enter_subsection("starting points");
    {
      add_parameter("V starting point", V_starting_point_str);
      add_parameter("n starting point", n_starting_point_str);
      add_parameter("p starting point", p_starting_point_str);
    }
    leave_subsection();
    enter_subsection("expected solutions");
    add_parameter("expected V solution",
                  expected_V_solution_str,
                  "The expected solution for the potential");
    add_parameter("expected n solution",
                  expected_n_solution_str,
                  "The expected solution for the electron density");
    add_parameter("expected p solution",
                  expected_p_solution_str,
                  "The expected solution for the hole density");
    leave_subsection();
    enter_subsection("nonlinear solver");
    add_parameter(
      "absolute tolerance",
      nonlinear_solver_absolute_tolerance,
      "If the update is smaller than this value (i.e. in every node "
      "the difference between the old solution and the new one is smaller than "
      "this value) then the iteration stops");
    add_parameter(
      "relative tolerance",
      nonlinear_solver_relative_tolerance,
      "If the ratio between the update and the old solution is smaller "
      "than this value (in every node) then the iteration stops");
    add_parameter("max number of iterations",
                  nonlinear_solver_max_number_of_iterations,
                  "If this number of iterations is reached, the code will stop "
                  "its execution");
    leave_subsection();
  };

  unsigned int initial_refinements = 2;
  unsigned int n_cycles            = 4;

  double left  = -1;
  double right = 1;

  unsigned int V_degree = 1;
  unsigned int n_degree = 1;
  unsigned int p_degree = 1;

  double V_tau = 1.;
  double n_tau = 1.;
  double p_tau = 1.;

  bool iterative_linear_solver = true;
  bool multithreading          = true;

  double nonlinear_solver_absolute_tolerance       = 1e-12;
  double nonlinear_solver_relative_tolerance       = 1e-12;
  int    nonlinear_solver_max_number_of_iterations = 100;

  std::string expected_V_solution_str = "0.";
  std::string expected_n_solution_str = "0.";
  std::string expected_p_solution_str = "0.";

  std::shared_ptr<dealii::FunctionParser<dim>> expected_V_solution;
  std::shared_ptr<dealii::FunctionParser<dim>> expected_n_solution;
  std::shared_ptr<dealii::FunctionParser<dim>> expected_p_solution;

  std::string V_boundary_function_str = "0.";
  std::string n_boundary_function_str = "0.";
  std::string p_boundary_function_str = "0.";

  std::shared_ptr<dealii::FunctionParser<dim>> V_boundary_function;
  std::shared_ptr<dealii::FunctionParser<dim>> n_boundary_function;
  std::shared_ptr<dealii::FunctionParser<dim>> p_boundary_function;

  std::string V_starting_point_str = "0.";
  std::string n_starting_point_str = "0.";
  std::string p_starting_point_str = "0.";

  std::shared_ptr<dealii::FunctionParser<dim>> V_starting_point;
  std::shared_ptr<dealii::FunctionParser<dim>> n_starting_point;
  std::shared_ptr<dealii::FunctionParser<dim>> p_starting_point;

  std::string                                  temperature_str = "25.";
  std::shared_ptr<dealii::FunctionParser<dim>> temperature;

  std::string                                  doping_str = "0.";
  std::shared_ptr<dealii::FunctionParser<dim>> doping;

  std::string recombination_term_constant_term = "0.";
  std::string recombination_term_n_coefficient = "0.";
  std::string recombination_term_p_coefficient = "0.";

  double conduction_band_density     = 4.7e23;
  double valence_band_density        = 9.0e24;
  double conduction_band_edge_energy = 1.424;
  double valence_band_edge_energy    = 0.;

  double length_scale                = 1.;
  double doping_magnitude            = 1.;
  double electron_mobility_magnitude = 1.;

  std::shared_ptr<Ddhdg::ConvergenceTable> error_table;

  void
  read_parameters_file()
  {
    std::cout << "No parameter file submitted from command line. " << std::endl
              << "Looking for a file named " << DEFAULT_PARAMETER_FILE
              << " in the current working dir..." << std::endl;
    if (file_exists(DEFAULT_PARAMETER_FILE))
      {
        std::cout << "File found! Reading it..." << std::endl;
        read_parameters_file(DEFAULT_PARAMETER_FILE);
      }
    else
      {
        std::cout << "File *NOT* found! Using default values!" << std::endl;
        after_reading_operations();
      }
  }

  void
  read_parameters_file(const std::string &parameters_file_path)
  {
    if (parameters_file_path == "-")
      {
        return read_parameters_from_stdin();
      }
    std::cout << "Reading parameter file " << parameters_file_path << std::endl;
    parse_input(parameters_file_path);
    after_reading_operations();
  }

  void
  read_parameters_from_stdin()
  {
    std::cout << "Reading parameter file from standard input.." << std::endl;
    parse_input(std::cin, std::string("standard input"));
    after_reading_operations();
  }

private:
  void
  parse_arguments()
  {
    expected_V_solution = std::make_shared<dealii::FunctionParser<dim>>(1);
    expected_n_solution = std::make_shared<dealii::FunctionParser<dim>>(1);
    expected_p_solution = std::make_shared<dealii::FunctionParser<dim>>(1);
    V_boundary_function = std::make_shared<dealii::FunctionParser<dim>>(1);
    n_boundary_function = std::make_shared<dealii::FunctionParser<dim>>(1);
    p_boundary_function = std::make_shared<dealii::FunctionParser<dim>>(1);
    temperature         = std::make_shared<dealii::FunctionParser<dim>>(1);
    doping              = std::make_shared<dealii::FunctionParser<dim>>(1);
    V_starting_point    = std::make_shared<dealii::FunctionParser<dim>>(1);
    n_starting_point    = std::make_shared<dealii::FunctionParser<dim>>(1);
    p_starting_point    = std::make_shared<dealii::FunctionParser<dim>>(1);

    expected_V_solution->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_V_solution_str,
      Ddhdg::Constants::constants);
    expected_n_solution->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_n_solution_str,
      Ddhdg::Constants::constants);
    expected_p_solution->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_p_solution_str,
      Ddhdg::Constants::constants);
    V_boundary_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      V_boundary_function_str,
      Ddhdg::Constants::constants);
    n_boundary_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      n_boundary_function_str,
      Ddhdg::Constants::constants);
    p_boundary_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      p_boundary_function_str,
      Ddhdg::Constants::constants);
    temperature->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      temperature_str,
      Ddhdg::Constants::constants);
    doping->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                       doping_str,
                       Ddhdg::Constants::constants);
    V_starting_point->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      V_starting_point_str,
      Ddhdg::Constants::constants);
    n_starting_point->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      n_starting_point_str,
      Ddhdg::Constants::constants);
    p_starting_point->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      p_starting_point_str,
      Ddhdg::Constants::constants);
  }

  void
  after_reading_operations()
  {
    parse_arguments();

    // std::ofstream ofile("used_parameters.prm");
    print_parameters(std::cout, dealii::ParameterHandler::ShortText);
  }
};


int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize initialization(argc, argv);
  dealii::deallog.depth_console(2);

  // Read the content of the parameter file
  AssertThrow(argc <= 2, TooManyCommandLineArguments());
  ProblemParameters prm;
  if (argc == 2)
    prm.read_parameters_file(argv[1]);
  else
    prm.read_parameters_file();

  // Create a triangulation
  std::shared_ptr<dealii::Triangulation<dim>> triangulation =
    std::make_shared<dealii::Triangulation<dim>>();

  dealii::GridGenerator::hyper_cube(*triangulation, prm.left, prm.right, true);

  // Set the boundary conditions
  std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>> boundary_handler =
    std::make_shared<Ddhdg::BoundaryConditionHandler<dim>>();

  for (unsigned int i = 0; i < dealii::GeometryInfo<dim>::faces_per_cell; i++)
    {
      boundary_handler->add_boundary_condition(i,
                                               Ddhdg::dirichlet,
                                               Ddhdg::V,
                                               prm.V_boundary_function);
      boundary_handler->add_boundary_condition(i,
                                               Ddhdg::dirichlet,
                                               Ddhdg::n,
                                               prm.n_boundary_function);
      boundary_handler->add_boundary_condition(i,
                                               Ddhdg::dirichlet,
                                               Ddhdg::p,
                                               prm.p_boundary_function);
    }
  // For the time being, we will fix the permittivity (epsilon0) to one
  const std::shared_ptr<const Ddhdg::HomogeneousPermittivity<dim>>
    permittivity =
      std::make_shared<const Ddhdg::HomogeneousPermittivity<dim>>(1.);

  // The same for the electron mobility
  const std::shared_ptr<const Ddhdg::HomogeneousMobility<dim>>
    electron_mobility = std::make_shared<const Ddhdg::HomogeneousMobility<dim>>(
      1., Ddhdg::Component::n);

  // Set the recombination terms
  const std::shared_ptr<const Ddhdg::RecombinationTerm<dim>>
    recombination_term =
      std::make_shared<const Ddhdg::LinearRecombinationTerm<dim>>(
        prm.recombination_term_constant_term,
        prm.recombination_term_n_coefficient,
        prm.recombination_term_p_coefficient);

  // Create an object that represent the problem we are going to solve
  std::shared_ptr<const Ddhdg::HomogeneousProblem<dim>> problem =
    std::make_shared<const Ddhdg::HomogeneousProblem<dim>>(
      triangulation,
      permittivity,
      electron_mobility,
      electron_mobility,
      recombination_term,
      prm.temperature,
      prm.doping,
      boundary_handler,
      prm.conduction_band_density,
      prm.valence_band_density,
      prm.conduction_band_edge_energy,
      prm.valence_band_edge_energy);

  // Choose the parameters for the solver
  std::shared_ptr<Ddhdg::NonlinearSolverParameters>
    nonlinear_solver_parameters =
      std::make_shared<Ddhdg::NonlinearSolverParameters>(
        prm.nonlinear_solver_absolute_tolerance,
        prm.nonlinear_solver_relative_tolerance,
        prm.nonlinear_solver_max_number_of_iterations);

  std::shared_ptr<Ddhdg::FixedTauNPSolverParameters> parameters =
    std::make_shared<Ddhdg::FixedTauNPSolverParameters>(
      prm.V_degree,
      prm.n_degree,
      prm.p_degree,
      nonlinear_solver_parameters,
      prm.V_tau,
      prm.n_tau,
      prm.p_tau,
      prm.iterative_linear_solver,
      prm.multithreading);

  std::shared_ptr<const Ddhdg::Adimensionalizer> adimensionalizer =
    std::make_shared<const Ddhdg::Adimensionalizer>(
      prm.length_scale, prm.doping_magnitude, prm.electron_mobility_magnitude);

  // Create a solver for the problem
  Ddhdg::NPSolver<dim, Ddhdg::HomogeneousProblem<dim>> solver(problem,
                                                              parameters,
                                                              adimensionalizer);

  std::cout << std::endl
            << std::endl
            << "--------------------------------------------------------------"
            << std::endl
            << "STARTING COMPUTATION" << std::endl
            << "--------------------------------------------------------------"
            << std::endl
            << std::endl;
  solver.print_convergence_table(prm.error_table,
                                 prm.expected_V_solution,
                                 prm.expected_n_solution,
                                 prm.expected_p_solution,
                                 prm.V_starting_point,
                                 prm.n_starting_point,
                                 prm.p_starting_point,
                                 prm.n_cycles,
                                 prm.initial_refinements,
                                 std::cout);

  return 0;
}
