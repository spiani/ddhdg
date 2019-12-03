#define DEFAULT_PARAMETER_FILE "parameters.prm"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <recombination_term.h>
#include <sys/stat.h>

#include <iostream>

#include "constants.h"
#include "ddhdg.h"

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
    : error_table(std::make_shared<Ddhdg::ConvergenceTable>(2))
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
    add_parameter("tau", tau, "The value of the stabilization constant");
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
    add_parameter(
      "recombination term zero order term",
      recombination_term_constant_term,
      "A function of the space that represent the value of the recombination term when n = 0");
    add_parameter(
      "recombination term first order term",
      recombination_term_constant_term,
      "A function of the space that represent the linear coefficient that "
      "approximates the dependency of the recombination term from n");
    leave_subsection();

    enter_subsection("boundary conditions");
    {
      add_parameter(
        "V boundary function",
        V_boundary_function_str,
        "The function that will be used to specify the Dirichlet boundary conditions for V");
      add_parameter(
        "n boundary function",
        n_boundary_function_str,
        "The function that will be used to specify the Dirichlet boundary conditions for n");
    }
    leave_subsection();
    enter_subsection("domain geometry");
    {
      add_parameter("left border", left);
      add_parameter("right border", right);
    }
    leave_subsection();
    add_parameter("expected V solution",
                  expected_V_solution_str,
                  "The expected solution for the potential");
    add_parameter("expected n solution",
                  expected_n_solution_str,
                  "The expected solution for the electron density");
    enter_subsection("nonlinear solver");
    add_parameter(
      "tolerance",
      nonlinear_solver_tolerance,
      "If the distance between the current solution and the "
      "previous one (in H1 norm) is smaller than this tolerance, the algorithm "
      "will stop");
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

  double tau = 1.;

  bool iterative_linear_solver = true;
  bool multithreading          = true;

  double nonlinear_solver_tolerance                = 1e-7;
  int    nonlinear_solver_max_number_of_iterations = 100;

  std::string expected_V_solution_str = "0.";
  std::string expected_n_solution_str = "0.";

  std::shared_ptr<dealii::FunctionParser<2>> expected_V_solution;
  std::shared_ptr<dealii::FunctionParser<2>> expected_n_solution;

  std::string V_boundary_function_str = "0.";
  std::string n_boundary_function_str = "0.";

  std::shared_ptr<dealii::FunctionParser<2>> V_boundary_function;
  std::shared_ptr<dealii::FunctionParser<2>> n_boundary_function;

  std::string                                temperature_str = "25.";
  std::shared_ptr<dealii::FunctionParser<2>> temperature;

  std::string recombination_term_constant_term = "0.";
  std::string recombination_term_linear_factor = "0.";

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
    std::cout << "Reading parameter file " << parameters_file_path << std::endl;
    parse_input(parameters_file_path);
    after_reading_operations();
  }

  void
  read_parameters_from_stdin()
  {
    std::cout << "Reading parameter file from standard input.." << std::endl;
    parse_input(std::cin, std::string("File name"));
    after_reading_operations();
  }

private:
  void
  parse_arguments()
  {
    expected_V_solution = std::make_shared<dealii::FunctionParser<2>>(1);
    expected_n_solution = std::make_shared<dealii::FunctionParser<2>>(1);
    V_boundary_function = std::make_shared<dealii::FunctionParser<2>>(1);
    n_boundary_function = std::make_shared<dealii::FunctionParser<2>>(1);
    temperature         = std::make_shared<dealii::FunctionParser<2>>(1);

    expected_V_solution->initialize("x, y",
                                    expected_V_solution_str,
                                    Ddhdg::Constants::constants);
    expected_n_solution->initialize("x, y",
                                    expected_n_solution_str,
                                    Ddhdg::Constants::constants);
    V_boundary_function->initialize("x, y",
                                    V_boundary_function_str,
                                    Ddhdg::Constants::constants);
    n_boundary_function->initialize("x, y",
                                    n_boundary_function_str,
                                    Ddhdg::Constants::constants);
    temperature->initialize("x, y",
                            temperature_str,
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
  ProblemParameters prm;
  prm.read_parameters_file();

  // Create a triangulation
  std::shared_ptr<dealii::Triangulation<2>> triangulation =
    std::make_shared<dealii::Triangulation<2>>();

  dealii::GridGenerator::hyper_cube(*triangulation, prm.left, prm.right, true);

  // Set the boundary conditions
  std::shared_ptr<Ddhdg::BoundaryConditionHandler<2>> boundary_handler =
    std::make_shared<Ddhdg::BoundaryConditionHandler<2>>();

  for (unsigned int i = 0; i < 4; i++)
    {
      boundary_handler->add_boundary_condition(i,
                                               Ddhdg::dirichlet,
                                               Ddhdg::V,
                                               prm.V_boundary_function);
      boundary_handler->add_boundary_condition(i,
                                               Ddhdg::dirichlet,
                                               Ddhdg::n,
                                               prm.n_boundary_function);
    }
  // For the time being, we will fix the permittivity (epsilon0) to one
  const std::shared_ptr<const Ddhdg::Permittivity<2>> permittivity =
    std::make_shared<const Ddhdg::HomogeneousPermittivity<2>>(1.);

  // The same for the electron mobility
  const std::shared_ptr<const Ddhdg::ElectronMobility<2>> electron_mobility =
    std::make_shared<const Ddhdg::HomogeneousElectronMobility<2>>(1.);

  // Set the recombination term
  const std::shared_ptr<const Ddhdg::RecombinationTerm<2>> recombination_term =
    std::make_shared<const Ddhdg::LinearRecombinationTerm<2>>(
      prm.recombination_term_constant_term,
      prm.recombination_term_linear_factor);

  // Create an object that represent the problem we are going to solve
  std::shared_ptr<const Ddhdg::Problem<2>> problem =
    std::make_shared<const Ddhdg::Problem<2>>(triangulation,
                                              permittivity,
                                              electron_mobility,
                                              recombination_term,
                                              prm.temperature,
                                              boundary_handler);

  // Choose the parameters for the solver
  std::shared_ptr<const Ddhdg::SolverParameters> parameters =
    std::make_shared<const Ddhdg::SolverParameters>(
      prm.V_degree,
      prm.n_degree,
      prm.nonlinear_solver_tolerance,
      prm.nonlinear_solver_max_number_of_iterations,
      Ddhdg::VectorTools::H1_norm,
      prm.tau,
      prm.iterative_linear_solver,
      prm.multithreading);

  // Create a solver for the problem
  Ddhdg::Solver<2> solver(problem, parameters);

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
                                 prm.n_cycles,
                                 prm.initial_refinements);

  return 0;
}
