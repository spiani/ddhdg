#define DEFAULT_PARAMETER_FILE "parameters.prm"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <sys/stat.h>

#include <cmath>
#include <iostream>

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
    : error_table(std::make_shared<Ddhdg::ConvergenceTable>())
  {
    constants.insert({"pi", M_PI});
    constants.insert({"e", std::exp(1.0)});

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
    add_parameter("multithreading",
                  multithreading,
                  "Shall the code run in multithreading mode?");

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
    add_parameter("f",
                  f_str,
                  "The right term of the equation: laplacian(u) = f");
    add_parameter("expected V solution",
                  expected_V_solution_str,
                  "The expected solution for the potential");
    add_parameter("expected n solution",
                  expected_n_solution_str,
                  "The expected solution for the electron density");
  };

  unsigned int initial_refinements = 2;
  unsigned int n_cycles            = 4;

  double left  = -1;
  double right = 1;

  unsigned int V_degree = 1;
  unsigned int n_degree = 1;

  bool multithreading = true;

  std::string f_str = "0.";

  std::shared_ptr<dealii::FunctionParser<2>> f;

  std::string expected_V_solution_str = "0.";
  std::string expected_n_solution_str = "0.";

  std::shared_ptr<dealii::FunctionParser<2>> expected_V_solution;
  std::shared_ptr<dealii::FunctionParser<2>> expected_n_solution;

  std::string V_boundary_function_str = "0.";
  std::string n_boundary_function_str = "0.";

  std::shared_ptr<dealii::FunctionParser<2>> V_boundary_function;
  std::shared_ptr<dealii::FunctionParser<2>> n_boundary_function;

  std::map<std::string, double> constants;

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
    f                   = std::make_shared<dealii::FunctionParser<2>>(1);
    expected_V_solution = std::make_shared<dealii::FunctionParser<2>>(1);
    expected_n_solution = std::make_shared<dealii::FunctionParser<2>>(1);
    V_boundary_function = std::make_shared<dealii::FunctionParser<2>>(1);
    n_boundary_function = std::make_shared<dealii::FunctionParser<2>>(1);

    f->initialize("x, y", f_str, constants);
    expected_V_solution->initialize("x, y", expected_V_solution_str, constants);
    expected_n_solution->initialize("x, y", expected_n_solution_str, constants);
    V_boundary_function->initialize("x, y", V_boundary_function_str, constants);
    n_boundary_function->initialize("x, y", n_boundary_function_str, constants);
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

  // Create an object that represent the problem we are going to solve
  Ddhdg::Problem<2> current_problem(triangulation, boundary_handler, prm.f);

  // Create a solver for the problem
  Ddhdg::Solver<2> solver(current_problem,
                          prm.V_degree,
                          prm.n_degree,
                          prm.multithreading);

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
