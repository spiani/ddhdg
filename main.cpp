#define DEFAULT_PARAMETER_FILE "parameters.prm"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_convergence_table.h>

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
    : error_table({"E", "E", "V", "W", "W", "n"},
                  {{},
                   {dealii::VectorTools::H1_norm, dealii::VectorTools::L2_norm},
                   {},
                   {dealii::VectorTools::H1_norm,
                    dealii::VectorTools::L2_norm}}){};

  unsigned int initial_refinements = 2;
  unsigned int n_cycles            = 4;

  double left  = -1;
  double right = 1;

  dealii::ParsedConvergenceTable error_table;

  void
  read_parameters(int argc, char const *const argv[])
  {
    enter_subsection("Error table");
    error_table.add_parameters(*this);
    leave_subsection();

    add_parameter("number of refinements",
                  initial_refinements,
                  "How many time the square grid will be refined");

    add_parameter("number of refinement cycles", n_cycles);

    declare_entry(
      "V degree",
      "2",
      dealii::Patterns::Integer(),
      "The degree of the polynomials used to approximate the potential");
    declare_entry("n degree",
                  "2",
                  dealii::Patterns::Integer(),
                  "The degree of the polynomials used to approximate the "
                  "electron density");
    declare_entry("multithreading",
                  "true",
                  dealii::Patterns::Bool(),
                  "Shall the code run in multithreading mode?");
    enter_subsection("boundary conditions");
    {
      declare_entry("V boundary function",
                    "0",
                    dealii::Patterns::Anything(),
                    "The function that will be used to specify the Dirichlet "
                    "boundary conditions for V");
      declare_entry("n boundary function",
                    "0",
                    dealii::Patterns::Anything(),
                    "The function that will be used to specify the Dirichlet "
                    "boundary conditions for n");
    }
    leave_subsection();
    enter_subsection("domain geometry");
    {
      add_parameter("left border", left);
      add_parameter("right border", right);
    }
    leave_subsection();
    declare_entry("f",
                  "0",
                  dealii::Patterns::Anything(),
                  "The right term of the equation: laplacian(u) = f");
    declare_entry("expected V solution",
                  "0",
                  dealii::Patterns::Anything(),
                  "The expected solution for the potential");
    declare_entry("expected n solution",
                  "0",
                  dealii::Patterns::Anything(),
                  "The expected solution for the electron density");

    // Check where is the parameter file
    if (argc == 1)
      {
        std::cout << "No parameter file submitted from command line. "
                  << std::endl
                  << "Looking for a file named " << DEFAULT_PARAMETER_FILE
                  << " in the current working dir..." << std::endl;
        if (file_exists(DEFAULT_PARAMETER_FILE))
          {
            std::cout << "File found! Reading it..." << std::endl;
            parse_input(DEFAULT_PARAMETER_FILE);
          }
        else
          {
            std::cout << "File *NOT* found! Using default values!" << std::endl;
          }
      }
    else
      {
        std::string parameter_file_path(argv[1]);
        if (parameter_file_path != "-")
          {
            std::cout << "Reading parameter file " << parameter_file_path
                      << std::endl;
            parse_input(parameter_file_path);
          }
        else
          {
            std::cout << "Reading parameter file from standard input.."
                      << std::endl;
            parse_input(std::cin, std::string("File name"));
          }
      }
    std::ofstream ofile("used_parameters.prm");
    print_parameters(ofile, dealii::ParameterHandler::ShortText);
  }
};

int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize initialization(argc, argv);
  dealii::deallog.depth_console(2);

  // Prepare the constants that will be used when reading the functions
  std::map<std::string, double> constants;
  constants.insert({"pi", M_PI});
  constants.insert({"e", std::exp(1.0)});

  // Read the content of the parameter file
  ProblemParameters prm;
  prm.read_parameters(argc, argv);

  // Set the boundary conditions
  std::vector<std::string> boundary_conditions = {"boundary conditions"};
  std::shared_ptr<dealii::FunctionParser<2>> V_boundary_function =
    std::make_shared<dealii::FunctionParser<2>>(1);
  std::string V_boundary_description =
    prm.get(boundary_conditions, "V boundary function");
  V_boundary_function->initialize("x, y", V_boundary_description, constants);

  std::shared_ptr<dealii::FunctionParser<2>> n_boundary_function =
    std::make_shared<dealii::FunctionParser<2>>(1);
  std::string n_boundary_description =
    prm.get(boundary_conditions, "n boundary function");
  n_boundary_function->initialize("x, y", n_boundary_description, constants);

  std::shared_ptr<Ddhdg::BoundaryConditionHandler<2>> boundary_handler =
    std::make_shared<Ddhdg::BoundaryConditionHandler<2>>();

  for (unsigned int i = 0; i < 4; i++)
    {
      boundary_handler->add_boundary_condition(i,
                                               Ddhdg::dirichlet,
                                               Ddhdg::V,
                                               V_boundary_function);
      boundary_handler->add_boundary_condition(i,
                                               Ddhdg::dirichlet,
                                               Ddhdg::n,
                                               n_boundary_function);
    }

  // Set the function f for the equation laplacian(u) = f
  std::shared_ptr<dealii::FunctionParser<2>> f =
    std::make_shared<dealii::FunctionParser<2>>(1);
  std::string f_description = prm.get("f");
  f->initialize("x, y", f_description, constants);

  // Read the degree of the polynomial spaces
  const int V_degree = prm.get_integer("V degree");
  const int n_degree = prm.get_integer("n degree");

  // Create a triangulation
  std::shared_ptr<dealii::Triangulation<2>> triangulation =
    std::make_shared<dealii::Triangulation<2>>();

  dealii::GridGenerator::hyper_cube(*triangulation, prm.left, prm.right, true);

  triangulation->refine_global(prm.initial_refinements);

  std::shared_ptr<dealii::FunctionParser<2>> expected_V_solution =
    std::make_shared<dealii::FunctionParser<2>>();
  std::shared_ptr<dealii::FunctionParser<2>> expected_n_solution =
    std::make_shared<dealii::FunctionParser<2>>();
  std::string expected_V_solution_description = prm.get("expected V solution");
  std::string expected_n_solution_description = prm.get("expected n solution");
  expected_V_solution->initialize("x, y",
                                  expected_V_solution_description,
                                  constants);
  expected_n_solution->initialize("x, y",
                                  expected_n_solution_description,
                                  constants);

  dealii::FunctionParser<2> exact_solution(6);
  exact_solution.initialize("x,y",
                            "0;0;" + expected_V_solution_description + ";0;0;" +
                              expected_n_solution_description,
                            constants);

  prm.print_parameters(std::cout, dealii::ParameterHandler::Text);

  for (unsigned int cycle = 0; cycle < prm.n_cycles; ++cycle)
    {
      // Create the main problem that must be solved
      Ddhdg::Problem<2> current_problem(triangulation, boundary_handler, f);
      Ddhdg::Solver<2>  solver(current_problem, V_degree, n_degree);
      solver.run(prm.get_bool("multithreading"));

      prm.error_table.error_from_exact(solver.dof_handler_local,
                                       solver.solution_local,
                                       exact_solution);

      solver.output_results("solution_" + std::to_string(cycle) + ".vtk",
                            "trace_" + std::to_string(cycle) + ".vtk");
      triangulation->refine_global(1);
    }
  prm.error_table.output_table(std::cout);

  return 0;
}
