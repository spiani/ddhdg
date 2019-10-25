#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

#include <iostream>
#include <sys/stat.h>

#include "ddhdg.h"

#define DEFAULT_PARAMETER_FILE "parameters.prm"

template<int dim>
class F
        : public dealii::Function<dim> {
public:

    double value(const dealii::Point<dim>& p,
            const unsigned int component = 0) const override
    {
        (void) p;
        (void) component;
        return 2.;
    }

    dealii::Tensor<1, dim>
    gradient(const dealii::Point<dim>& p,
            const unsigned int component = 0) const override
    {
        (void) p;
        (void) component;
        dealii::Tensor<1, dim> zeros;
        for (int i = 0; i<dim; i++)
            zeros[i] = 0.;
        return zeros;
    }
};

class BoundaryFunction
        : public dealii::Function<2> {
public:

    BoundaryFunction()
            :dealii::Function<2>(2) { }

    double value(const dealii::Point<2>& p,
            const unsigned int component = 0) const override
    {
        if (component==0) {
            double x = p[0];
            double y = p[1];
            return x*x-x+y;
        }
        return 0.;
    }

    dealii::Tensor<1, 2>
    gradient(const dealii::Point<2>& p,
            const unsigned int component = 0) const override
    {
        (void) component;
        double x = p[0];
        dealii::Tensor<1, 2> grdt;
        grdt[0] = 2*x-1;
        grdt[1] = 1.;
        return grdt;
    }
};

bool file_exists(const std::string& file_path)
{
    struct stat buffer;
    return (stat(file_path.c_str(), &buffer)==0);
}

void read_parameters(dealii::ParameterHandler& prm, int argc, char const* const argv[])
{
    prm.declare_entry("number of refinements",
            "0",
            dealii::Patterns::Integer(),
            "How many time the square grid will be refined");
    prm.declare_entry("V degree",
            "2",
            dealii::Patterns::Integer(),
            "The degree of the polynomials used to approximate the potential");
    prm.declare_entry("n degree",
            "2",
            dealii::Patterns::Integer(),
            "The degree of the polynomials used to approximate the electron density");
    prm.declare_entry("multithreading",
            "true",
            dealii::Patterns::Bool(),
            "Shall the code run in multithreading mode?");

    // Check where is the parameter file
    if (argc==1) {
        std::cout << "No parameter file submitted from command line. "
                  << std::endl
                  << "Looking for a file named "
                  << DEFAULT_PARAMETER_FILE
                  << " in the current working dir..."
                  << std::endl;
        if(file_exists(DEFAULT_PARAMETER_FILE)) {
            std::cout << "File found! Reading it..." << std::endl;
            prm.parse_input(DEFAULT_PARAMETER_FILE);
        } else {
            std::cout << "File *NOT* found! Using default values!" << std::endl;
        }
    }
    else {
        std::string parameter_file_path(argv[1]);
        if (parameter_file_path!="-") {
            std::cout << "Reading parameter file " << parameter_file_path << std::endl;
            prm.parse_input(parameter_file_path);
        } else {
            std::cout << "Reading parameter file from standard input.." << std::endl;
            prm.parse_input(std::cin, std::string("File name"));
        }
    }
}

int main(int argc, char const* const argv[])
{
    dealii::deallog.depth_console(2);

    // Read the content of the parameter file
    dealii::ParameterHandler prm;
    read_parameters(prm, argc, argv);

    // Create a triangulation
    std::shared_ptr<dealii::Triangulation<2>> triangulation =
            std::make_shared<dealii::Triangulation<2>>();
    dealii::GridGenerator::hyper_cube(*triangulation, -1., 1., true);

    // Refine it
    const unsigned int refine_times = prm.get_integer("number of refinements");
    if (refine_times>0)
        triangulation->refine_global(refine_times);

    // Set the boundary conditions
    std::shared_ptr<const dealii::Function<2>> boundary_function =
            std::make_shared<const BoundaryFunction>();

    std::shared_ptr<Ddhdg::BoundaryConditionHandler<2>> boundary_handler =
            std::make_shared<Ddhdg::BoundaryConditionHandler<2>>();

    for (unsigned int i = 0; i<4; i++) {
        boundary_handler->add_boundary_condition(i, Ddhdg::dirichlet, Ddhdg::V, boundary_function);
        boundary_handler->add_boundary_condition(i, Ddhdg::dirichlet, Ddhdg::n, boundary_function);
    }

    // Set the function f for the equation laplacian(u) = f
    std::shared_ptr<dealii::Function<2>> f = std::make_shared<F<2>>();

    // Read the degree of the polynomial spaces
    const int V_degree = prm.get_integer("V degree");
    const int n_degree = prm.get_integer("n degree");

    // Create the main problem that must be solved
    Ddhdg::Problem<2> current_problem(triangulation, boundary_handler, f);

    prm.print_parameters(std::cout, dealii::ParameterHandler::Text);

    try {
        std::cout << std::endl
                  << "============================================="
                  << std::endl
                  << "Solving the problem"
                  << std::endl
                  << "============================================="
                  << std::endl
                  << std::endl;
        Ddhdg::Solver<2> solver(current_problem, V_degree, n_degree);
        solver.run(prm.get_bool("multithreading"));
        solver.output_results(
                "solution.vtk",
                "trace.vtk"
        );
        std::cout << std::endl;
    }
    catch (std::exception& exc) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}
