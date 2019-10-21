#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/base/exceptions.h>

#include "ddhdg.h"

#include <iostream>

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

int main()
{
    dealii::deallog.depth_console(2);

    // Create a triangulation
    std::shared_ptr<dealii::Triangulation<2>> triangulation =
            std::make_shared<dealii::Triangulation<2>>();
    dealii::GridGenerator::hyper_cube(*triangulation, -1., 1., true);
    triangulation->refine_global(2);


    // Set the boundary conditions
    std::shared_ptr<const dealii::Function<2>> boundary_function =
            std::make_shared<const BoundaryFunction>();

    std::shared_ptr<Ddhdg::BoundaryConditionHandler<2>> boundary_handler =
            std::make_shared<Ddhdg::BoundaryConditionHandler<2>>();

    for(unsigned int i=0; i<4; i++) {
        boundary_handler->add_boundary_condition(i, Ddhdg::dirichlet, Ddhdg::V, boundary_function);
        boundary_handler->add_boundary_condition(i, Ddhdg::dirichlet, Ddhdg::n, boundary_function);
    }

    // Set the function f for the equation laplacian(u) = f
    std::shared_ptr<dealii::Function<2>> f = std::make_shared<F<2>>();

    // Create the main problem that must be solved
    Ddhdg::Problem<2> current_problem(triangulation, boundary_handler, f);

    try {
        std::cout << "Solving with Q3 elements"
                  << std::endl
                  << "============================================="
                  << std::endl
                  << std::endl;
        Ddhdg::Solver<2> solver(current_problem, 7, 5);
        solver.run();
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
