#!/usr/bin/env python3

from parameter_file import *
from run_ddhdg import run_ddhdg

V_FUNCTION = '-2 * log(x)'
N_FUNCTION = '-2 / x^2'


def test_convergence_table(degree, refinements):
    nonlinear_solver_parameters = NonlinearSolverParameters(
        iterations = 200,
        tolerance = 1e-10
    )

    physical_quantities_parameters = PhysicalQuantitiesParameters()

    domain_parameters = DomainParameters(left_border=1., right_border=2.)

    boundary_conditions_parameters = BoundaryConditionsParameters(
        V_boundary_function = V_FUNCTION,
        n_boundary_function = N_FUNCTION
    )

    starting_point_parameters = StartingPointParameters(
        starting_V = "0",
        starting_n = "0"
    )

    expected_solutions_parameters = ExpectedSolutionsParameters(
        expected_V_solution = V_FUNCTION,
        expected_n_solution = N_FUNCTION
    )

    execution_parameters = ExecutionParameters(
        V_degree = degree,
        n_degree = degree,
        refinements = refinements,
        nonlinear_solver_parameters = nonlinear_solver_parameters,
        physical_quantities_parameters = physical_quantities_parameters,
        domain_parameters = domain_parameters,
        boundary_conditions_parameters = boundary_conditions_parameters,
        starting_point_parameters = starting_point_parameters,
        expected_solutions_parameters = expected_solutions_parameters
    )

    run_ddhdg(execution_parameters)


if __name__ == '__main__':
    refinements = -1
    for i in range(6):
        if i < 2:
            refinements = 10
        elif i < 4:
             refinements = 8
        else:
            refinements = 6
        test_convergence_table(i, refinements)
