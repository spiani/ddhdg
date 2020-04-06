from inspect import getmembers


class ParameterSubsection:
    def to_prm_file(self, indent=4):
        output = 'subsection {}\n'.format(self.subsection_name)
        for val, item in getmembers(self):
            if val.startswith('_'):
                continue
            if val in ('subsection_name', 'to_prm_file', 'to_dict'):
                continue

            if hasattr(item, "to_prm_file"):
                adding_lines = item.to_prm_file(indent)
                adding_lines = adding_lines.replace('\n', '\n' + ' ' * indent)
                adding_lines = ' ' * indent + adding_lines
                adding_lines = adding_lines[:-indent]
                output += adding_lines
                continue

            val_real_name = val.replace('_', ' ')
            output += ' ' * indent + 'set {} = {}\n'.format(
                val_real_name,
                item
            )
        output += 'end\n'
        return output

    def to_dict(self):
        output = {}
        for val, item in getmembers(self):
            if val.startswith('_'):
                continue
            if val in ('subsection_name', 'to_prm_file', 'to_dict'):
                continue
            output[val] = item
        return output


class NRecombinationTerm(ParameterSubsection):
    subsection_name = "n recombination term"

    def __init__(self, zero_order_term="0.", n_coefficient="0.",
                 p_coefficient="0."):
        self._zero_order_term = zero_order_term
        self._n_coefficient = n_coefficient
        self._p_coefficient = p_coefficient

    @property
    def zero_order_term(self):
        return self._zero_order_term

    @property
    def n_coefficient(self):
        return self._n_coefficient

    @property
    def p_coefficient(self):
        return self._p_coefficient


class PRecombinationTerm(ParameterSubsection):
    subsection_name = "p recombination term"

    def __init__(self, zero_order_term="0.", n_coefficient="0.",
                 p_coefficient="0."):
        self._zero_order_term = zero_order_term
        self._n_coefficient = n_coefficient
        self._p_coefficient = p_coefficient

    @property
    def zero_order_term(self):
        return self._zero_order_term

    @property
    def n_coefficient(self):
        return self._n_coefficient

    @property
    def p_coefficient(self):
        return self._p_coefficient


class PhysicalQuantitiesParameters(ParameterSubsection):
    subsection_name = "physical quantities"

    def __init__(self, n_recombination_term=NRecombinationTerm(),
                 p_recombination_term=PRecombinationTerm(),
                 temperature="q / kb", doping="0."):
        self._n_recombination_term = n_recombination_term
        self._p_recombination_term = p_recombination_term
        self._temperature = temperature
        self._doping = doping

    @property
    def temperature(self):
        return self._temperature

    @property
    def doping(self):
        return self._doping

    @property
    def n_recombination_term(self):
        return self._n_recombination_term

    @property
    def p_recombination_term(self):
        return self._p_recombination_term


class NonlinearSolverParameters(ParameterSubsection):
    subsection_name = "nonlinear solver"

    def __init__(self, iterations=100, absolute_tolerance=1e-9,
                 relative_tolerance=1e-9):
        self._iterations = int(iterations)
        self._absolute_tolerance = absolute_tolerance
        self._relative_tolerance = relative_tolerance

    @property
    def max_number_of_iterations(self):
        return self._iterations

    @property
    def absolute_tolerance(self):
        return self._absolute_tolerance

    @property
    def relative_tolerance(self):
        return self._relative_tolerance


class DomainParameters(ParameterSubsection):
    subsection_name = "domain geometry"

    def __init__(self, left_border=0., right_border=1.):
        self._right = right_border
        self._left = left_border

    @property
    def left_border(self):
        return self._left

    @property
    def right_border(self):
        return self._right


class BoundaryConditionsParameters(ParameterSubsection):
    subsection_name = "boundary conditions"

    def __init__(self, V_boundary_function="0.", n_boundary_function="0.",
                 p_boundary_function="0."):
        self._V = V_boundary_function
        self._n = n_boundary_function
        self._p = p_boundary_function

    @property
    def V_boundary_function(self):
        return self._V

    @property
    def n_boundary_function(self):
        return self._n

    @property
    def p_boundary_function(self):
        return self._p


class StartingPointParameters(ParameterSubsection):
    subsection_name = "starting points"

    def __init__(self, starting_V="0.", starting_n="0.", starting_p="0."):
        self._V = starting_V
        self._n = starting_n
        self._p = starting_p

    @property
    def V_starting_point(self):
        return self._V

    @property
    def n_starting_point(self):
        return self._n

    @property
    def p_starting_point(self):
        return self._p


class ExpectedSolutionsParameters(ParameterSubsection):
    subsection_name = "expected solutions"

    def __init__(self, expected_V_solution="0.", expected_n_solution="0.",
                 expected_p_solution="0."):
        self._V = expected_V_solution
        self._n = expected_n_solution
        self._p = expected_p_solution

    @property
    def expected_V_solution(self):
        return self._V

    @property
    def expected_n_solution(self):
        return self._n

    @property
    def expected_p_solution(self):
        return self._p


class ExecutionParameters:
    def __init__(self, V_degree=1, n_degree=1, p_degree=1,
                 initial_refinements=0, refinements=2,
                 iterative_linear_solver=False, multithreading=True,
                 V_tau=1., n_tau=1., p_tau=1.,
                 nonlinear_solver_parameters=NonlinearSolverParameters(),
                 physical_quantities_parameters=PhysicalQuantitiesParameters(),
                 domain_parameters=DomainParameters(),
                 boundary_conditions_parameters=BoundaryConditionsParameters(),
                 starting_point_parameters=StartingPointParameters(),
                 expected_solutions_parameters=ExpectedSolutionsParameters()
                 ):
        self._V_degree = int(V_degree)
        self._n_degree = int(n_degree)
        self._p_degree = int(p_degree)
        self._initial_refinements = int(initial_refinements)
        self._refinements = int(refinements)

        if iterative_linear_solver is True or iterative_linear_solver is False:
            self._iterative_linear_solver = iterative_linear_solver
        else:
            raise ValueError(
                'Invalid value for iterative_linear_solver: {}'.format(
                    repr(iterative_linear_solver)
                )
            )

        if multithreading is True or multithreading is False:
            self._multithreading = multithreading
        else:
            raise ValueError(
                'Invalid value for multithreading: {}'.format(
                    repr(multithreading)
                )
            )

        self._V_tau = float(V_tau)
        self._n_tau = float(n_tau)
        self._p_tau = float(p_tau)

        self._subsections = []

        self._nonlinear_solver_parameters = nonlinear_solver_parameters
        self._subsections.append(nonlinear_solver_parameters)

        self._physical_quantities_pararmeters = physical_quantities_parameters
        self._subsections.append(physical_quantities_parameters)

        self._domain_parameters = domain_parameters
        self._subsections.append(domain_parameters)

        self._boundary_conditions_parameters = boundary_conditions_parameters
        self._subsections.append(boundary_conditions_parameters)

        self._starting_point_parameters = starting_point_parameters
        self._subsections.append(starting_point_parameters)

        self._expected_solutions_parameters = expected_solutions_parameters
        self._subsections.append(expected_solutions_parameters)

    @property
    def V_degree(self):
        return self._V_degree

    @property
    def n_degree(self):
        return self._n_degree

    @property
    def p_degree(self):
        return self._p_degree

    @property
    def initial_refinements(self):
        return self._initial_refinements

    @property
    def refinements(self):
        return self._refinements

    @property
    def multithreading(self):
        return self._multithreading

    @property
    def iterative_linear_solver(self):
        return self._iterative_linear_solver

    @property
    def V_tau(self):
        return self._V_tau

    @property
    def n_tau(self):
        return self._n_tau

    @property
    def p_tau(self):
        return self._p_tau

    def to_prm_file(self):
        multithreading_str = 'true' if self._multithreading else 'false'
        ils_str = 'true' if self.iterative_linear_solver else 'false'

        output = ''
        output += 'set V degree                    = {}\n'.format(self._V_degree)
        output += 'set n degree                    = {}\n'.format(self._n_degree)
        output += 'set p degree                    = {}\n'.format(self._p_degree)
        output += 'set initial refinements         = {}\n'.format(self._initial_refinements)
        output += 'set number of refinement cycles = {}\n'.format(self._refinements)
        output += 'set use iterative linear solver = {}\n'.format(ils_str)
        output += 'set multithreading              = {}\n'.format(multithreading_str)
        output += 'set V tau                       = {}\n'.format(self.V_tau)
        output += 'set n tau                       = {}\n'.format(self.n_tau)
        output += 'set p tau                       = {}\n'.format(self.p_tau)
        output += '\n'

        for subsection in self._subsections:
            output += subsection.to_prm_file()
            output += '\n'

        return output

    def to_dict(self):
        attributes = (
            "V_degree", "n_degree", "p_degree", "initial_refinements",
            "refinements", "multithreading", "iterative_linear_solver", "V_tau",
            "n_tau", "p_tau"
        )

        output = {attrib: getattr(self, attrib) for attrib in attributes}
        for subsection in self._subsections:
            class_name = subsection.__class__.__name__
            parameter_name = ''
            for char in class_name:
                if char.lower() == char:
                    parameter_name += char
                else:
                    if len(parameter_name) == 0:
                        parameter_name += char.lower()
                    else:
                        parameter_name += '_' + char.lower()
            output[parameter_name] = subsection.to_dict()
        return output

    def __hash__(self):
        dict_self = self.to_dict()
        attributes = []
        for attr_name, attr_val in dict_self.items():
            if isinstance(attr_val, dict):
                dict_hash = hash(tuple(attr_val.items()))
                attributes.append((attr_name, dict_hash))
            else:
                attributes.append((attr_name, hash(attr_val)))

        return hash(hash(tuple(attributes)) * 2147483647)

    def __eq__(self, other):
        if not hasattr(other, "to_dict"):
            return False
        return self.to_dict() == other.to_dict()
