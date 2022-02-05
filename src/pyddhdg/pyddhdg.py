from importlib import import_module
import logging
import numpy as np
import warnings

import pyddhdg.pyddhdg_common


# We import the modules for the three dimensions and we put them
# in a dictionary
_MODULE_PER_DIM = {
    1: import_module('pyddhdg.pyddhdg1d.pyddhdg1d'),
    2: import_module('pyddhdg.pyddhdg2d.pyddhdg2d'),
    3: import_module('pyddhdg.pyddhdg3d.pyddhdg3d')
}

LOGGER = logging.getLogger(__name__)

# Optional modules
MATPLOTLIB_IMPORTED = True
try:
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    from matplotlib.colors import Colormap
except ModuleNotFoundError:
    MATPLOTLIB_IMPORTED = False
    warnings.warn(
        "Module matplotlib not found! Some features will be disabled"
    )
    LOGGER.debug("Error while importing matplotlib!", exc_info=True)

SCIPY_IMPORTED = True
try:
    from scipy.interpolate import barycentric_interpolate
except ModuleNotFoundError:
    SCIPY_IMPORTED = False
    warnings.warn("Module scipy not found! Some features will be disabled")
    LOGGER.debug("Error while importing scipy!", exc_info=True)

# These constants are useful only if matplotlib is imported
if MATPLOTLIB_IMPORTED:
    CMAP = get_cmap('plasma')
else:
    CMAP = None

GRID_LINEWIDTH = 0.1
GRID_LINESTYLE = "dotted"
EPS = 0.01


class InvalidDimensionException(ValueError):
    pass


class InvalidTypenameException(ValueError):
    pass


# Make the constants available in the current namespace in a nice way
class ConstantsNamespace:
    def __init__(self):
        self._data = {
            'q': pyddhdg_common.CONSTANT_Q,
            'kB': pyddhdg_common.CONSTANT_KB,
            'eps0': pyddhdg_common.CONSTANT_EPS0,
            'epsilon0': pyddhdg_common.CONSTANT_EPS0,
        }

    def __getattr__(self, attr):
        if attr in self._data:
            return self._data[attr]
        raise AttributeError(
            '"{}" is not a valid constant'.format(attr)
        )

    def __getitem__(self, item):
        return self._data[item]

    def as_dict(self):
        return self._data.copy()


Constants = ConstantsNamespace()


class ComponentsNamespace:
    def __init__(self):
        self._data = {
            'v': pyddhdg_common.Component.v,
            'n': pyddhdg_common.Component.n,
            'p': pyddhdg_common.Component.p,
            'phi_n': pyddhdg_common.Component.phi_n,
            'phi_p': pyddhdg_common.Component.phi_p
        }

    def __getattr__(self, attr):
        if attr in self._data:
            return self._data[attr]
        raise AttributeError(
            '"{}" is not a valid component'.format(attr)
        )

    def __getitem__(self, item):
        return self._data[item]

    def as_dict(self):
        return self._data.copy()

    def principal_components(self):
        v = self._data['v']
        n = self._data['n']
        p = self._data['p']
        return v, n, p

    def __iter__(self):
        return self.principal_components().__iter__()


Components = ComponentsNamespace()

# Export also the components and the others enums; in this way we avoid to
# pollute the global namespace with the name of the single elements of the
# enums (i.e. it will be necessary to write Displacement.E and not simply E)
Displacements = pyddhdg_common.Displacement
BoundaryConditionType = pyddhdg_common.BoundaryConditionType
DDFluxType = pyddhdg_common.DDFluxType

# Now, we also import the common classes from the pyddhdg_common module
ErrorPerCell = pyddhdg_common.ErrorPerCell
NonlinearSolverParameters = pyddhdg_common.NonlinearSolverParameters
Adimensionalizer = pyddhdg_common.Adimensionalizer
NonlinearIterationResults = pyddhdg_common.NonlinearIterationResults


class TemplateClass:
    def __init__(self, class_name, bind_methods=None):
        if bind_methods is None:
            bind_methods = []

        self._name = class_name

        self._bind_methods = {}
        for mthd in bind_methods:
            mthd_name = mthd.__name__
            while mthd_name.startswith('_'):
                mthd_name = mthd_name[1:]
            self._bind_methods[mthd_name] = mthd

        self._reference_class = {}
        for i in range(1, 4):
            current_class = getattr(_MODULE_PER_DIM[i], class_name)
            if len(self._bind_methods) == 0:
                self._reference_class[i] = current_class
                continue

            # If we need to bind methods, we dinamycally create a wrapper class
            class_wrapper = type(
                class_name + "_wrapper",
                (current_class,),
                self._bind_methods
            )
            self._reference_class[i] = class_wrapper

    def __getattr__(self, attr):
        raise AttributeError(
            'The class {} is a TemplateClass. Specify its dimension with {}'
            '[d] (where d is a integer number between 1 and 3) before calling '
            'a method on it'.format(self._name, self._name)
        )

    def __call__(self, *args, **kwargs):
        raise AttributeError(
            'The class {} is a TemplateClass. Specify its dimension with {}'
            '[d](*args) (where d is a integer number between 1 and 3) to '
            'initialize a object of this class'.format(self._name, self._name)
        )

    def __getitem__(self, dimension):
        if dimension not in (1, 2, 3):
            raise InvalidDimensionException(
                'The only valid dimensions accepted are 1, 2, or 3. Received '
                '{}'.format(dimension)
            )

        return self._reference_class[dimension]


class _NPSolverParametersTemplate:
    _error_message = 'The class NPSolverParameters is a TemplateClass over '
    'a typename. Specify this typename with the syntax '
    'NPSolverParameters[typename] where typename is a string that can be '
    'chosen among: "FixedTau", "CellFaceTau"'

    def __init__(self):
        self._reference_class = {
            "FixedTau": pyddhdg_common.FixedTauNPSolverParameters,
            "CellFaceTau": pyddhdg_common.CellFaceTauNPSolverParameters
        }

    def __getattr__(self, attr):
        raise AttributeError(_NPSolverParametersTemplate._error_message)

    def __call__(self, *args, **kwargs):
        AttributeError(_NPSolverParametersTemplate._error_message)

    def __getitem__(self, typename):
        if typename not in self._reference_class:
            raise InvalidTypenameException(
                'The only valid typename accepted are: {}'.format(
                    ', '.join(['"' + k + '"' for k in self._reference_class])
                )
            )

        return self._reference_class[typename]


NPSolverParameters = _NPSolverParametersTemplate()


def _get_points_to_evaluate(all_cell_vertices, degree, to_be_interp=True):
    cells = all_cell_vertices.shape[0]
    interpolation_points = degree + 1
    plot_points = 2 if degree < 2 else max(2, 1000 // cells)

    points = interpolation_points if to_be_interp else plot_points

    cell_left_boundaries = all_cell_vertices[:, 0, :].flatten()
    cell_order = np.argsort(cell_left_boundaries)

    cell_mins = np.empty((cells,), dtype=np.float64)
    cell_maxs = np.empty((cells,), dtype=np.float64)

    evaluation_points = np.empty((cells * points,), dtype=np.float64)

    if to_be_interp:
        cheb_points = np.array(
            [np.cos((2 * k + 1) / (2 * points) * np.pi) for k in range(points)],
            dtype=np.float64
        )

        # Rescale into [0, 1]
        cheb_points = 0.5 + cheb_points / 2.

    for cell in range(cells):
        cell_vertices = all_cell_vertices[cell_order[cell]]
        cell_x_min = np.min(cell_vertices)
        cell_x_max = np.max(cell_vertices)

        current_index = cell * points

        # Interpolate a polynomial on the current cell
        if to_be_interp:
            cell_points = cell_x_min + cheb_points * (cell_x_max - cell_x_min)
        else:
            eps = (cell_x_max - cell_x_min) * EPS
            cell_points = np.linspace(
                cell_x_min + eps,
                cell_x_max - eps,
                points
            )
        evaluation_points[current_index: current_index + points] = \
            cell_points[:]

        cell_mins[cell] = cell_x_min
        cell_maxs[cell] = cell_x_max

    return points, cell_mins, cell_maxs, evaluation_points


# Before initializing the template classes, we introduce the methods that will
# be bound to some of them
def _plot_solution(solver, component, plot_grid=False, ax=None, colors=None,
                   linewidth=1, grid_color=(.8, .8, .8)):
    if not MATPLOTLIB_IMPORTED:
        raise ModuleNotFoundError('The method "plot_solution" requires matplotlib')
    if not SCIPY_IMPORTED:
        raise ModuleNotFoundError('The method "plot_solution" requires scipy')

    dim = solver.dimension
    if dim != 1:
        raise ValueError('No plot available in {}D'.format(dim))

    if ax is None:
        ax = plt.gca()
    if colors is None:
        colors = CMAP

    # For phi_n and phi_p, the degree is the max degree between V (the
    # potential) and n (or p, if we are using phi_p)
    if component in Components.principal_components():
        degree = solver.get_parameters().degree(component)
    else:
        if component == Components.phi_n:
            prnc_component = Components.n
        elif component == Components.phi_p:
            prnc_component = Components.p
        else:
            raise ValueError('Invalid component: {}'.format(component))

        v_degree = solver.get_parameters().degree(Components.v)
        prnc_degree = solver.get_parameters().degree(prnc_component)
        degree = max(v_degree, prnc_degree)

    cells = solver.n_active_cells
    points = degree + 1

    plot_points = 2 if degree < 2 else max(2, 1000 // cells)

    all_cell_vertices = solver.get_cell_vertices()

    domain_x_min = np.min(all_cell_vertices)
    domain_x_max = np.max(all_cell_vertices)
    domain_range = domain_x_max - domain_x_min

    if plot_grid is True:
        # Plot the horizontal line of the grid
        ax.plot(
            [domain_x_min, domain_x_max],
            [0, 0],
            color=grid_color,
            linewidth=GRID_LINEWIDTH,
            linestyle=GRID_LINESTYLE
        )

        # Plot the last vertical line
        p = pyddhdg.Point[dim](domain_x_max)
        y_last = solver.get_solution_on_a_point(p, component)
        ax.plot(
            [domain_x_max, domain_x_max],
            [0, y_last],
            color=grid_color,
            linewidth=GRID_LINEWIDTH,
            linestyle=GRID_LINESTYLE
        )

        # We initialize this value that remembers the height of the last
        # point of the previous cell
        y_previous_cell = 0

    cell_left_boundaries = all_cell_vertices[:, 0, :].flatten()
    cell_order = np.argsort(cell_left_boundaries)

    interpolate = component in Components.principal_components()

    points_per_cell, cell_x_mins, cell_x_maxs, eval_points = \
        _get_points_to_evaluate(
            all_cell_vertices,
            degree,
            interpolate
        )

    for cell in range(cells):
        # Interpolate a polynomial on the current cell
        p_start = cell * points_per_cell
        p_end = (cell + 1) * points_per_cell
        x_points = eval_points[p_start:p_end]
        y_points = np.empty_like(x_points)
        for i in range(points_per_cell):
            p = pyddhdg.Point[dim](x_points[i])
            y_points[i] = solver.get_solution_on_a_point(p, component)

        if interpolate:
            cell_x_min = cell_x_mins[cell]
            cell_x_max = cell_x_maxs[cell]
            x_plot_points = np.linspace(cell_x_min, cell_x_max, plot_points)
            y_plot_points = barycentric_interpolate(x_points, y_points, x_plot_points)
        else:
            x_plot_points = x_points
            y_plot_points = y_points

        # Choose the color
        if isinstance(colors, Colormap):
            if cells == 1:
                color = CMAP(0.5)
            else:
                cell_mean = (cell_x_max + cell_x_min) * 0.5
                color = CMAP((cell_mean - domain_x_min) / domain_range)
        elif isinstance(colors, list) or isinstance(colors, tuple):
            color = colors[cell % len(colors)]
        else:
            color = colors

        ax.plot(x_plot_points, y_plot_points, color=color, linewidth=linewidth)

        if plot_grid is True:
            y1 = min([0, y_previous_cell, y_plot_points[0]])
            y2 = max([0, y_previous_cell, y_plot_points[0]])
            ax.plot(
                [cell_x_min, cell_x_min],
                [y1, y2],
                color=grid_color,
                linewidth=GRID_LINEWIDTH,
                linestyle=GRID_LINESTYLE
            )
            y_previous_cell = y_plot_points[-1]


# Before initializing the template classes, we introduce the methods that will
# be bound to some of them
def _plot_value_per_cell(solver, value_per_cell, ax=None, colors=None,
                         linewidth=1):
    if not MATPLOTLIB_IMPORTED:
        raise ModuleNotFoundError('This method requires matplotlib')

    dim = solver.dimension
    if dim != 1:
        raise ValueError('No plot available in {}D'.format(dim))

    if ax is None:
        ax = plt.gca()
    if colors is None:
        colors = CMAP

    cells = solver.n_active_cells
    all_cell_vertices = solver.get_cell_vertices()

    domain_x_min = np.min(all_cell_vertices)
    domain_x_max = np.max(all_cell_vertices)
    domain_range = domain_x_max - domain_x_min

    cell_left_boundaries = all_cell_vertices[:, 0, :].flatten()
    cell_order = np.argsort(cell_left_boundaries)

    for cell in range(cells):
        cell_vertices = all_cell_vertices[cell_order[cell]]
        cell_x_min = np.min(cell_vertices)
        cell_x_max = np.max(cell_vertices)
        v = value_per_cell[cell_order[cell]]

        # Choose the color
        if isinstance(colors, Colormap):
            if cells == 1:
                color = CMAP(0.5)
            else:
                cell_mean = (cell_x_max + cell_x_min) * 0.5
                color = CMAP((cell_mean - domain_x_min) / domain_range)
        elif isinstance(colors, list) or isinstance(colors, tuple):
            color = colors[cell % len(colors)]
        else:
            color = colors

        ax.plot(
            (cell_x_min, cell_x_max),
            (v, v),
            color=color,
            linewidth=linewidth
        )


def _plot_error_per_cell(solver, component, ax=None, colors=None, linewidth=1):
    v_per_cell = solver.estimate_error_per_cell(component).as_numpy_array()
    return _plot_value_per_cell(
        solver,
        v_per_cell,
        ax=ax,
        colors=colors,
        linewidth=linewidth
    )


def _plot_l2_error_per_cell(solver, expected_solution, component, ax=None,
                            colors=None, linewidth=1):
    v_per_cell = solver.estimate_l2_error_per_cell(
        expected_solution,
        component
    )
    return _plot_value_per_cell(
        solver,
        v_per_cell.as_numpy_array(),
        ax=ax,
        colors=colors,
        linewidth=linewidth
    )


def _plot_h1_error_per_cell(solver, expected_solution, component, ax=None,
                            colors=None, linewidth=1):
    v_per_cell = solver.estimate_h1_error_per_cell(
        expected_solution,
        component
    )
    return _plot_value_per_cell(
        solver,
        v_per_cell.as_numpy_array(),
        ax=ax,
        colors=colors,
        linewidth=linewidth
    )


def _plot_linfty_error_per_cell(solver, expected_solution, component, ax=None,
                                colors=None, linewidth=1):
    v_per_cell = solver.estimate_linfty_error_per_cell(
        expected_solution,
        component
    )
    return _plot_value_per_cell(
        solver,
        v_per_cell.as_numpy_array(),
        ax=ax,
        colors=colors,
        linewidth=linewidth
    )


def _plot_solution_on_trace(solver, component, ax=None, color=None,
                            marker='.', linestyle=''):
    if not MATPLOTLIB_IMPORTED:
        raise ModuleNotFoundError(
            'The method "plot_solution_on_trace" requires matplotlib'
        )

    dim = solver.dimension
    if dim != 1:
        raise ValueError('No plot available in {}D'.format(dim))

    if ax is None:
        ax = plt.gca()

    x_values, data_values = solver._get_trace_plot_data()
    y_values = data_values[component]

    kwargs = {}
    if color is not None:
        kwargs['color'] = color

    return ax.plot(
        x_values,
        y_values,
        color=color,
        linestyle=linestyle,
        marker=marker
    )


# These are the classes that are templatized over dimension
HomogeneousPermittivity = TemplateClass('HomogeneousPermittivity')
HomogeneousMobility = TemplateClass('HomogeneousMobility')
DealIIFunction = TemplateClass('DealIIFunction')
AnalyticFunction = TemplateClass('AnalyticFunction')
PiecewiseFunction = TemplateClass('PiecewiseFunction')
LinearRecombinationTerm = TemplateClass('LinearRecombinationTerm')
ShockleyReadHallFixedTemperature = TemplateClass('ShockleyReadHallFixedTemperature')
AugerFixedTemperature = TemplateClass('AugerFixedTemperature')
ShockleyReadHall = TemplateClass('ShockleyReadHall')
Auger = TemplateClass('Auger')
SuperimposedRecombinationTerm = TemplateClass('SuperimposedRecombinationTerm')
CustomRecombinationTerm = TemplateClass('CustomRecombinationTerm')
CustomSpacialRecombinationTerm = TemplateClass('CustomSpacialRecombinationTerm')
BoundaryConditionHandler = TemplateClass('BoundaryConditionHandler')
Point = TemplateClass('Point')
Problem = TemplateClass('Problem')
NPSolver = TemplateClass(
    'NPSolver', (
        _plot_solution,
        _plot_error_per_cell,
        _plot_l2_error_per_cell,
        _plot_linfty_error_per_cell,
        _plot_h1_error_per_cell,
        _plot_solution_on_trace
    )
)
