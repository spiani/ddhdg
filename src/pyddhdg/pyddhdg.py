import logging
import warnings
from importlib import import_module
from types import MethodType

import numpy as np
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

# This constant are useful only if matplotlib is imported
if MATPLOTLIB_IMPORTED:
    CMAP = get_cmap('plasma')
else:
    CMAP = None

GRID_LINEWIDTH = 0.1
GRID_LINESTYLE = "dotted"


class InvalidDimensionException(ValueError):
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

# Export also the components and the others enum; in this way we avoid to
# pollute the global namespace with the name of the single elements of the
# enums (i.e. it will be necessary to write Component.v and not simply v)
Component = pyddhdg_common.Component
Displacement = pyddhdg_common.Displacement
BoundaryConditionType = pyddhdg_common.BoundaryConditionType

# Now, we also import the common classes from the pyddhdg_common module
ErrorPerCell = pyddhdg_common.ErrorPerCell
NPSolverParameters = pyddhdg_common.NPSolverParameters
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


# Before initialize the template class, we introduce the methods that will be binded to
# some of them

def _plot_solution(solver, component, plot_grid=False, ax=None, colors=None,
                   linewidth=1, grid_color=(.8, .8, .8)):
    if not MATPLOTLIB_IMPORTED:
        raise ModuleNotFoundError('The method "plot_solution" requires matplotlib')
    if not SCIPY_IMPORTED:
        raise ModuleNotFoundError('The method "plot_solution" requires scipy')

    if ax is None:
        ax = plt.gca()

    if colors is None:
        colors = CMAP

    degree = solver.get_parameters().degree(component)
    cells = solver.n_active_cells
    points = degree + 1

    dim = solver.dimension
    if dim != 1:
        raise ValueError('No plot available in {}D'.format(dim))

    plot_points = points if degree < 2 else max(1, 1000 // cells)

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

    for cell in range(cells):
        cell_vertices = all_cell_vertices[cell_order[cell]]
        cell_x_min = np.min(cell_vertices)
        cell_x_max = np.max(cell_vertices)

        # Interpolate a polynomial on the current cell
        x_points = np.linspace(cell_x_min, cell_x_max, points + 2)[1:-1]
        y_points = np.empty_like(x_points)
        for i in range(points):
            p = pyddhdg.Point[dim](x_points[i])
            y_points[i] = solver.get_solution_on_a_point(p, component)

        x_plot_points = np.linspace(cell_x_min, cell_x_max, plot_points)
        y_plot_points = barycentric_interpolate(x_points, y_points, x_plot_points)

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


# These are the classes that are templatized over dimension
HomogeneousPermittivity = TemplateClass('HomogeneousPermittivity')
HomogeneousElectronMobility = TemplateClass('HomogeneousElectronMobility')
DealIIFunction = TemplateClass('DealIIFunction')
AnalyticFunction = TemplateClass('AnalyticFunction')
PiecewiseFunction = TemplateClass('PiecewiseFunction')
LinearRecombinationTerm = TemplateClass('LinearRecombinationTerm')
BoundaryConditionHandler = TemplateClass('BoundaryConditionHandler')
Point = TemplateClass('Point')
Problem = TemplateClass('Problem')
NPSolver = TemplateClass('NPSolver', (_plot_solution,))
