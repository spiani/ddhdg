from importlib import import_module

import pyddhdg.pyddhdg_common

# We import the modules for the three dimensions and we put them
# in a dictionary
_MODULE_PER_DIM = {
    1: import_module('pyddhdg.pyddhdg1d.pyddhdg1d'),
    2: import_module('pyddhdg.pyddhdg2d.pyddhdg2d'),
    3: import_module('pyddhdg.pyddhdg3d.pyddhdg3d')
}


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
    def __init__(self, class_name):
        self._reference_class = {}
        for i in range(1, 4):
            self._reference_class[i] = getattr(_MODULE_PER_DIM[i], class_name)
        self._name = class_name

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


# These are the classes that are templatized over dimension
HomogeneousPermittivity = TemplateClass('HomogeneousPermittivity')
HomogeneousElectronMobility = TemplateClass('HomogeneousElectronMobility')
DealIIFunction = TemplateClass('DealIIFunction')
AnalyticFunction = TemplateClass('AnalyticFunction')
PiecewiseFunction = TemplateClass('PiecewiseFunction')
LinearRecombinationTerm = TemplateClass('LinearRecombinationTerm')
BoundaryConditionHandler = TemplateClass('BoundaryConditionHandler')
Problem = TemplateClass('Problem')
NPSolver = TemplateClass('NPSolver')
