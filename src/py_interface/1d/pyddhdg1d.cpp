#include <pybind11/pybind11.h>

#include "py_interface/pyddhdg.h"

namespace py = pybind11;

const unsigned int dim = 1;

namespace pyddhdg
{
  PYBIND11_MODULE(pyddhdg1d, m)
  {
    m.doc() = "A python interface for ddhdg in 2D";

    py::enum_<Ddhdg::Component>(m, "Component")
      .value("v", Ddhdg::Component::V)
      .value("n", Ddhdg::Component::n)
      .export_values();

    py::class_<Permittivity<dim>>(m, "Permittivity");

    py::class_<HomogeneousPermittivity<dim>, Permittivity<dim>>(
      m, "HomogeneousPermittivity")
      .def(py::init<const double &>());

    py::class_<ElectronMobility<dim>>(m, "ElectronMobility");

    py::class_<HomogeneousElectronMobility<dim>, ElectronMobility<dim>>(
      m, "HomogeneousElectronMobility")
      .def(py::init<const double &>());

    py::class_<PythonFunction<dim>>(m, "AnalyticFunction")
      .def(py::init<const std::string &>())
      .def("get_expression", &PythonFunction<dim>::get_expression);

    py::class_<RecombinationTerm<dim>>(m, "RecombinationTerm");

    py::class_<LinearRecombinationTerm<dim>, RecombinationTerm<dim>>(
      m, "LinearRecombinationTerm")
      .def(py::init<const PythonFunction<dim> &, const PythonFunction<dim> &>())
      .def("get_constant_term",
           &LinearRecombinationTerm<dim>::get_constant_term)
      .def("get_linear_coefficient",
           &LinearRecombinationTerm<dim>::get_linear_coefficient);

    py::class_<Temperature<dim>>(m, "Temperature")
      .def(py::init<const std::string &>())
      .def("get_expression", &Temperature<dim>::get_expression);

    py::class_<Doping<dim>>(m, "Doping")
      .def(py::init<const std::string &>())
      .def("get_expression", &Doping<dim>::get_expression);

    py::enum_<Ddhdg::BoundaryConditionType>(m, "BoundaryConditionType")
      .value("DIRICHLET", Ddhdg::BoundaryConditionType::dirichlet)
      .value("NEUMANN", Ddhdg::BoundaryConditionType::neumann)
      .value("ROBIN", Ddhdg::BoundaryConditionType::robin)
      .export_values();

    py::class_<BoundaryConditionHandler<dim>>(m, "BoundaryConditionHandler")
      .def(py::init<>())
      .def("add_boundary_condition_from_function",
           &BoundaryConditionHandler<dim>::add_boundary_condition_from_function)
      .def("add_boundary_condition_from_string",
           &BoundaryConditionHandler<dim>::add_boundary_condition_from_string)
      .def("has_dirichlet_boundary_conditions",
           &BoundaryConditionHandler<dim>::has_dirichlet_boundary_conditions)
      .def("has_neumann_boundary_conditions",
           &BoundaryConditionHandler<dim>::has_neumann_boundary_conditions);

    py::class_<Problem<dim>>(m, "Problem")
      .def(py::init<Permittivity<dim> &,
                    ElectronMobility<dim> &,
                    RecombinationTerm<dim> &,
                    Temperature<dim> &,
                    Doping<dim> &,
                    BoundaryConditionHandler<dim> &>());

    py::class_<Ddhdg::SolverParameters>(m, "SolverParameters")
      .def(py::init<const unsigned int,
                    const unsigned int,
                    const double,
                    const double,
                    const int,
                    const double,
                    const double,
                    const bool,
                    const bool>(),
           py::arg("v_degree")                 = 1,
           py::arg("n_degree")                 = 1,
           py::arg("abs_tolerance")            = 1e-9,
           py::arg("rel_tolerance")            = 1e-9,
           py::arg("max_number_of_iterations") = 100,
           py::arg("v tau")                    = 1.,
           py::arg("n tau")                    = 1.,
           py::arg("iterative linear solver")  = false,
           py::arg("multithreading")           = true)
      .def_readonly("v_degree", &Ddhdg::SolverParameters::V_degree)
      .def_readonly("n_degree", &Ddhdg::SolverParameters::n_degree)
      .def_readonly(
        "abs_tolerance",
        &Ddhdg::SolverParameters::nonlinear_solver_absolute_tolerance)
      .def_readonly(
        "rel_tolerance",
        &Ddhdg::SolverParameters::nonlinear_solver_relative_tolerance)
      .def_readonly(
        "max_number_of_iterations",
        &Ddhdg::SolverParameters::nonlinear_solver_max_number_of_iterations)
      .def_readonly("iterative_linear_solver",
                    &Ddhdg::SolverParameters::iterative_linear_solver)
      .def_readonly("multithreading", &Ddhdg::SolverParameters::multithreading)
      .def_property_readonly("v_tau",
                             [](const Ddhdg::SolverParameters &a) {
                               return a.tau.at(Ddhdg::Component::V);
                             })
      .def_property_readonly("n_tau", [](const Ddhdg::SolverParameters &a) {
        return a.tau.at(Ddhdg::Component::n);
      });
  }
} // namespace pyddhdg
