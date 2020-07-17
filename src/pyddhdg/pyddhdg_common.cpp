#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "pyddhdg/pyddhdg.h"

namespace py = pybind11;

namespace pyddhdg
{
  PYBIND11_MODULE(pyddhdg_common, m)
  {
    m.doc() = "The common components of pyddhdg among all dimensions";

    m.attr("CONSTANT_Q")    = py::float_(Ddhdg::Constants::Q);
    m.attr("CONSTANT_KB")   = py::float_(Ddhdg::Constants::KB);
    m.attr("CONSTANT_EPS0") = py::float_(Ddhdg::Constants::EPSILON0);

    py::enum_<Ddhdg::Component>(m, "Component")
      .value("v", Ddhdg::Component::V)
      .value("n", Ddhdg::Component::n)
      .value("p", Ddhdg::Component::p)
      .export_values();

    py::enum_<Ddhdg::Displacement>(m, "Displacement")
      .value("e", Ddhdg::Displacement::E)
      .value("wn", Ddhdg::Displacement::Wn)
      .value("wp", Ddhdg::Displacement::Wp)
      .export_values();

    py::enum_<Ddhdg::BoundaryConditionType>(m, "BoundaryConditionType")
      .value("DIRICHLET", Ddhdg::BoundaryConditionType::dirichlet)
      .value("NEUMANN", Ddhdg::BoundaryConditionType::neumann)
      .value("ROBIN", Ddhdg::BoundaryConditionType::robin)
      .export_values();

    py::class_<ErrorPerCell>(m, "ErrorPerCell")
      .def("as_numpy_array", [](const ErrorPerCell &self) {
        const size_t size = self.data_vector->size();
        // Copy the data in a new buffer
        auto *data = new double[size];
        for (size_t i = 0; i < size; i++)
          {
            data[i] = (double)((*(self.data_vector))[i]);
          }

        // Create a Python object that will free the allocated
        // memory when destroyed:
        py::capsule free_when_done(data, [](void *f) {
          auto *data = reinterpret_cast<double *>(f);
          delete[] data;
        });

        return py::array_t<double>(
          {size},          // shape
          {8},             // strides
          data,            // the data pointer
          free_when_done); // numpy array references this parent
      });

    py::class_<Ddhdg::NPSolverParameters>(m, "NPSolverParameters")
      .def(py::init<const unsigned int,
                    const unsigned int,
                    const unsigned int,
                    const double,
                    const double,
                    const int,
                    const double,
                    const double,
                    const double,
                    const bool,
                    const bool>(),
           py::arg("v_degree")                 = 1,
           py::arg("n_degree")                 = 1,
           py::arg("p_degree")                 = 1,
           py::arg("abs_tolerance")            = 1e-9,
           py::arg("rel_tolerance")            = 1e-9,
           py::arg("max_number_of_iterations") = 100,
           py::arg("v_tau")                    = 1.,
           py::arg("n_tau")                    = 1.,
           py::arg("p_tau")                    = 1.,
           py::arg("iterative_linear_solver")  = false,
           py::arg("multithreading")           = true)
      .def_property_readonly("v_degree",
                             [](const Ddhdg::NPSolverParameters &a) {
                               return a.degree.at(Ddhdg::Component::V);
                             })
      .def_property_readonly("n_degree",
                             [](const Ddhdg::NPSolverParameters &a) {
                               return a.degree.at(Ddhdg::Component::n);
                             })
      .def_property_readonly("p_degree",
                             [](const Ddhdg::NPSolverParameters &a) {
                               return a.degree.at(Ddhdg::Component::p);
                             })
      .def_readonly(
        "abs_tolerance",
        &Ddhdg::NPSolverParameters::nonlinear_solver_absolute_tolerance)
      .def_readonly(
        "rel_tolerance",
        &Ddhdg::NPSolverParameters::nonlinear_solver_relative_tolerance)
      .def_readonly(
        "max_number_of_iterations",
        &Ddhdg::NPSolverParameters::nonlinear_solver_max_number_of_iterations)
      .def_readonly("iterative_linear_solver",
                    &Ddhdg::NPSolverParameters::iterative_linear_solver)
      .def_readonly("multithreading",
                    &Ddhdg::NPSolverParameters::multithreading)
      .def_property_readonly("v_tau",
                             [](const Ddhdg::NPSolverParameters &a) {
                               return a.tau.at(Ddhdg::Component::V);
                             })
      .def_property_readonly("n_tau",
                             [](const Ddhdg::NPSolverParameters &a) {
                               return a.tau.at(Ddhdg::Component::n);
                             })
      .def_property_readonly("p_tau", [](const Ddhdg::NPSolverParameters &a) {
        return a.tau.at(Ddhdg::Component::p);
      });

    py::class_<Ddhdg::Adimensionalizer>(m, "Adimensionalizer")
      .def(py::init<double, double, double, double>(),
           py::arg("scale_length")                = 1.,
           py::arg("temperature_magnitude")       = 1.,
           py::arg("doping_magnitude")            = 1.,
           py::arg("electron_mobility_magnitude") = 1.)
      .def_readonly("scale_length", &Ddhdg::Adimensionalizer::scale_length)
      .def_readonly("temperature_magnitude",
                    &Ddhdg::Adimensionalizer::temperature_magnitude)
      .def_readonly("doping_magnitude",
                    &Ddhdg::Adimensionalizer::doping_magnitude)
      .def_readonly("electron_mobility_magnitude",
                    &Ddhdg::Adimensionalizer::electron_mobility_magnitude);

    py::class_<Ddhdg::NonlinearIterationResults>(m, "NonlinearIterationResults")
      .def(py::init<const bool, const unsigned int, const double>())
      .def_readonly("converged", &Ddhdg::NonlinearIterationResults::converged)
      .def_readonly("iterations", &Ddhdg::NonlinearIterationResults::iterations)
      .def_readonly("last_update_norm",
                    &Ddhdg::NonlinearIterationResults::last_update_norm);
  }
} // namespace pyddhdg