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
      .value("phi_n", Ddhdg::Component::phi_n)
      .value("phi_p", Ddhdg::Component::phi_p)
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

    py::enum_<Ddhdg::DDFluxType>(m, "DDFluxType")
      .value("use_cell", Ddhdg::DDFluxType::use_cell)
      .value("use_trace", Ddhdg::DDFluxType::use_trace)
      .value("qiu_shi_stabilization", Ddhdg::DDFluxType::qiu_shi_stabilization)
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



    py::class_<Ddhdg::NonlinearSolverParameters,
               std::shared_ptr<Ddhdg::NonlinearSolverParameters>>(
      m, "NonlinearSolverParameters")
      .def(py::init<const double, const double, const int, const double>(),
           py::arg("abs_tolerance")            = 1e-9,
           py::arg("rel_tolerance")            = 1e-9,
           py::arg("max_number_of_iterations") = 100,
           py::arg("alpha")                    = 1.)
      .def_readwrite("abs_tolerance",
                     &Ddhdg::NonlinearSolverParameters::absolute_tolerance)
      .def_readwrite("rel_tolerance",
                     &Ddhdg::NonlinearSolverParameters::relative_tolerance)
      .def_readwrite(
        "max_number_of_iterations",
        &Ddhdg::NonlinearSolverParameters::max_number_of_iterations)
      .def_readwrite("alpha", &Ddhdg::NonlinearSolverParameters::alpha);



    py::class_<Ddhdg::NPSolverParameters,
               std::shared_ptr<Ddhdg::NPSolverParameters>>(m,
                                                           "NPSolverParameters")
      .def("degree",
           [](const Ddhdg::NPSolverParameters &a, const Ddhdg::Component c) {
             return a.degree.at(c);
           })
      .def_readonly("nonlinear_parameters",
                    &Ddhdg::NPSolverParameters::nonlinear_parameters)
      .def_readonly("iterative_linear_solver",
                    &Ddhdg::NPSolverParameters::iterative_linear_solver)
      .def_readonly("multithreading",
                    &Ddhdg::NPSolverParameters::multithreading);



    py::class_<Ddhdg::FixedTauNPSolverParameters,
               Ddhdg::NPSolverParameters,
               std::shared_ptr<Ddhdg::FixedTauNPSolverParameters>>(
      m, "FixedTauNPSolverParameters")
      .def(py::init<const unsigned int,
                    const unsigned int,
                    const unsigned int,
                    const std::shared_ptr<Ddhdg::NonlinearSolverParameters>,
                    const double,
                    const double,
                    const double,
                    const bool,
                    const bool,
                    const Ddhdg::DDFluxType,
                    const bool>(),
           py::arg("v_degree") = 1,
           py::arg("n_degree") = 1,
           py::arg("p_degree") = 1,
           py::arg("nonlinear_parameters") =
             std::make_shared<Ddhdg::NonlinearSolverParameters>(),
           py::arg("v_tau")                   = 1.,
           py::arg("n_tau")                   = 1.,
           py::arg("p_tau")                   = 1.,
           py::arg("iterative_linear_solver") = false,
           py::arg("multithreading")          = true,
           py::arg("dd_flux_type")            = Ddhdg::DDFluxType::use_cell,
           py::arg("linearize_on_phi")        = false)
      .def_property_readonly("v_tau",
                             [](const Ddhdg::FixedTauNPSolverParameters &a) {
                               return a.get_tau(Ddhdg::Component::V);
                             })
      .def_property_readonly("n_tau",
                             [](const Ddhdg::FixedTauNPSolverParameters &a) {
                               return a.get_tau(Ddhdg::Component::n);
                             })
      .def_property_readonly("p_tau",
                             [](const Ddhdg::FixedTauNPSolverParameters &a) {
                               return a.get_tau(Ddhdg::Component::p);
                             });



    py::class_<Ddhdg::CellFaceTauNPSolverParameters,
               Ddhdg::NPSolverParameters,
               std::shared_ptr<Ddhdg::CellFaceTauNPSolverParameters>>(
      m, "CellFaceTauNPSolverParameters")
      .def(py::init<const unsigned int,
                    const unsigned int,
                    const unsigned int,
                    const std::shared_ptr<Ddhdg::NonlinearSolverParameters>,
                    const bool,
                    const bool,
                    const Ddhdg::DDFluxType,
                    const bool>(),
           py::arg("v_degree") = 1,
           py::arg("n_degree") = 1,
           py::arg("p_degree") = 1,
           py::arg("nonlinear_parameters") =
             std::make_shared<Ddhdg::NonlinearSolverParameters>(),
           py::arg("iterative_linear_solver") = false,
           py::arg("multithreading")          = true,
           py::arg("dd_flux_type")            = Ddhdg::DDFluxType::use_cell,
           py::arg("linearize_on_phi")        = false)
      .def(
        "set_face",
        [](Ddhdg::CellFaceTauNPSolverParameters      &parameters,
           const dealii::python::CellAccessorWrapper &cell,
           const unsigned int                         face,
           const double                               V_tau,
           const double                               n_tau,
           const double                               p_tau) {
          const int dim = cell.get_dim();

          Assert(dim == cell.get_spacedim(),
                 dealii::ExcMessage(
                   "spacedim different from dim is not allowed"));

          const unsigned int faces_per_cell =
            (dim == 1) ? dealii::GeometryInfo<1>::faces_per_cell :
            (dim == 2) ? dealii::GeometryInfo<2>::faces_per_cell :
                         dealii::GeometryInfo<3>::faces_per_cell;

          const unsigned int level = cell.level();
          const unsigned int index = cell.index();

          parameters.set_face(
            level, index, faces_per_cell, face, V_tau, n_tau, p_tau);
        },
        py::arg("cell"),
        py::arg("face"),
        py::arg("V_tau_value"),
        py::arg("n_tau_value"),
        py::arg("p_tau_value"))
      .def(
        "set_multiple_faces",
        [](Ddhdg::CellFaceTauNPSolverParameters &parameters,
           const unsigned int                    dim,
           const pybind11::list                 &cell_levels,
           const pybind11::list                 &cell_indices,
           const pybind11::list                 &faces,
           const pybind11::list                 &V_tau,
           const pybind11::list                 &n_tau,
           const pybind11::list                 &p_tau) {
          const unsigned int n_of_elements = cell_levels.size();

          AssertThrow(
            cell_indices.size() == n_of_elements,
            dealii::ExcMessage(
              "The list passed to this function must have the same size"));
          AssertThrow(
            faces.size() == n_of_elements,
            dealii::ExcMessage(
              "The list passed to this function must have the same size"));
          AssertThrow(
            V_tau.size() == n_of_elements,
            dealii::ExcMessage(
              "The list passed to this function must have the same size"));
          AssertThrow(
            n_tau.size() == n_of_elements,
            dealii::ExcMessage(
              "The list passed to this function must have the same size"));
          AssertThrow(
            p_tau.size() == n_of_elements,
            dealii::ExcMessage(
              "The list passed to this function must have the same size"));

          const unsigned int faces_per_cell =
            (dim == 1) ? dealii::GeometryInfo<1>::faces_per_cell :
            (dim == 2) ? dealii::GeometryInfo<2>::faces_per_cell :
                         dealii::GeometryInfo<3>::faces_per_cell;

          for (unsigned int i = 0; i < n_of_elements; ++i)
            parameters.set_face(pybind11::cast<unsigned int>(cell_levels[i]),
                                pybind11::cast<unsigned int>(cell_indices[i]),
                                faces_per_cell,
                                pybind11::cast<unsigned int>(faces[i]),
                                pybind11::cast<double>(V_tau[i]),
                                pybind11::cast<double>(n_tau[i]),
                                pybind11::cast<double>(p_tau[i]));
        },
        py::arg("dim"),
        py::arg("cell_levels"),
        py::arg("cell_indices"),
        py::arg("faces"),
        py::arg("V_tau_value"),
        py::arg("n_tau_value"),
        py::arg("p_tau_value"));



    py::class_<Ddhdg::Adimensionalizer>(m, "Adimensionalizer")
      .def(py::init<double, double, double, double>(),
           py::arg("scale_length") = 1.,
           py::arg("temperature_magnitude") =
             Ddhdg::Constants::Q / Ddhdg::Constants::KB,
           py::arg("doping_magnitude")            = 1.,
           py::arg("electron_mobility_magnitude") = 1.)
      .def_readonly("scale_length", &Ddhdg::Adimensionalizer::scale_length)
      .def_readonly("temperature_magnitude",
                    &Ddhdg::Adimensionalizer::temperature_magnitude)
      .def_readonly("doping_magnitude",
                    &Ddhdg::Adimensionalizer::doping_magnitude)
      .def_readonly("electron_mobility_magnitude",
                    &Ddhdg::Adimensionalizer::mobility_magnitude);



    py::class_<Ddhdg::NonlinearIterationResults>(m, "NonlinearIterationResults")
      .def(py::init<const bool, const unsigned int, const double>())
      .def_readonly("converged", &Ddhdg::NonlinearIterationResults::converged)
      .def_readonly("iterations", &Ddhdg::NonlinearIterationResults::iterations)
      .def_readonly("last_update_norm",
                    &Ddhdg::NonlinearIterationResults::last_update_norm);
  }
} // namespace pyddhdg