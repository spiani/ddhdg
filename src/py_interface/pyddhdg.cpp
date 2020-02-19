#include "py_interface/pyddhdg.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pyddhdg
{
  template <int dim>
  std::shared_ptr<Ddhdg::Permittivity<dim>>
  HomogeneousPermittivity<dim>::generate_ddhdg_permittivity()
  {
    return std::make_shared<Ddhdg::HomogeneousPermittivity<dim>>(this->epsilon);
  }



  template <int dim>
  std::shared_ptr<Ddhdg::ElectronMobility<dim>>
  HomogeneousElectronMobility<dim>::generate_ddhdg_electron_mobility()
  {
    return std::make_shared<Ddhdg::HomogeneousElectronMobility<dim>>(this->mu);
  }


  template <int dim>
  PythonFunction<dim>::PythonFunction(const std::string &f_expr)
    : f_expr(f_expr)
    , f(std::make_shared<dealii::FunctionParser<dim>>(1))
  {
    f->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                  this->f_expr,
                  Ddhdg::Constants::constants);
  }



  template <int dim>
  std::shared_ptr<dealii::Function<dim>>
  PythonFunction<dim>::get_dealii_function() const
  {
    return this->f;
  }



  template <int dim>
  std::string
  PythonFunction<dim>::get_expression() const
  {
    return this->f_expr;
  }



  template <int dim>
  LinearRecombinationTerm<dim>::LinearRecombinationTerm(
    const PythonFunction<dim> &zero_term,
    const PythonFunction<dim> &first_term)
    : zero_term(zero_term.get_expression())
    , first_term(first_term.get_expression())
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::RecombinationTerm<dim>>
  LinearRecombinationTerm<dim>::generate_ddhdg_recombination_term()
  {
    return std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>(
      this->zero_term.get_expression(), this->first_term.get_expression());
  }



  template <int dim>
  std::string
  LinearRecombinationTerm<dim>::get_constant_term() const
  {
    return this->zero_term.get_expression();
  }


  template <int dim>
  std::string
  LinearRecombinationTerm<dim>::get_linear_coefficient() const
  {
    return this->first_term.get_expression();
  }


  template <int dim>
  std::shared_ptr<dealii::Triangulation<dim>>
  Problem<dim>::generate_triangulation()
  {
    std::shared_ptr<dealii::Triangulation<dim>> triangulation =
      std::make_shared<dealii::Triangulation<dim>>();

    dealii::GridGenerator::hyper_cube(*triangulation, 0., 1., true);

    return triangulation;
  }



  PYBIND11_MODULE(pyddhdg, m)
  {
    m.doc() = "A python interface for ddhdg in 2D";

    py::enum_<Ddhdg::Component>(m, "Component")
      .value("v", Ddhdg::Component::V)
      .value("n", Ddhdg::Component::n)
      .export_values();

    py::class_<Permittivity<2>>(m, "Permittivity");

    py::class_<HomogeneousPermittivity<2>, Permittivity<2>>(
      m, "HomogeneousPermittivity")
      .def(py::init<const double &>());

    py::class_<ElectronMobility<2>>(m, "ElectronMobility");

    py::class_<HomogeneousElectronMobility<2>, ElectronMobility<2>>(
      m, "HomogeneousElectronMobility")
      .def(py::init<const double &>());

    py::class_<PythonFunction<2>>(m, "AnalyticFunction")
      .def(py::init<const std::string &>())
      .def("get_expression", &PythonFunction<2>::get_expression);

    py::class_<RecombinationTerm<2>>(m, "RecombinationTerm");

    py::class_<LinearRecombinationTerm<2>, RecombinationTerm<2>>(
      m, "LinearRecombinationTerm")
      .def(py::init<const PythonFunction<2> &, const PythonFunction<2> &>())
      .def("get_constant_term", &LinearRecombinationTerm<2>::get_constant_term)
      .def("get_linear_coefficient",
           &LinearRecombinationTerm<2>::get_linear_coefficient);

    py::class_<Temperature<2>>(m, "Temperature")
      .def(py::init<const std::string &>())
      .def("get_expression", &Temperature<2>::get_expression);

    py::class_<Doping<2>>(m, "Doping")
      .def(py::init<const std::string &>())
      .def("get_expression", &Doping<2>::get_expression);

    py::enum_<Ddhdg::BoundaryConditionType>(m, "BoundaryConditionType")
      .value("DIRICHLET", Ddhdg::BoundaryConditionType::dirichlet)
      .value("NEUMANN", Ddhdg::BoundaryConditionType::neumann)
      .value("ROBIN", Ddhdg::BoundaryConditionType::robin)
      .export_values();

    py::class_<BoundaryConditionHandler<2>>(m, "BoundaryConditionHandler")
      .def(py::init<>())
      .def("add_boundary_condition_from_function",
           &BoundaryConditionHandler<2>::add_boundary_condition_from_function)
      .def("add_boundary_condition_from_string",
           &BoundaryConditionHandler<2>::add_boundary_condition_from_string)
      .def("has_dirichlet_boundary_conditions",
           &BoundaryConditionHandler<2>::has_dirichlet_boundary_conditions)
      .def("has_neumann_boundary_conditions",
           &BoundaryConditionHandler<2>::has_neumann_boundary_conditions);

    py::class_<Problem<2>>(m, "Problem")
      .def(py::init<Permittivity<2> &,
                    ElectronMobility<2> &,
                    RecombinationTerm<2> &,
                    Temperature<2> &,
                    Doping<2> &,
                    BoundaryConditionHandler<2> &>());

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
           py::arg("multithreading")           = true);
  }

} // namespace pyddhdg
