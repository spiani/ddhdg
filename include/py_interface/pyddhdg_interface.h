#ifdef DIM
py::enum_<Ddhdg::Component>(m, "Component")
  .value("v", Ddhdg::Component::V)
  .value("n", Ddhdg::Component::n)
  .export_values();

py::class_<Permittivity<DIM>>(m, "Permittivity");

py::class_<HomogeneousPermittivity<DIM>, Permittivity<DIM>>(
  m,
  "HomogeneousPermittivity")
  .def(py::init<const double &>());

py::class_<ElectronMobility<DIM>>(m, "ElectronMobility");

py::class_<HomogeneousElectronMobility<DIM>, ElectronMobility<DIM>>(
  m,
  "HomogeneousElectronMobility")
  .def(py::init<const double &>());

py::class_<PythonFunction<DIM>>(m, "AnalyticFunction")
  .def(py::init<const std::string &>())
  .def("get_expression", &PythonFunction<DIM>::get_expression);

py::class_<RecombinationTerm<DIM>>(m, "RecombinationTerm");

py::class_<LinearRecombinationTerm<DIM>, RecombinationTerm<DIM>>(
  m,
  "LinearRecombinationTerm")
  .def(py::init<const PythonFunction<DIM> &, const PythonFunction<DIM> &>())
  .def("get_constant_term", &LinearRecombinationTerm<DIM>::get_constant_term)
  .def("get_linear_coefficient",
       &LinearRecombinationTerm<DIM>::get_linear_coefficient);

py::class_<Temperature<DIM>>(m, "Temperature")
  .def(py::init<const std::string &>())
  .def("get_expression", &Temperature<DIM>::get_expression);

py::class_<Doping<DIM>>(m, "Doping")
  .def(py::init<const std::string &>())
  .def("get_expression", &Doping<DIM>::get_expression);

py::enum_<Ddhdg::BoundaryConditionType>(m, "BoundaryConditionType")
  .value("DIRICHLET", Ddhdg::BoundaryConditionType::dirichlet)
  .value("NEUMANN", Ddhdg::BoundaryConditionType::neumann)
  .value("ROBIN", Ddhdg::BoundaryConditionType::robin)
  .export_values();

py::class_<BoundaryConditionHandler<DIM>>(m, "BoundaryConditionHandler")
  .def(py::init<>())
  .def("add_boundary_condition_from_function",
       &BoundaryConditionHandler<DIM>::add_boundary_condition_from_function)
  .def("add_boundary_condition_from_string",
       &BoundaryConditionHandler<DIM>::add_boundary_condition_from_string)
  .def("has_dirichlet_boundary_conditions",
       &BoundaryConditionHandler<DIM>::has_dirichlet_boundary_conditions)
  .def("has_neumann_boundary_conditions",
       &BoundaryConditionHandler<DIM>::has_neumann_boundary_conditions);

py::class_<Problem<DIM>>(m, "Problem")
  .def(py::init<Permittivity<DIM> &,
                ElectronMobility<DIM> &,
                RecombinationTerm<DIM> &,
                Temperature<DIM> &,
                Doping<DIM> &,
                BoundaryConditionHandler<DIM> &>());

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
       py::arg("v_tau")                    = 1.,
       py::arg("n_tau")                    = 1.,
       py::arg("iterative_linear_solver")  = false,
       py::arg("multithreading")           = true)
  .def_readonly("v_degree", &Ddhdg::SolverParameters::V_degree)
  .def_readonly("n_degree", &Ddhdg::SolverParameters::n_degree)
  .def_readonly("abs_tolerance",
                &Ddhdg::SolverParameters::nonlinear_solver_absolute_tolerance)
  .def_readonly("rel_tolerance",
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

py::class_<Ddhdg::NonlinearIteratorStatus>(m, "NonlinearIteratorStatus")
  .def(py::init<const bool, const unsigned int, const double>())
  .def_readonly("converged", &Ddhdg::NonlinearIteratorStatus::converged)
  .def_readonly("iterations", &Ddhdg::NonlinearIteratorStatus::iterations)
  .def_readonly("last_update_norm",
                &Ddhdg::NonlinearIteratorStatus::last_update_norm);

py::class_<Solver<DIM>>(m, "Solver")
  .def(py::init<const Problem<DIM> &, Ddhdg::SolverParameters &>())
  .def("refine_grid", &Solver<DIM>::refine_grid, py::arg("n") = 1)
  .def("set_component", &Solver<DIM>::set_component)
  .def("set_current_solution", &Solver<DIM>::set_current_solution)
  .def("set_multithreading", &Solver<DIM>::set_multithreading)
  .def("run", &Solver<DIM>::run)
  .def("estimate_l2_error", &Solver<DIM>::estimate_l2_error)
  .def("estimate_h1_error", &Solver<DIM>::estimate_h1_error)
  .def("estimate_linfty_error", &Solver<DIM>::estimate_linfty_error)
  .def("output_results",
       py::overload_cast<const std::string &, const bool>(
         &Solver<DIM>::output_results,
         py::const_),
       py::arg("solution_filename"),
       py::arg("save_update") = false)
#  if DIM != 1
  .def("output_results",
       py::overload_cast<const std::string &, const std::string &, const bool>(
         &Solver<DIM>::output_results,
         py::const_),
       py::arg("solution_filename"),
       py::arg("trace_filename"),
       py::arg("save_update") = false)
#  endif
  .def("print_convergence_table",
       py::overload_cast<const std::string &,
                         const std::string &,
                         const unsigned int,
                         const unsigned int>(
         &Solver<DIM>::print_convergence_table),
       py::arg("expected_v_solution"),
       py::arg("expected_n_solution"),
       py::arg("n_cycles"),
       py::arg("initial_refinements") = 0)
  .def("print_convergence_table",
       py::overload_cast<const std::string &,
                         const std::string &,
                         const std::string &,
                         const std::string &,
                         const unsigned int,
                         const unsigned int>(
         &Solver<DIM>::print_convergence_table),
       py::arg("expected_v_solution"),
       py::arg("expected_n_solution"),
       py::arg("initial_v_function"),
       py::arg("initial_n_function"),
       py::arg("n_cycles"),
       py::arg("initial_refinements") = 0);
#endif
