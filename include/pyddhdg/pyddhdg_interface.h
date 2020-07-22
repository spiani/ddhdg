#ifdef DIM
py::class_<HomogeneousPermittivity<DIM>>(m, "HomogeneousPermittivity")
  .def(py::init<const double &>());

py::class_<ElectronMobility<DIM>>(m, "ElectronMobility");

py::class_<HomogeneousElectronMobility<DIM>, ElectronMobility<DIM>>(
  m,
  "HomogeneousElectronMobility")
  .def(py::init<const double &>());

py::class_<DealIIFunction<DIM>>(m, "DealIIFunction").def(py::init<double>());

py::class_<AnalyticFunction<DIM>, DealIIFunction<DIM>>(m, "AnalyticFunction")
  .def(py::init<std::string>())
  .def("get_expression", &AnalyticFunction<DIM>::get_expression);

py::class_<PiecewiseFunction<DIM>, DealIIFunction<DIM>>(m, "PiecewiseFunction")
  .def(py::init<const DealIIFunction<DIM> &,
                const DealIIFunction<DIM> &,
                const DealIIFunction<DIM> &>())
  .def(
    py::init<const std::string &, const std::string &, const std::string &>())
  .def(py::init<const std::string &, const std::string &, double>())
  .def(py::init<const std::string &, double, const std::string &>())
  .def(py::init<const std::string &, double, double>());

py::class_<RecombinationTerm<DIM>>(m, "RecombinationTerm");

py::class_<LinearRecombinationTerm<DIM>, RecombinationTerm<DIM>>(
  m,
  "LinearRecombinationTerm")
  .def(py::init<const DealIIFunction<DIM> &,
                const DealIIFunction<DIM> &,
                const DealIIFunction<DIM> &>())
  .def("get_constant_term", &LinearRecombinationTerm<DIM>::get_constant_term)
  .def("get_n_linear_coefficient",
       &LinearRecombinationTerm<DIM>::get_n_linear_coefficient)
  .def("get_p_linear_coefficient",
       &LinearRecombinationTerm<DIM>::get_p_linear_coefficient);

py::class_<BoundaryConditionHandler<DIM>>(m, "BoundaryConditionHandler")
  .def(py::init<>())
  .def("add_boundary_condition",
       py::overload_cast<dealii::types::boundary_id,
                         Ddhdg::BoundaryConditionType,
                         Ddhdg::Component,
                         const DealIIFunction<DIM> &>(
         &BoundaryConditionHandler<DIM>::add_boundary_condition))
  .def("add_boundary_condition",
       py::overload_cast<dealii::types::boundary_id,
                         Ddhdg::BoundaryConditionType,
                         Ddhdg::Component,
                         const std::string &>(
         &BoundaryConditionHandler<DIM>::add_boundary_condition))
  .def("has_dirichlet_boundary_conditions",
       &BoundaryConditionHandler<DIM>::has_dirichlet_boundary_conditions)
  .def("has_neumann_boundary_conditions",
       &BoundaryConditionHandler<DIM>::has_neumann_boundary_conditions);

py::class_<Problem<DIM>>(m, "Problem")
  .def(py::init<double,
                double,
                HomogeneousPermittivity<DIM> &,
                ElectronMobility<DIM> &,
                ElectronMobility<DIM> &,
                RecombinationTerm<DIM> &,
                DealIIFunction<DIM> &,
                DealIIFunction<DIM> &,
                BoundaryConditionHandler<DIM> &,
                double,
                double,
                double,
                double>(),
       py::arg("left"),
       py::arg("right"),
       py::arg("permittivity"),
       py::arg("n_electron_mobility"),
       py::arg("p_electron_mobility"),
       py::arg("recombination_term"),
       py::arg("temperature"),
       py::arg("doping"),
       py::arg("boundary_condition_handler"),
       py::arg("conduction_band_density"),
       py::arg("valence_band_density"),
       py::arg("conduction_band_edge_energy") = 0,
       py::arg("valence_band_edge_energy")    = 0);

py::class_<NPSolver<DIM>>(m, "NPSolver")
  .def(py::init<const Problem<DIM> &,
                const Ddhdg::NPSolverParameters &,
                const Ddhdg::Adimensionalizer &>())
  .def_property_readonly("dim", [](const NPSolver<DIM> &) { return DIM; })
  .def_property_readonly("dimension", [](const NPSolver<DIM> &) { return DIM; })
  .def("refine_grid",
       &NPSolver<DIM>::refine_grid,
       py::arg("n")                 = 1,
       py::arg("preserve_solution") = false)
  .def("refine_and_coarsen_fixed_fraction",
       &NPSolver<DIM>::refine_and_coarsen_fixed_fraction,
       py::arg("error_per_cell"),
       py::arg("top_fraction"),
       py::arg("bottom_fraction"),
       py::arg("max_n_cells") = std::numeric_limits<unsigned int>::max())
  .def_property_readonly("n_of_triangulation_levels",
                         &NPSolver<DIM>::n_of_triangulation_levels)
  .def("get_n_dofs", &NPSolver<DIM>::get_n_dofs, py::arg("for_trace") = false)
  .def_property_readonly("n_active_cells", &NPSolver<DIM>::get_n_active_cells)
  .def("get_cell_vertices",
       [](const NPSolver<DIM> &self) {
         const unsigned int active_cells = self.get_n_active_cells();
         const unsigned int vertices_per_cell =
           dealii::GeometryInfo<DIM>::vertices_per_cell;
         const size_t size = active_cells * vertices_per_cell * DIM;
         // Copy the data in a new buffer
         auto *data = new double[size];
         self.get_cell_vertices(data);

         // Create a Python object that will free the allocated
         // memory when destroyed:
         py::capsule free_when_done(data, [](void *f) {
           auto *data = reinterpret_cast<double *>(f);
           delete[] data;
         });

         const auto vector_shape =
           std::vector<long>{active_cells, vertices_per_cell, DIM};
         const auto vector_stride =
           std::vector<long>{vertices_per_cell * DIM * 8, DIM * 8, 8};

         return py::array_t<double, py::array::c_style>(
           vector_shape,    // shape
           vector_stride,   // strides
           data,            // the data pointer
           free_when_done); // numpy array references this parent
       })
  .def("set_component",
       py::overload_cast<Ddhdg::Component, const std::string &, bool>(
         &NPSolver<DIM>::set_component),
       py::arg("component"),
       py::arg("analytic_function"),
       py::arg("use_projection") = false)
  .def("set_component",
       py::overload_cast<Ddhdg::Component, DealIIFunction<DIM>, bool>(
         &NPSolver<DIM>::set_component),
       py::arg("component"),
       py::arg("analytic_function"),
       py::arg("use_projection") = false)
  .def("set_current_solution",
       &NPSolver<DIM>::set_current_solution,
       py::arg("V_function"),
       py::arg("n_function"),
       py::arg("p_function"),
       py::arg("use_projection") = false)
  .def("set_multithreading", &NPSolver<DIM>::set_multithreading)
  .def("enable_component",
       &NPSolver<DIM>::enable_component,
       py::arg("component"))
  .def("disable_component",
       &NPSolver<DIM>::disable_component,
       py::arg("component"))
  .def("set_enabled_components",
       &NPSolver<DIM>::set_enabled_components,
       py::arg("v_enabled"),
       py::arg("n_enabled"),
       py::arg("p_enabled"))
  .def("run", &NPSolver<DIM>::run)
  .def("copy_triangulation_from", &NPSolver<DIM>::copy_triangulation_from)
  .def("copy_solution_from", &NPSolver<DIM>::copy_solution_from)
  .def("get_parameters", &NPSolver<DIM>::get_parameters)
  .def("compute_local_charge_neutrality_on_trace",
       &NPSolver<DIM>::compute_local_charge_neutrality_on_trace,
       py::arg("only_at_boundary") = false)
  .def("compute_local_charge_neutrality",
       &NPSolver<DIM>::compute_local_charge_neutrality)
  .def(
    "compute_thermodynamic_equilibrium",
    py::overload_cast<bool>(&NPSolver<DIM>::compute_thermodynamic_equilibrium),
    py::arg("generate_first_guess") = true)
  .def("compute_thermodynamic_equilibrium",
       py::overload_cast<double, double, int, bool>(
         &NPSolver<DIM>::compute_thermodynamic_equilibrium),
       py::arg("absolute_tol"),
       py::arg("relative_tol"),
       py::arg("max_number_of_iterations"),
       py::arg("generate_first_guess") = true)
  .def("estimate_error_per_cell",
       py::overload_cast<const Ddhdg::Component>(
         &NPSolver<DIM>::estimate_error_per_cell,
         py::const_),
       py::arg("component"))
  .def("estimate_l2_error_per_cell",
       py::overload_cast<const DealIIFunction<DIM>, const Ddhdg::Component>(
         &NPSolver<DIM>::estimate_l2_error_per_cell,
         py::const_),
       py::arg("expected_solution"),
       py::arg("component"))
  .def("estimate_l2_error_per_cell",
       py::overload_cast<const DealIIFunction<DIM>, const Ddhdg::Displacement>(
         &NPSolver<DIM>::estimate_l2_error_per_cell,
         py::const_),
       py::arg("expected_solution"),
       py::arg("displacement"))
  .def("estimate_l2_error_per_cell",
       py::overload_cast<const NPSolver<DIM>, const Ddhdg::Component>(
         &NPSolver<DIM>::estimate_l2_error_per_cell,
         py::const_),
       py::arg("solver"),
       py::arg("component"))
  .def("estimate_l2_error_per_cell",
       py::overload_cast<const NPSolver<DIM>, const Ddhdg::Displacement>(
         &NPSolver<DIM>::estimate_l2_error_per_cell,
         py::const_),
       py::arg("solver"),
       py::arg("displacement"))
  .def("estimate_h1_error_per_cell",
       py::overload_cast<const DealIIFunction<DIM>, const Ddhdg::Component>(
         &NPSolver<DIM>::estimate_h1_error_per_cell,
         py::const_),
       py::arg("expected_solution"),
       py::arg("component"))
  .def("estimate_h1_error_per_cell",
       py::overload_cast<const DealIIFunction<DIM>, const Ddhdg::Displacement>(
         &NPSolver<DIM>::estimate_h1_error_per_cell,
         py::const_),
       py::arg("expected_solution"),
       py::arg("displacement"))
  .def("estimate_h1_error_per_cell",
       py::overload_cast<const NPSolver<DIM>, const Ddhdg::Component>(
         &NPSolver<DIM>::estimate_h1_error_per_cell,
         py::const_),
       py::arg("solver"),
       py::arg("component"))
  .def("estimate_h1_error_per_cell",
       py::overload_cast<const NPSolver<DIM>, const Ddhdg::Displacement>(
         &NPSolver<DIM>::estimate_h1_error_per_cell,
         py::const_),
       py::arg("solver"),
       py::arg("displacement"))
  .def("estimate_linfty_error_per_cell",
       py::overload_cast<const DealIIFunction<DIM>, const Ddhdg::Component>(
         &NPSolver<DIM>::estimate_linfty_error_per_cell,
         py::const_),
       py::arg("expected_solution"),
       py::arg("component"))
  .def("estimate_linfty_error_per_cell",
       py::overload_cast<const DealIIFunction<DIM>, const Ddhdg::Displacement>(
         &NPSolver<DIM>::estimate_linfty_error_per_cell,
         py::const_),
       py::arg("expected_solution"),
       py::arg("displacement"))
  .def("estimate_linfty_error_per_cell",
       py::overload_cast<const NPSolver<DIM>, const Ddhdg::Component>(
         &NPSolver<DIM>::estimate_linfty_error_per_cell,
         py::const_),
       py::arg("solver"),
       py::arg("component"))
  .def("estimate_linfty_error_per_cell",
       py::overload_cast<const NPSolver<DIM>, const Ddhdg::Displacement>(
         &NPSolver<DIM>::estimate_linfty_error_per_cell,
         py::const_),
       py::arg("solver"),
       py::arg("displacement"))
  .def("estimate_l2_error",
       py::overload_cast<const DealIIFunction<DIM>, const Ddhdg::Component>(
         &NPSolver<DIM>::estimate_l2_error,
         py::const_),
       py::arg("expected_solution"),
       py::arg("component"))
  .def("estimate_l2_error",
       py::overload_cast<const DealIIFunction<DIM>, const Ddhdg::Displacement>(
         &NPSolver<DIM>::estimate_l2_error,
         py::const_),
       py::arg("expected_solution"),
       py::arg("displacement"))
  .def("estimate_l2_error",
       py::overload_cast<const NPSolver<DIM>, const Ddhdg::Component>(
         &NPSolver<DIM>::estimate_l2_error,
         py::const_),
       py::arg("solver"),
       py::arg("component"))
  .def("estimate_l2_error",
       py::overload_cast<const NPSolver<DIM>, const Ddhdg::Displacement>(
         &NPSolver<DIM>::estimate_l2_error,
         py::const_),
       py::arg("solver"),
       py::arg("displacement"))
  .def("estimate_h1_error",
       py::overload_cast<const DealIIFunction<DIM>, const Ddhdg::Component>(
         &NPSolver<DIM>::estimate_h1_error,
         py::const_),
       py::arg("expected_solution"),
       py::arg("component"))
  .def("estimate_h1_error",
       py::overload_cast<const DealIIFunction<DIM>, const Ddhdg::Displacement>(
         &NPSolver<DIM>::estimate_h1_error,
         py::const_),
       py::arg("expected_solution"),
       py::arg("displacement"))
  .def("estimate_h1_error",
       py::overload_cast<const NPSolver<DIM>, const Ddhdg::Component>(
         &NPSolver<DIM>::estimate_h1_error,
         py::const_),
       py::arg("solver"),
       py::arg("component"))
  .def("estimate_h1_error",
       py::overload_cast<const NPSolver<DIM>, const Ddhdg::Displacement>(
         &NPSolver<DIM>::estimate_h1_error,
         py::const_),
       py::arg("solver"),
       py::arg("displacement"))
  .def("estimate_linfty_error",
       py::overload_cast<const DealIIFunction<DIM>, const Ddhdg::Component>(
         &NPSolver<DIM>::estimate_linfty_error,
         py::const_),
       py::arg("expected_solution"),
       py::arg("component"))
  .def("estimate_linfty_error",
       py::overload_cast<const DealIIFunction<DIM>, const Ddhdg::Displacement>(
         &NPSolver<DIM>::estimate_linfty_error,
         py::const_),
       py::arg("expected_solution"),
       py::arg("displacement"))
  .def("estimate_linfty_error",
       py::overload_cast<const NPSolver<DIM>, const Ddhdg::Component>(
         &NPSolver<DIM>::estimate_linfty_error,
         py::const_),
       py::arg("solver"),
       py::arg("component"))
  .def("estimate_linfty_error",
       py::overload_cast<const NPSolver<DIM>, const Ddhdg::Displacement>(
         &NPSolver<DIM>::estimate_linfty_error,
         py::const_),
       py::arg("solver"),
       py::arg("displacement"))
  .def("estimate_l2_error_on_trace",
       &NPSolver<DIM>::estimate_l2_error_on_trace,
       py::arg("expected_solution"),
       py::arg("component"))
  .def("estimate_linfty_error_on_trace",
       &NPSolver<DIM>::estimate_linfty_error_on_trace,
       py::arg("expected_solution"),
       py::arg("component"))
  .def("get_solution", &NPSolver<DIM>::get_solution, py::arg("component"))
  .def("get_solution_on_a_point",
       &NPSolver<DIM>::get_solution_on_a_point,
       py::arg("point"),
       py::arg("component"))
  .def("output_results",
       py::overload_cast<const std::string &, const bool>(
         &NPSolver<DIM>::output_results,
         py::const_),
       py::arg("solution_filename"),
       py::arg("save_update") = false)
#  if DIM != 1
  .def("output_results",
       py::overload_cast<const std::string &, const std::string &, const bool>(
         &NPSolver<DIM>::output_results,
         py::const_),
       py::arg("solution_filename"),
       py::arg("trace_filename"),
       py::arg("save_update") = false)
#  endif
  .def("print_convergence_table",
       py::overload_cast<const std::string &,
                         const std::string &,
                         const std::string &,
                         const unsigned int,
                         const unsigned int>(
         &NPSolver<DIM>::print_convergence_table),
       py::arg("expected_v_solution"),
       py::arg("expected_n_solution"),
       py::arg("expected_p_solution"),
       py::arg("n_cycles"),
       py::arg("initial_refinements") = 0)
  .def("print_convergence_table",
       py::overload_cast<const std::string &,
                         const std::string &,
                         const std::string &,
                         const std::string &,
                         const std::string &,
                         const std::string &,
                         const unsigned int,
                         const unsigned int>(
         &NPSolver<DIM>::print_convergence_table),
       py::arg("expected_v_solution"),
       py::arg("expected_n_solution"),
       py::arg("expected_p_solution"),
       py::arg("initial_v_function"),
       py::arg("initial_n_function"),
       py::arg("initial_p_function"),
       py::arg("n_cycles"),
       py::arg("initial_refinements") = 0);
#endif
