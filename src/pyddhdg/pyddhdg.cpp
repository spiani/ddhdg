#include "pyddhdg/pyddhdg.h"

#include <utility>

#include "function_tools.h"

namespace pyddhdg
{
  template <int dim>
  HomogeneousPermittivity<dim>::HomogeneousPermittivity(const double epsilon)
    : epsilon(epsilon)
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::HomogeneousPermittivity<dim>>
  HomogeneousPermittivity<dim>::generate_ddhdg_permittivity()
  {
    return std::make_shared<Ddhdg::HomogeneousPermittivity<dim>>(this->epsilon);
  }



  template <int dim>
  HomogeneousElectronMobility<dim>::HomogeneousElectronMobility(const double mu)
    : mu(mu)
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::HomogeneousElectronMobility<dim>>
  HomogeneousElectronMobility<dim>::generate_ddhdg_electron_mobility()
  {
    return std::make_shared<Ddhdg::HomogeneousElectronMobility<dim>>(this->mu);
  }



  template <int dim>
  DealIIFunction<dim>::DealIIFunction(
    const std::shared_ptr<dealii::Function<dim>> f)
    : f(f)
  {}


  template <int dim>
  DealIIFunction<dim>::DealIIFunction(const double f_const)
    : f((f_const == 0.) ?
          std::make_shared<dealii::Functions::ZeroFunction<dim>>() :
          std::make_shared<dealii::Functions::ConstantFunction<dim>>(f_const))
  {}



  template <int dim>
  std::shared_ptr<dealii::Function<dim>>
  DealIIFunction<dim>::get_dealii_function() const
  {
    return this->f;
  }



  template <int dim>
  std::shared_ptr<dealii::FunctionParser<dim>>
  AnalyticFunction<dim>::get_function_from_string(const std::string &f_expr)
  {
    auto f = std::make_shared<dealii::FunctionParser<dim>>();
    f->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                  f_expr,
                  Ddhdg::Constants::constants);
    return f;
  }



  template <int dim>
  AnalyticFunction<dim>::AnalyticFunction(std::string f_expr)
    : DealIIFunction<dim>(get_function_from_string(f_expr))
    , f_expr(f_expr)
  {}



  template <int dim>
  std::string
  AnalyticFunction<dim>::get_expression() const
  {
    return this->f_expr;
  }



  template <int dim>
  PiecewiseFunction<dim>::PiecewiseFunction(
    const DealIIFunction<dim> &condition,
    const DealIIFunction<dim> &f1,
    const DealIIFunction<dim> &f2)
    : DealIIFunction<dim>(std::make_shared<Ddhdg::PiecewiseFunction<dim>>(
        condition.get_dealii_function(),
        f1.get_dealii_function(),
        f2.get_dealii_function()))
  {}



  template <int dim>
  PiecewiseFunction<dim>::PiecewiseFunction(const std::string &condition,
                                            const std::string &f1,
                                            const std::string &f2)
    : PiecewiseFunction(AnalyticFunction<dim>(condition),
                        AnalyticFunction<dim>(f1),
                        AnalyticFunction<dim>(f2))
  {}



  template <int dim>
  PiecewiseFunction<dim>::PiecewiseFunction(const std::string &condition,
                                            const std::string &f1,
                                            double             f2)
    : PiecewiseFunction(AnalyticFunction<dim>(condition),
                        AnalyticFunction<dim>(f1),
                        DealIIFunction<dim>(f2))
  {}



  template <int dim>
  PiecewiseFunction<dim>::PiecewiseFunction(const std::string &condition,
                                            double             f1,
                                            const std::string &f2)
    : PiecewiseFunction(AnalyticFunction<dim>(condition),
                        DealIIFunction<dim>(f1),
                        AnalyticFunction<dim>(f2))
  {}



  template <int dim>
  PiecewiseFunction<dim>::PiecewiseFunction(const std::string &condition,
                                            double             f1,
                                            double             f2)
    : PiecewiseFunction(AnalyticFunction<dim>(condition),
                        DealIIFunction<dim>(f1),
                        DealIIFunction<dim>(f2))
  {}



  template <int dim>
  LinearRecombinationTerm<dim>::LinearRecombinationTerm(
    const DealIIFunction<dim> &zero_term,
    const DealIIFunction<dim> &n_linear_coefficient,
    const DealIIFunction<dim> &p_linear_coefficient)
    : zero_term(zero_term)
    , n_linear_coefficient(n_linear_coefficient)
    , p_linear_coefficient(p_linear_coefficient)
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::RecombinationTerm<dim>>
  LinearRecombinationTerm<dim>::generate_ddhdg_recombination_term()
  {
    return std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>(
      this->zero_term.get_dealii_function(),
      this->n_linear_coefficient.get_dealii_function(),
      this->p_linear_coefficient.get_dealii_function());
  }



  template <int dim>
  DealIIFunction<dim>
  LinearRecombinationTerm<dim>::get_constant_term() const
  {
    return this->zero_term;
  }



  template <int dim>
  DealIIFunction<dim>
  LinearRecombinationTerm<dim>::get_n_linear_coefficient() const
  {
    return this->p_linear_coefficient;
  }



  template <int dim>
  DealIIFunction<dim>
  LinearRecombinationTerm<dim>::get_p_linear_coefficient() const
  {
    return this->n_linear_coefficient;
  }



  template <int dim>
  BoundaryConditionHandler<dim>::BoundaryConditionHandler()
    : bc_handler(std::make_shared<Ddhdg::BoundaryConditionHandler<dim>>())
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>>
  BoundaryConditionHandler<dim>::get_ddhdg_boundary_condition_handler()
  {
    return this->bc_handler;
  }



  template <int dim>
  void
  BoundaryConditionHandler<dim>::add_boundary_condition(
    const dealii::types::boundary_id   id,
    const Ddhdg::BoundaryConditionType bc_type,
    const Ddhdg::Component             c,
    const DealIIFunction<dim> &        f)
  {
    this->bc_handler->add_boundary_condition(id,
                                             bc_type,
                                             c,
                                             f.get_dealii_function());
  }



  template <int dim>
  void
  BoundaryConditionHandler<dim>::add_boundary_condition(
    const dealii::types::boundary_id   id,
    const Ddhdg::BoundaryConditionType bc_type,
    const Ddhdg::Component             c,
    const std::string &                f)
  {
    this->add_boundary_condition(id, bc_type, c, AnalyticFunction<dim>(f));
  }



  template <int dim>
  bool
  BoundaryConditionHandler<dim>::has_dirichlet_boundary_conditions() const
  {
    return this->bc_handler->has_dirichlet_boundary_conditions();
  }



  template <int dim>
  bool
  BoundaryConditionHandler<dim>::has_neumann_boundary_conditions() const
  {
    return this->bc_handler->has_neumann_boundary_conditions();
  }



  template <int dim>
  Problem<dim>::Problem(const double                      left,
                        const double                      right,
                        HomogeneousPermittivity<dim> &    permittivity,
                        HomogeneousElectronMobility<dim> &n_electron_mobility,
                        HomogeneousElectronMobility<dim> &p_electron_mobility,
                        RecombinationTerm<dim> &          recombination_term,
                        DealIIFunction<dim> &             temperature,
                        DealIIFunction<dim> &             doping,
                        BoundaryConditionHandler<dim> &   bc_handler,
                        const double conduction_band_density,
                        const double valence_band_density,
                        const double conduction_band_edge_energy,
                        const double valence_band_edge_energy)
    : ddhdg_problem(std::make_shared<Ddhdg::HomogeneousProblem<dim>>(
        generate_triangulation(left, right),
        permittivity.generate_ddhdg_permittivity(),
        n_electron_mobility.generate_ddhdg_electron_mobility(),
        p_electron_mobility.generate_ddhdg_electron_mobility(),
        recombination_term.generate_ddhdg_recombination_term(),
        temperature.get_dealii_function(),
        doping.get_dealii_function(),
        bc_handler.get_ddhdg_boundary_condition_handler(),
        conduction_band_density,
        valence_band_density,
        conduction_band_edge_energy,
        valence_band_edge_energy))
  {}



  template <int dim>
  Problem<dim>::Problem(const Problem<dim> &problem)
    : ddhdg_problem(problem.ddhdg_problem)
  {}



  template <int dim>
  std::shared_ptr<const Ddhdg::HomogeneousProblem<dim>>
  Problem<dim>::get_ddhdg_problem() const
  {
    return this->ddhdg_problem;
  }



  template <int dim>
  std::shared_ptr<dealii::Triangulation<dim>>
  Problem<dim>::generate_triangulation(const double left, const double right)
  {
    std::shared_ptr<dealii::Triangulation<dim>> triangulation =
      std::make_shared<dealii::Triangulation<dim>>();

    dealii::Point<dim>        p1, p2;
    std::vector<unsigned int> subdivisions(dim);
    for (unsigned int i = 0; i < dim; ++i)
      {
        p1[i]           = left;
        p2[i]           = right;
        subdivisions[i] = 1;
      }

    dealii::GridGenerator::subdivided_hyper_rectangle(
      *triangulation, subdivisions, p1, p2, true);

    return triangulation;
  }



  ErrorPerCell::ErrorPerCell(const unsigned int size)
  {
    this->data_vector = std::make_shared<dealii::Vector<float>>(size);
  }



  ErrorPerCell::ErrorPerCell(const ErrorPerCell &other)
  {
    this->data_vector = other.data_vector;
  }



  template <int dim>
  NPSolver<dim>::NPSolver(const Problem<dim> &             problem,
                          const Ddhdg::NPSolverParameters &parameters,
                          const Ddhdg::Adimensionalizer &  adimensionalizer,
                          const bool                       verbose)
    : ddhdg_solver(
        std::make_shared<Ddhdg::NPSolver<dim, Ddhdg::HomogeneousProblem<dim>>>(
          problem.get_ddhdg_problem(),
          std::make_shared<const Ddhdg::NPSolverParameters>(parameters),
          std::make_shared<const Ddhdg::Adimensionalizer>(adimensionalizer),
          verbose))
  {}



  template <int dim>
  void
  NPSolver<dim>::set_verbose(const bool verbose)
  {
    if (verbose)
      this->ddhdg_solver->log_standard_level =
        Ddhdg::Logging::severity_level::info;
    else
      this->ddhdg_solver->log_standard_level =
        Ddhdg::Logging::severity_level::debug;
  }



  template <int dim>
  void
  NPSolver<dim>::refine_grid(const unsigned int i, const bool preserve_solution)
  {
    this->ddhdg_solver->refine_grid(i, preserve_solution);
  }



  template <int dim>
  void
  NPSolver<dim>::refine_and_coarsen_fixed_fraction(
    const ErrorPerCell error_per_cell,
    const double       top_fraction,
    const double       bottom_fraction,
    const unsigned int max_n_cells)
  {
    this->ddhdg_solver->refine_and_coarsen_fixed_fraction(
      *(error_per_cell.data_vector),
      top_fraction,
      bottom_fraction,
      max_n_cells);
  }



  template <int dim>
  unsigned int
  NPSolver<dim>::n_of_triangulation_levels() const
  {
    return this->ddhdg_solver->n_of_triangulation_levels();
  }


  template <int dim>
  unsigned int
  NPSolver<dim>::get_n_dofs(const bool for_trace) const
  {
    return this->ddhdg_solver->get_n_dofs(for_trace);
  }


  template <int dim>
  unsigned int
  NPSolver<dim>::get_n_active_cells() const
  {
    return this->ddhdg_solver->get_n_active_cells();
  }


  template <int dim>
  void
  NPSolver<dim>::get_cell_vertices(double vertices[]) const
  {
    const unsigned int vertices_per_cell =
      dealii::GeometryInfo<dim>::vertices_per_cell;
    dealii::Point<dim> *p;
    unsigned int        cell_number = 0;
    for (const auto &cell :
         this->ddhdg_solver->dof_handler_cell.active_cell_iterators())
      {
        for (unsigned int v = 0; v < vertices_per_cell; v++)
          {
            p = &(cell->vertex(v));
            for (unsigned int i = 0; i < dim; i++)
              {
                const double       p_i = (*p)[i];
                const unsigned int k =
                  cell_number * (vertices_per_cell * dim) + v * dim + i;
                vertices[k] = p_i;
              }
          }
        ++cell_number;
      }
  }



  template <int dim>
  void
  NPSolver<dim>::set_component(const Ddhdg::Component c,
                               const std::string &    f,
                               const bool             use_projection)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> c_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    c_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      f,
      Ddhdg::Constants::constants);
    this->ddhdg_solver->set_component(c, c_function, use_projection);
  }



  template <int dim>
  void
  NPSolver<dim>::set_component(const Ddhdg::Component    c,
                               const DealIIFunction<dim> f,
                               const bool                use_projection)
  {
    this->ddhdg_solver->set_component(c,
                                      f.get_dealii_function(),
                                      use_projection);
  }



  template <int dim>
  void
  NPSolver<dim>::set_current_solution(const std::string &v_f,
                                      const std::string &n_f,
                                      const std::string &p_f,
                                      const bool         use_projection)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> v_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> n_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> p_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    v_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      v_f,
      Ddhdg::Constants::constants);
    n_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      n_f,
      Ddhdg::Constants::constants);
    p_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      p_f,
      Ddhdg::Constants::constants);
    this->ddhdg_solver->set_current_solution(v_function,
                                             n_function,
                                             p_function,
                                             use_projection);
  }



  template <int dim>
  void
  NPSolver<dim>::set_multithreading(const bool multithreading)
  {
    this->ddhdg_solver->set_multithreading(multithreading);
  }



  template <int dim>
  bool
  NPSolver<dim>::is_enabled(Ddhdg::Component c) const
  {
    return this->ddhdg_solver->is_enabled(c);
  }



  template <int dim>
  void
  NPSolver<dim>::enable_component(Ddhdg::Component c)
  {
    this->ddhdg_solver->enable_component(c);
  }



  template <int dim>
  void
  NPSolver<dim>::disable_component(Ddhdg::Component c)
  {
    this->ddhdg_solver->disable_component(c);
  }



  template <int dim>
  void
  NPSolver<dim>::set_enabled_components(const bool V_enabled,
                                        const bool n_enabled,
                                        const bool p_enabled)
  {
    this->ddhdg_solver->set_enabled_components(V_enabled, n_enabled, p_enabled);
  }



  template <int dim>
  void
  NPSolver<dim>::copy_triangulation_from(NPSolver<dim> other)
  {
    this->ddhdg_solver->copy_triangulation_from(*(other.ddhdg_solver));
  }



  template <int dim>
  void
  NPSolver<dim>::copy_solution_from(NPSolver<dim> other)
  {
    this->ddhdg_solver->copy_solution_from(*(other.ddhdg_solver));
  }



  template <int dim>
  Ddhdg::NPSolverParameters
  NPSolver<dim>::get_parameters() const
  {
    return *(this->ddhdg_solver->parameters);
  }



  template <int dim>
  Ddhdg::NonlinearIterationResults
  NPSolver<dim>::run()
  {
    return this->ddhdg_solver->run();
  }



  template <int dim>
  void
  NPSolver<dim>::compute_local_charge_neutrality_on_trace(
    const bool only_at_boundary)
  {
    return this->ddhdg_solver->compute_local_charge_neutrality_on_trace(
      only_at_boundary);
  }



  template <int dim>
  void
  NPSolver<dim>::compute_local_charge_neutrality()
  {
    return this->ddhdg_solver->compute_local_charge_neutrality();
  }



  template <int dim>
  Ddhdg::NonlinearIterationResults
  NPSolver<dim>::compute_thermodynamic_equilibrium(bool generate_first_guess)
  {
    return this->ddhdg_solver->compute_thermodynamic_equilibrium(
      generate_first_guess);
  }



  template <int dim>
  Ddhdg::NonlinearIterationResults
  NPSolver<dim>::compute_thermodynamic_equilibrium(double absolute_tol,
                                                   double relative_tol,
                                                   int max_number_of_iterations,
                                                   bool generate_first_guess)
  {
    return this->ddhdg_solver->compute_thermodynamic_equilibrium(
      absolute_tol,
      relative_tol,
      max_number_of_iterations,
      generate_first_guess);
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_error_per_cell(const Ddhdg::Component c) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(c,
                                                *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_l2_error_per_cell(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Component    c) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      expected_solution.get_dealii_function(),
      c,
      dealii::VectorTools::NormType::L2_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_l2_error_per_cell(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Displacement d) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      expected_solution.get_dealii_function(),
      d,
      dealii::VectorTools::NormType::L2_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_l2_error_per_cell(const NPSolver<dim>    other,
                                            const Ddhdg::Component c) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      *(other.ddhdg_solver),
      c,
      dealii::VectorTools::NormType::L2_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_l2_error_per_cell(const NPSolver<dim>       other,
                                            const Ddhdg::Displacement d) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      *(other.ddhdg_solver),
      d,
      dealii::VectorTools::NormType::L2_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_h1_error_per_cell(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Component    c) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      expected_solution.get_dealii_function(),
      c,
      dealii::VectorTools::NormType::H1_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_h1_error_per_cell(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Displacement d) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      expected_solution.get_dealii_function(),
      d,
      dealii::VectorTools::NormType::H1_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_h1_error_per_cell(const NPSolver<dim>    other,
                                            const Ddhdg::Component c) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      *(other.ddhdg_solver),
      c,
      dealii::VectorTools::NormType::H1_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_h1_error_per_cell(const NPSolver<dim>       other,
                                            const Ddhdg::Displacement d) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      *(other.ddhdg_solver),
      d,
      dealii::VectorTools::NormType::H1_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_linfty_error_per_cell(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Component    c) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      expected_solution.get_dealii_function(),
      c,
      dealii::VectorTools::NormType::Linfty_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_linfty_error_per_cell(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Displacement d) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      expected_solution.get_dealii_function(),
      d,
      dealii::VectorTools::NormType::Linfty_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_linfty_error_per_cell(const NPSolver<dim>    other,
                                                const Ddhdg::Component c) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      *(other.ddhdg_solver),
      c,
      dealii::VectorTools::NormType::Linfty_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_linfty_error_per_cell(
    const NPSolver<dim>       other,
    const Ddhdg::Displacement d) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      *(other.ddhdg_solver),
      d,
      dealii::VectorTools::NormType::Linfty_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(const DealIIFunction<dim> expected_solution,
                                   const Ddhdg::Component    c) const
  {
    return this->ddhdg_solver->estimate_l2_error(
      expected_solution.get_dealii_function(), c);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(const DealIIFunction<dim> expected_solution,
                                   const Ddhdg::Displacement d) const
  {
    return this->ddhdg_solver->estimate_l2_error(
      expected_solution.get_dealii_function(), d);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(const NPSolver<dim>    solver,
                                   const Ddhdg::Component c) const
  {
    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              c,
                                              dealii::VectorTools::L2_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(const NPSolver<dim>       solver,
                                   const Ddhdg::Displacement d) const
  {
    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              d,
                                              dealii::VectorTools::L2_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(const DealIIFunction<dim> expected_solution,
                                   const Ddhdg::Component    c) const
  {
    return this->ddhdg_solver->estimate_h1_error(
      expected_solution.get_dealii_function(), c);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(const DealIIFunction<dim> expected_solution,
                                   const Ddhdg::Displacement d) const
  {
    return this->ddhdg_solver->estimate_h1_error(
      expected_solution.get_dealii_function(), d);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(const NPSolver<dim>    solver,
                                   const Ddhdg::Component c) const
  {
    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              c,
                                              dealii::VectorTools::H1_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(const NPSolver<dim>       solver,
                                   const Ddhdg::Displacement d) const
  {
    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              d,
                                              dealii::VectorTools::H1_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Component    c) const
  {
    return this->ddhdg_solver->estimate_linfty_error(
      expected_solution.get_dealii_function(), c);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Displacement d) const
  {
    return this->ddhdg_solver->estimate_linfty_error(
      expected_solution.get_dealii_function(), d);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(const NPSolver<dim>    solver,
                                       const Ddhdg::Component c) const
  {
    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              c,
                                              dealii::VectorTools::Linfty_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(const NPSolver<dim>       solver,
                                       const Ddhdg::Displacement d) const
  {
    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              d,
                                              dealii::VectorTools::Linfty_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error_on_trace(
    const std::string &    expected_solution,
    const Ddhdg::Component c) const
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_solution,
      Ddhdg::Constants::constants);
    return this->ddhdg_solver->estimate_l2_error_on_trace(expected_solution_f,
                                                          c);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error_on_trace(
    const std::string &    expected_solution,
    const Ddhdg::Component c) const
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_solution,
      Ddhdg::Constants::constants);
    return this->ddhdg_solver->estimate_linfty_error_on_trace(
      expected_solution_f, c);
  }



  template <int dim>
  DealIIFunction<dim>
  NPSolver<dim>::get_solution(const Ddhdg::Component c) const
  {
    return DealIIFunction(this->ddhdg_solver->get_solution(c));
  }



  template <int dim>
  double
  NPSolver<dim>::get_solution_on_a_point(const dealii::Point<dim> p,
                                         const Ddhdg::Component   c) const
  {
    return this->ddhdg_solver->get_solution_on_a_point(p, c);
  }



  template <int dim>
  void
  NPSolver<dim>::output_results(const std::string &solution_filename,
                                const bool         save_update) const
  {
    this->ddhdg_solver->output_results(solution_filename, save_update);
  }



  template <int dim>
  void
  NPSolver<dim>::output_results(const std::string &solution_filename,
                                const std::string &trace_filename,
                                const bool         save_update) const
  {
    this->ddhdg_solver->output_results(solution_filename,
                                       trace_filename,
                                       save_update);
  }



  template <int dim>
  void
  NPSolver<dim>::print_convergence_table(const std::string &expected_V_solution,
                                         const std::string &expected_n_solution,
                                         const std::string &expected_p_solution,
                                         const unsigned int n_cycles,
                                         const unsigned int initial_refinements)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_V_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> expected_n_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> expected_p_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_V_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_V_solution,
      Ddhdg::Constants::constants);
    expected_n_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_n_solution,
      Ddhdg::Constants::constants);
    expected_p_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_p_solution,
      Ddhdg::Constants::constants);
    this->ddhdg_solver->print_convergence_table(
      std::make_shared<Ddhdg::ConvergenceTable>(dim),
      expected_V_solution_f,
      expected_n_solution_f,
      expected_p_solution_f,
      n_cycles,
      initial_refinements);
  }



  template <int dim>
  void
  NPSolver<dim>::print_convergence_table(const std::string &expected_V_solution,
                                         const std::string &expected_n_solution,
                                         const std::string &expected_p_solution,
                                         const std::string &initial_V_function,
                                         const std::string &initial_n_function,
                                         const std::string &initial_p_function,
                                         const unsigned int n_cycles,
                                         const unsigned int initial_refinements)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_V_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> expected_n_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> expected_p_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> initial_V_function_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> initial_n_function_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> initial_p_function_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_V_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_V_solution,
      Ddhdg::Constants::constants);
    expected_n_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_n_solution,
      Ddhdg::Constants::constants);
    expected_p_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_p_solution,
      Ddhdg::Constants::constants);
    initial_V_function_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      initial_V_function,
      Ddhdg::Constants::constants);
    initial_n_function_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      initial_n_function,
      Ddhdg::Constants::constants);
    initial_p_function_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      initial_p_function,
      Ddhdg::Constants::constants);
    this->ddhdg_solver->print_convergence_table(
      std::make_shared<Ddhdg::ConvergenceTable>(dim),
      expected_V_solution_f,
      expected_n_solution_f,
      expected_p_solution_f,
      initial_V_function_f,
      initial_n_function_f,
      initial_p_function_f,
      n_cycles,
      initial_refinements);
  }



  template class HomogeneousPermittivity<1>;
  template class HomogeneousPermittivity<2>;
  template class HomogeneousPermittivity<3>;

  template class HomogeneousElectronMobility<1>;
  template class HomogeneousElectronMobility<2>;
  template class HomogeneousElectronMobility<3>;

  template class DealIIFunction<1>;
  template class DealIIFunction<2>;
  template class DealIIFunction<3>;

  template class AnalyticFunction<1>;
  template class AnalyticFunction<2>;
  template class AnalyticFunction<3>;

  template class PiecewiseFunction<1>;
  template class PiecewiseFunction<2>;
  template class PiecewiseFunction<3>;

  template class LinearRecombinationTerm<1>;
  template class LinearRecombinationTerm<2>;
  template class LinearRecombinationTerm<3>;

  template class BoundaryConditionHandler<1>;
  template class BoundaryConditionHandler<2>;
  template class BoundaryConditionHandler<3>;

  template class Problem<1>;
  template class Problem<2>;
  template class Problem<3>;

  template class NPSolver<1>;
  template class NPSolver<2>;
  template class NPSolver<3>;
} // namespace pyddhdg
