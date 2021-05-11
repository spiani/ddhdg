#include <deal.II/grid/grid_generator.h>

#include <ddhdg.h>
#include <gtest/gtest.h>

constexpr double domain_size = 1.;


template <typename D>
class CopyTrace
  : public Ddhdg::NPSolver<D::value, Ddhdg::HomogeneousProblem<D::value>>,
    public ::testing::Test
{
public:
  CopyTrace()
    : Ddhdg::NPSolver<D::value, Ddhdg::HomogeneousProblem<D::value>>(
        get_problem(),
        std::make_shared<Ddhdg::NPSolverParameters>(3, 1, 2),
        std::make_shared<Ddhdg::Adimensionalizer>(1,
                                                  Ddhdg::Constants::Q /
                                                    Ddhdg::Constants::KB,
                                                  1 / Ddhdg::Constants::Q,
                                                  1),
        false){};

protected:
  static std::shared_ptr<dealii::FunctionParser<D::value>>
  get_function(const std::string &f_str)
  {
    const unsigned int dim = D::value;

    std::shared_ptr<dealii::FunctionParser<dim>> f =
      std::make_shared<dealii::FunctionParser<dim>>(1);

    f->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                  f_str,
                  Ddhdg::Constants::constants);

    return f;
  }

  static std::shared_ptr<dealii::Triangulation<D::value>>
  get_triangulation()
  {
    const unsigned int dim = D::value;

    std::shared_ptr<dealii::Triangulation<dim>> triangulation =
      std::make_shared<dealii::Triangulation<dim>>();

    dealii::GridGenerator::hyper_cube(*triangulation,
                                      -domain_size / 2.,
                                      domain_size / 2.,
                                      true);
    return triangulation;
  }

  static std::shared_ptr<dealii::Function<D::value>>
  get_temperature()
  {
    return get_function("q / kb");
  }

  static std::shared_ptr<dealii::Function<D::value>>
  get_doping()
  {
    return get_function("-2*(3*x^3*y + 3*x*y^3 - x)/q");
  }

  static std::shared_ptr<Ddhdg::BoundaryConditionHandler<D::value>>
  get_boundary_conditions()
  {
    const unsigned int dim = D::value;

    std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>> boundary_handler =
      std::make_shared<Ddhdg::BoundaryConditionHandler<dim>>();

    const std::string v_solution_str = (dim == 1) ? "x^3" : "x^3 * y^3";

    for (unsigned int i = 0; i < 6; i++)
      {
        boundary_handler->add_boundary_condition(i,
                                                 Ddhdg::dirichlet,
                                                 Ddhdg::V,
                                                 get_function(v_solution_str));
        boundary_handler->add_boundary_condition(i,
                                                 Ddhdg::dirichlet,
                                                 Ddhdg::p,
                                                 get_function("-x / q"));
      }

    boundary_handler->add_boundary_condition(0,
                                             Ddhdg::neumann,
                                             Ddhdg::n,
                                             get_function("-1"));
    boundary_handler->add_boundary_condition(1,
                                             Ddhdg::neumann,
                                             Ddhdg::n,
                                             get_function("1"));

    for (unsigned int i = 2; i < 6; i++)
      {
        boundary_handler->add_boundary_condition(i,
                                                 Ddhdg::dirichlet,
                                                 Ddhdg::n,
                                                 get_function("x / q"));
      }

    return boundary_handler;
  }

  static std::shared_ptr<Ddhdg::HomogeneousProblem<D::value>>
  get_problem()
  {
    const unsigned int dim = D::value;

    std::shared_ptr<Ddhdg::HomogeneousProblem<dim>> problem =
      std::make_shared<Ddhdg::HomogeneousProblem<dim>>(
        get_triangulation(),
        std::make_shared<const Ddhdg::HomogeneousPermittivity<dim>>(1.),
        std::make_shared<const Ddhdg::HomogeneousMobility<dim>>(
          1., Ddhdg::Component::n),
        std::make_shared<const Ddhdg::HomogeneousMobility<dim>>(
          1., Ddhdg::Component::p),
        std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>(
          "-(6*x^4*y - 9*x^2*y^3) / q", "0", "0"),
        get_temperature(),
        get_doping(),
        get_boundary_conditions(),
        1.,
        1.,
        0.,
        0.);
    return problem;
  }
};


using dimensions = ::testing::Types<std::integral_constant<unsigned int, 1>,
                                    std::integral_constant<unsigned int, 2>,
                                    std::integral_constant<unsigned int, 3>>;

TYPED_TEST_SUITE(CopyTrace, dimensions, );



TYPED_TEST(CopyTrace, project_trace_l2_average) // NOLINT
{
  constexpr double   TOLERANCE = 1e-12;
  const unsigned int dim       = TypeParam::value;

  this->set_multithreading(false);
  this->refine_grid(5 - dim, false);

  if (!this->initialized)
    this->setup_overall_system();

  std::shared_ptr<dealii::Function<dim>> V_function;
  if (dim == 1)
    V_function = TestFixture::get_function("x^3");
  else
    V_function = TestFixture::get_function("x^3*y^2");

  std::shared_ptr<dealii::Function<dim>> n_function;
  if (dim == 1)
    n_function = TestFixture::get_function("-x / q");
  else
    n_function = TestFixture::get_function("(-x + y) / q");

  std::shared_ptr<dealii::Function<dim>> p_function;
  if (dim == 1)
    p_function = TestFixture::get_function("x^2 / q");
  else
    p_function = TestFixture::get_function("(x^2 + y^2) / q");

  this->set_current_solution(V_function, n_function, p_function, false);

  // Create an array to store a copy of the trace
  dealii::Vector<double> trace_copy(this->current_solution_trace);

  // Now we completely destroy all the information of the trace
  for (unsigned int i = 0; i < this->dof_handler_trace.n_dofs(); ++i)
    this->current_solution_trace[i] = 2 / (i + 1);

  // And now we try to retrieve it back from the cells
  this->project_cell_function_on_trace();

  // Now we compute the error
  trace_copy -= this->current_solution_trace;
  const double error = trace_copy.linfty_norm();

  EXPECT_LT(error, TOLERANCE);
}



TYPED_TEST(CopyTrace, project_trace_reconstruct_solution) // NOLINT
{
  // We can not run this test for dim == 1
  if (TypeParam::value == 1)
    return;

  constexpr double   TOLERANCE = 5e-6;
  const unsigned int dim       = TypeParam::value;

  this->set_multithreading(false);
  this->refine_grid(7 - 2 * dim, false);

  if (!this->initialized)
    this->setup_overall_system();

  this->run();

  // Create an array to store a copy of the trace
  dealii::Vector<double> trace_copy(this->current_solution_trace);

  // Now we completely destroy all the information of the trace
  for (unsigned int i = 0; i < this->dof_handler_trace.n_dofs(); ++i)
    this->current_solution_trace[i] = 2 / (i + 1);

  // And now we try to retrieve it back from the cells
  this->project_cell_function_on_trace(Ddhdg::all_primary_components(),
                                       Ddhdg::reconstruct_problem_solution);

  // Now we compute the error
  trace_copy -= this->current_solution_trace;
  const double error = trace_copy.linfty_norm();

  EXPECT_LT(error, TOLERANCE);
}



TYPED_TEST(CopyTrace, project_trace_only_on_some_components) // NOLINT
{
  constexpr double   TOLERANCE = 1e-12;
  const unsigned int dim       = TypeParam::value;
  double             error;

  this->set_multithreading(false);
  this->refine_grid(5 - dim, false);

  if (!this->initialized)
    this->setup_overall_system();

  std::shared_ptr<dealii::Function<dim>> V_function;
  if (dim == 1)
    V_function = TestFixture::get_function("x^3");
  else
    V_function = TestFixture::get_function("x^3*y^2");

  std::shared_ptr<dealii::Function<dim>> n_function;
  if (dim == 1)
    n_function = TestFixture::get_function("- q * x");
  else
    n_function = TestFixture::get_function("(-x + y) / q");

  std::shared_ptr<dealii::Function<dim>> p_function;
  if (dim == 1)
    p_function = TestFixture::get_function("x^2 / q");
  else
    p_function = TestFixture::get_function("(x^2 + y^2) / q");

  this->set_current_solution(V_function, n_function, p_function, false);

  // Create an array to store a copy of the trace
  dealii::Vector<double> trace_copy(this->current_solution_trace);

  // Now we completely destroy all the information of the trace
  for (unsigned int i = 0; i < this->dof_handler_trace.n_dofs(); ++i)
    this->current_solution_trace[i] = 2 / (i + 1);

  // And now we try to retrieve it back from the cells, in several steps
  std::set<Ddhdg::Component> step1;
  std::set<Ddhdg::Component> step2;
  step1.insert(Ddhdg::Component::V);
  step1.insert(Ddhdg::Component::p);
  step2.insert(Ddhdg::Component::n);

  this->project_cell_function_on_trace(step1);
  this->project_cell_function_on_trace(step2);

  // Now we compute the error
  trace_copy -= this->current_solution_trace;
  error = trace_copy.linfty_norm();

  EXPECT_LT(error, TOLERANCE);

  trace_copy = this->current_solution_trace;

  // Again we remove the trace
  for (unsigned int i = 0; i < this->dof_handler_trace.n_dofs(); ++i)
    this->current_solution_trace[i] = 1.;

  // And now we try to retrieve it back from the cells, in several steps
  std::set<Ddhdg::Component> step3;
  std::set<Ddhdg::Component> step4;
  std::set<Ddhdg::Component> step5;
  step3.insert(Ddhdg::Component::p);
  step4.insert(Ddhdg::Component::n);
  step5.insert(Ddhdg::Component::V);

  this->project_cell_function_on_trace(step3);
  this->project_cell_function_on_trace(step4);
  this->project_cell_function_on_trace(step5);

  // Now we compute the error
  trace_copy -= this->current_solution_trace;
  error = trace_copy.linfty_norm();

  EXPECT_LT(error, TOLERANCE);
}



TYPED_TEST(CopyTrace, project_trace_local_refinement) // NOLINT
{
  constexpr double   TOLERANCE = 1e-12;
  const unsigned int dim       = TypeParam::value;

  this->set_multithreading(false);
  switch (dim)
    {
      case 2:
        this->refine_grid(2, false);
        break;
      case 3:
        this->refine_grid(1, false);
        break;
      default:
        this->refine_grid(4, false);
    }

  dealii::Vector<float> refine_flag;
  for (unsigned int j = 0; j < 5; j++)
    {
      // Refine one cell every seven
      refine_flag.reinit(this->get_n_active_cells());
      for (unsigned int i = 0; i < this->get_n_active_cells(); i++)
        refine_flag[i] = (i % 7 == 0) ? 1. : 0.;
      this->refine_and_coarsen_fixed_fraction(refine_flag, 1., 0.);
    }

  if (!this->initialized)
    this->setup_overall_system();

  std::shared_ptr<dealii::Function<dim>> V_function;
  if (dim == 1)
    V_function = TestFixture::get_function("x^3");
  else
    V_function = TestFixture::get_function("x^3*y^2");

  std::shared_ptr<dealii::Function<dim>> n_function;
  if (dim == 1)
    n_function = TestFixture::get_function("- x / q");
  else
    n_function = TestFixture::get_function("(-x + y) / q");

  std::shared_ptr<dealii::Function<dim>> p_function;
  if (dim == 1)
    p_function = TestFixture::get_function("x^2 / q");
  else
    p_function = TestFixture::get_function("(x^2 + y^2) / q");

  this->set_current_solution(V_function, n_function, p_function, false);

  // Create an array to store a copy of the trace
  dealii::Vector<double> trace_copy(this->current_solution_trace);

  // Now we completely destroy all the information of the trace
  for (unsigned int i = 0; i < this->dof_handler_trace.n_dofs(); ++i)
    this->current_solution_trace[i] = 2 / (i + 1);

  // And now we try to retrieve it back from the cells
  this->project_cell_function_on_trace();

  // Now we compute the error
  trace_copy -= this->current_solution_trace;
  const double error = trace_copy.linfty_norm();

  EXPECT_LT(error, TOLERANCE);
}
