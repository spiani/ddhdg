#include <deal.II/grid/grid_generator.h>

#include <ddhdg.h>
#include <gtest/gtest.h>

constexpr double domain_size = 1.;


template <typename D>
class ProjectTrace : public Ddhdg::NPSolver<D::value>, public ::testing::Test
{
public:
  ProjectTrace()
    : Ddhdg::NPSolver<D::value>(
        get_problem(),
        std::make_shared<Ddhdg::NPSolverParameters>(3, 1, 2),
        std::make_shared<Ddhdg::Adimensionalizer>(1,
                                                  Ddhdg::Constants::Q /
                                                    Ddhdg::Constants::KB,
                                                  1,
                                                  1)){};

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
                                      false);
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
    return std::make_shared<dealii::Functions::ZeroFunction<D::value>>();
  }

  static std::shared_ptr<Ddhdg::BoundaryConditionHandler<D::value>>
  get_boundary_conditions()
  {
    const unsigned int dim = D::value;

    std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>> boundary_handler =
      std::make_shared<Ddhdg::BoundaryConditionHandler<dim>>();

    return boundary_handler;
  }

  static std::shared_ptr<Ddhdg::Problem<D::value>>
  get_problem()
  {
    const unsigned int dim = D::value;

    std::shared_ptr<Ddhdg::Problem<dim>> problem =
      std::make_shared<Ddhdg::Problem<dim>>(
        get_triangulation(),
        std::make_shared<const Ddhdg::HomogeneousPermittivity<dim>>(1.),
        std::make_shared<const Ddhdg::HomogeneousElectronMobility<dim>>(1.),
        std::make_shared<const Ddhdg::HomogeneousElectronMobility<dim>>(1.),
        std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>("0", "0", "0"),
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


TYPED_TEST_CASE(ProjectTrace, dimensions);


TYPED_TEST(ProjectTrace, project_trace) // NOLINT
{
  constexpr double   TOLERANCE = 1e-12;
  const unsigned int dim       = TypeParam::value;

  this->set_multithreading(false);
  this->refine_grid(5 - dim);

  if (!this->initialized)
    this->setup_overall_system();

  std::shared_ptr<dealii::Function<dim>> V_function;
  if (dim == 1)
    V_function = TestFixture::get_function("x^3");
  else
    V_function = TestFixture::get_function("x^3*y^2");

  std::shared_ptr<dealii::Function<dim>> n_function;
  if (dim == 1)
    n_function = TestFixture::get_function("-x");
  else
    n_function = TestFixture::get_function("-x + y");

  std::shared_ptr<dealii::Function<dim>> p_function;
  if (dim == 1)
    p_function = TestFixture::get_function("x^2");
  else
    p_function = TestFixture::get_function("x^2 + y^2");

  this->set_current_solution(V_function, n_function, p_function, true);

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
