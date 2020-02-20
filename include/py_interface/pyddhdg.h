#pragma once

#include <deal.II/grid/grid_generator.h>

#include "boundary_conditions.h"
#include "ddhdg.h"

namespace pyddhdg
{
  template <int dim>
  class Permittivity
  {
  public:
    virtual ~Permittivity() = default;

    virtual std::shared_ptr<Ddhdg::Permittivity<dim>>
    generate_ddhdg_permittivity() = 0;
  };

  template <int dim>
  class HomogeneousPermittivity : public Permittivity<dim>
  {
  public:
    explicit HomogeneousPermittivity(double epsilon);

    virtual std::shared_ptr<Ddhdg::Permittivity<dim>>
    generate_ddhdg_permittivity();

    const double epsilon;
  };

  template <int dim>
  class ElectronMobility
  {
  public:
    virtual ~ElectronMobility() = default;

    virtual std::shared_ptr<Ddhdg::ElectronMobility<dim>>
    generate_ddhdg_electron_mobility() = 0;
  };

  template <int dim>
  class HomogeneousElectronMobility : public ElectronMobility<dim>
  {
  public:
    explicit HomogeneousElectronMobility(double mu);

    virtual std::shared_ptr<Ddhdg::ElectronMobility<dim>>
    generate_ddhdg_electron_mobility();

    const double mu;
  };

  template <int dim>
  class PythonFunction
  {
  public:
    explicit PythonFunction(std::string f_exp);

    std::shared_ptr<dealii::Function<dim>>
    get_dealii_function() const;

    [[nodiscard]] std::string
    get_expression() const;

  private:
    std::string                                        f_expr;
    const std::shared_ptr<dealii::FunctionParser<dim>> f;
  };

  template <int dim>
  class Temperature : public PythonFunction<dim>
  {
  public:
    explicit Temperature(const std::string &f_expr);
  };

  template <int dim>
  class Doping : public PythonFunction<dim>
  {
  public:
    explicit Doping(const std::string &f_expr);
  };

  template <int dim>
  class RecombinationTerm
  {
  public:
    virtual ~RecombinationTerm() = default;

    virtual std::shared_ptr<Ddhdg::RecombinationTerm<dim>>
    generate_ddhdg_recombination_term() = 0;
  };

  template <int dim>
  class LinearRecombinationTerm : public RecombinationTerm<dim>
  {
  public:
    LinearRecombinationTerm(const PythonFunction<dim> &zero_term,
                            const PythonFunction<dim> &first_term);

    virtual std::shared_ptr<Ddhdg::RecombinationTerm<dim>>
    generate_ddhdg_recombination_term();

    [[nodiscard]] std::string
    get_constant_term() const;

    [[nodiscard]] std::string
    get_linear_coefficient() const;

  private:
    const PythonFunction<dim> zero_term;
    const PythonFunction<dim> first_term;
  };

  template <int dim>
  class BoundaryConditionHandler
  {
  public:
    BoundaryConditionHandler();

    std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>>
    get_ddhdg_boundary_condition_handler();

    void
    add_boundary_condition_from_function(dealii::types::boundary_id   id,
                                         Ddhdg::BoundaryConditionType bc_type,
                                         Ddhdg::Component             c,
                                         const PythonFunction<dim> &  f);

    void
    add_boundary_condition_from_string(dealii::types::boundary_id   id,
                                       Ddhdg::BoundaryConditionType bc_type,
                                       Ddhdg::Component             c,
                                       const std::string &          f);

    [[nodiscard]] bool
    has_dirichlet_boundary_conditions() const;

    [[nodiscard]] bool
    has_neumann_boundary_conditions() const;

  private:
    const std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>> bc_handler;
  };

  template <int dim>
  class Problem
  {
  public:
    Problem(Permittivity<dim> &            permittivity,
            ElectronMobility<dim> &        electron_mobility,
            RecombinationTerm<dim> &       recombination_term,
            Temperature<dim> &             temperature,
            Doping<dim> &                  doping,
            BoundaryConditionHandler<dim> &bc_handler);

  private:
    static std::shared_ptr<dealii::Triangulation<dim>>
    generate_triangulation();

    const std::shared_ptr<const Ddhdg::Problem<dim>> ddhdg_problem;
  };

} // namespace pyddhdg
