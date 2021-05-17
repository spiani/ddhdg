#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <map>
#include <memory>

#include "adimensionalizer.h"
#include "components.h"

namespace Ddhdg
{
  enum DDFluxType
  {
    use_cell,
    use_trace,
    qiu_shi_stabilization
  };

  enum TauComputerType
  {
    not_implemented = -1,
    fixed_tau_computer,
  };

  struct NonlinearSolverParameters
  {
    explicit NonlinearSolverParameters(double absolute_tolerance       = 1e-10,
                                       double relative_tolerance       = 1e-10,
                                       int    max_number_of_iterations = 100,
                                       double alpha                    = 1.)
      : absolute_tolerance(absolute_tolerance)
      , relative_tolerance(relative_tolerance)
      , max_number_of_iterations(max_number_of_iterations)
      , alpha(alpha)
    {}

    double absolute_tolerance;
    double relative_tolerance;
    int    max_number_of_iterations;
    double alpha;
  };

  class TauComputer
  {
  public:
    virtual std::unique_ptr<TauComputer>
    make_copy() const = 0;

    virtual TauComputerType
    get_tau_computer_type() const
    {
      return TauComputerType::not_implemented;
    }

    virtual ~TauComputer(){};
  };

  class FixedTauComputer : public TauComputer
  {
  public:
    FixedTauComputer(const std::map<Component, double> &tau_vals,
                     const Adimensionalizer &           adimensionalizer);

    FixedTauComputer(const FixedTauComputer &fixed_tau_computer) = default;

    std::unique_ptr<TauComputer>
    make_copy() const override
    {
      return std::make_unique<FixedTauComputer>(*this);
    }

    TauComputerType
    get_tau_computer_type() const override
    {
      return TauComputerType::fixed_tau_computer;
    }

    template <int dim, Component c>
    inline void
    compute_tau(
      const std::vector<dealii::Point<dim>> quadrature_points,
      const typename dealii::DoFHandler<dim, dim>::cell_iterator &cell,
      unsigned int                                                face,
      std::vector<double> &                                       tau) const
    {
      Assert(c == Component::V || c == Component::n || c == Component::p,
             InvalidComponent());

      const unsigned int n_of_points = quadrature_points.size();

      (void)cell;
      (void)face;

      const double tau_val = (c == Component::V) ? this->V_tau_rescaled :
                             (c == Component::n) ? this->n_tau_rescaled :
                             (c == Component::p) ? this->p_tau_rescaled :
                                                   0.;

      for (unsigned int i = 0; i < n_of_points; ++i)
        tau[i] = tau_val;
    }

    template <int dim>
    inline void
    compute_tau(
      const std::vector<dealii::Point<dim>> quadrature_points,
      const typename dealii::DoFHandler<dim, dim>::cell_iterator &cell,
      unsigned int                                                face,
      Component                                                   c,
      std::vector<double> &                                       tau) const
    {
      Assert(c == Component::V || c == Component::n || c == Component::p,
             InvalidComponent());

      if (c == Component::V)
        this->template compute_tau<dim, Component::V>(quadrature_points,
                                                      cell,
                                                      face,
                                                      tau);
      if (c == Component::n)
        this->template compute_tau<dim, Component::n>(quadrature_points,
                                                      cell,
                                                      face,
                                                      tau);
      if (c == Component::p)
        this->template compute_tau<dim, Component::p>(quadrature_points,
                                                      cell,
                                                      face,
                                                      tau);
    }

    const double V_tau;
    const double n_tau;
    const double p_tau;

    const double V_rescaling_factor;
    const double n_rescaling_factor;
    const double p_rescaling_factor;

    const double V_tau_rescaled;
    const double n_tau_rescaled;
    const double p_tau_rescaled;
  };

  class NPSolverParameters
  {
  public:
    explicit NPSolverParameters(
      unsigned int                               V_degree,
      unsigned int                               n_degree,
      unsigned int                               p_degree,
      std::shared_ptr<NonlinearSolverParameters> nonlinear_parameters,
      bool                                       iterative_linear_solver,
      bool                                       multithreading,
      DDFluxType                                 dd_flux_type,
      bool                                       phi_linearize);

    virtual std::unique_ptr<NPSolverParameters>
    make_unique_copy() const = 0;

    virtual std::shared_ptr<NPSolverParameters>
    make_shared_copy() const = 0;

    virtual std::unique_ptr<TauComputer>
    get_tau_computer(const Adimensionalizer &adimensionalizer) const = 0;

    virtual TauComputerType
    get_tau_computer_type() const
    {
      return TauComputerType::not_implemented;
    }

    virtual ~NPSolverParameters()
    {}

    const std::map<Component, unsigned int> degree;

    const std::shared_ptr<NonlinearSolverParameters> nonlinear_parameters;

    bool       iterative_linear_solver;
    bool       multithreading;
    DDFluxType dd_flux_type;
    bool       phi_linearize;
  };


  class FixedTauNPSolverParameters : public NPSolverParameters
  {
  public:
    explicit FixedTauNPSolverParameters(
      unsigned int                               V_degree = 1,
      unsigned int                               n_degree = 1,
      unsigned int                               p_degree = 1,
      std::shared_ptr<NonlinearSolverParameters> nonlinear_parameters =
        std::make_shared<NonlinearSolverParameters>(),
      double     V_tau                   = 1.,
      double     n_tau                   = 1.,
      double     p_tau                   = 1.,
      bool       iterative_linear_solver = false,
      bool       multithreading          = true,
      DDFluxType dd_flux_type            = DDFluxType::use_cell,
      bool       phi_linearize           = false);

    std::unique_ptr<NPSolverParameters>
    make_unique_copy() const override
    {
      return std::make_unique<FixedTauNPSolverParameters>(*this);
    }

    std::shared_ptr<NPSolverParameters>
    make_shared_copy() const override
    {
      return std::make_shared<FixedTauNPSolverParameters>(*this);
    }

    std::unique_ptr<TauComputer>
    get_tau_computer(const Adimensionalizer &adimensionalizer) const override
    {
      return std::make_unique<FixedTauComputer>(this->tau, adimensionalizer);
    }

    TauComputerType
    get_tau_computer_type() const override
    {
      return TauComputerType::fixed_tau_computer;
    }

    [[nodiscard]] double
    get_tau(const Component c) const
    {
      return this->tau.at(c);
    }

  private:
    const std::map<Component, double> tau;
  };

} // namespace Ddhdg
