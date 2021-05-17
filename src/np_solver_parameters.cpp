#include "np_solver_parameters.h"

namespace Ddhdg
{
  FixedTauComputer::FixedTauComputer(
    const std::map<Component, double> &tau_vals,
    const Adimensionalizer &           adimensionalizer)
    : V_tau(tau_vals.at(Component::V))
    , n_tau(tau_vals.at(Component::n))
    , p_tau(tau_vals.at(Component::p))
    , V_rescaling_factor(
        adimensionalizer.get_tau_rescaling_factor<Component::V>())
    , n_rescaling_factor(
        adimensionalizer.get_tau_rescaling_factor<Component::n>())
    , p_rescaling_factor(
        adimensionalizer.get_tau_rescaling_factor<Component::p>())
    , V_tau_rescaled(this->V_tau / this->V_rescaling_factor)
    , n_tau_rescaled(this->n_tau / this->n_rescaling_factor)
    , p_tau_rescaled(this->p_tau / this->p_rescaling_factor)
  {}

  NPSolverParameters::NPSolverParameters(
    const unsigned int                               V_degree,
    const unsigned int                               n_degree,
    const unsigned int                               p_degree,
    const std::shared_ptr<NonlinearSolverParameters> nonlinear_parameters,
    const bool                                       iterative_linear_solver,
    const bool                                       multithreading,
    const DDFluxType                                 dd_flux_type,
    const bool                                       phi_linearize)
    : degree{{Component::V, V_degree},
             {Component::n, n_degree},
             {Component::p, p_degree}}
    , nonlinear_parameters(nonlinear_parameters)
    , iterative_linear_solver(iterative_linear_solver)
    , multithreading(multithreading)
    , dd_flux_type(dd_flux_type)
    , phi_linearize(phi_linearize)
  {}

  FixedTauNPSolverParameters::FixedTauNPSolverParameters(
    const unsigned int                               V_degree,
    const unsigned int                               n_degree,
    const unsigned int                               p_degree,
    const std::shared_ptr<NonlinearSolverParameters> nonlinear_parameters,
    const double                                     V_tau,
    const double                                     n_tau,
    const double                                     p_tau,
    const bool                                       iterative_linear_solver,
    const bool                                       multithreading,
    const DDFluxType                                 dd_flux_type,
    const bool                                       phi_linearize)
    : NPSolverParameters(V_degree,
                         n_degree,
                         p_degree,
                         nonlinear_parameters,
                         iterative_linear_solver,
                         multithreading,
                         dd_flux_type,
                         phi_linearize)
    , tau{{Component::V, V_tau}, {Component::n, n_tau}, {Component::p, p_tau}}
  {}

} // namespace Ddhdg
