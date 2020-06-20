#pragma once

#include <deal.II/base/function.h>

#include "components.h"
#include "constants.h"
#include "dof_types.h"
#include "function_tools.h"

namespace Ddhdg
{
  class Adimensionalizer
  {
  public:
    explicit Adimensionalizer(double scale_length          = 1,
                              double temperature_magnitude = Constants::Q /
                                                             Constants::KB,
                              double doping_magnitude            = 1,
                              double electron_mobility_magnitude = 1);

    Adimensionalizer(const Adimensionalizer &adm) = default;

    template <Component c>
    [[nodiscard]] inline double
    get_component_rescaling_factor() const
    {
      switch (c)
        {
          case Component::V:
            return this->temperature_magnitude * Constants::KB / Constants::Q;
          case Component::n:
            return this->doping_magnitude;
          case Component::p:
            return this->doping_magnitude;
          default:
            Assert(false, InvalidComponent());
            return 1;
        }
    }

    [[nodiscard]] double
    get_component_rescaling_factor(Component c) const;

    [[nodiscard]] inline double
    get_poisson_equation_density_constant() const
    {
      // Using Constants::Q, you do not rescale the Poisson equation
      // return Constants::Q;
      return 1.;
    }

    [[nodiscard]] inline double
    get_current_equation_constant() const
    {
      // Using this value, you do not rescale the drift-diffusion equations
      // return Constants::KB * this->temperature_magnitude *
      //        this->electron_mobility_magnitude * this->doping_magnitude /
      //        (this->scale_length * Constants::Q) * 1000;
      return 1.;
    }

    template <Component c>
    [[nodiscard]] inline double
    get_neumann_boundary_condition_rescaling_factor() const
    {
      switch (c)
        {
          case Component::V:
            return Constants::Q / this->get_poisson_equation_density_constant();
            case Component::n: {
              const double num = Constants::KB * this->doping_magnitude *
                                 this->temperature_magnitude *
                                 this->electron_mobility_magnitude;
              const double den =
                this->scale_length * this->get_current_equation_constant();
              return num / den;
            }
            case Component::p: {
              const double num = Constants::KB * this->doping_magnitude *
                                 this->temperature_magnitude *
                                 this->electron_mobility_magnitude;
              const double den =
                this->scale_length * this->get_current_equation_constant();
              return num / den;
            }
          default:
            Assert(false, InvalidComponent());
            return 1.;
        }
    }

    [[nodiscard]] double
    get_neumann_boundary_condition_rescaling_factor(Component c) const;

    template <Component c>
    void
    adimensionalize_component(const std::vector<double> &source,
                              std::vector<double> &      dest) const;

    template <Component c>
    void
    redimensionalize_component(const std::vector<double> &source,
                               std::vector<double> &      dest) const;

    template <Component c>
    void
    inplace_adimensionalize_component(std::vector<double> &data) const;

    template <Component c>
    void
    inplace_redimensionalize_component(std::vector<double> &data) const;

    void
    adimensionalize_component(const std::vector<double> &source,
                              Component                  c,
                              std::vector<double> &      dest) const;

    void
    redimensionalize_component(const std::vector<double> &source,
                               Component                  c,
                               std::vector<double> &      dest) const;

    void
    inplace_adimensionalize_component(std::vector<double> &data,
                                      Component            c) const;

    void
    inplace_redimensionalize_component(std::vector<double> &data,
                                       Component            c) const;

    void
    adimensionalize_dof_vector(
      const dealii::Vector<double> &dof_vector,
      const std::vector<Component> &dof_to_component_map,
      const std::vector<DofType> &  dof_to_dof_type,
      dealii::Vector<double> &      rescaled_vector) const;

    void
    redimensionalize_dof_vector(
      const dealii::Vector<double> &dof_vector,
      const std::vector<Component> &dof_to_component_map,
      const std::vector<DofType> &  dof_to_dof_type,
      dealii::Vector<double> &      rescaled_vector) const;

    template <int dim>
    std::shared_ptr<dealii::Function<dim>>
    adimensionalize_component_function(
      std::shared_ptr<const dealii::Function<dim>> f,
      Component                                    c) const;

    template <int dim>
    std::shared_ptr<dealii::Function<dim>>
    adimensionalize_doping_function(
      std::shared_ptr<const dealii::Function<dim>> doping) const;

    [[nodiscard]] inline double
    get_permittivity_rescaling_factor() const
    {
      const double den = this->temperature_magnitude * Constants::KB *
                         this->get_poisson_equation_density_constant();
      const double num = this->scale_length * this->scale_length *
                         Constants::Q * Constants::Q * this->doping_magnitude;

      return num / den;
    }

    [[nodiscard]] inline double
    get_thermal_voltage_rescaling_factor() const
    {
      return this->temperature_magnitude * Constants::KB / Constants::Q;
    }

    [[nodiscard]] inline double
    get_mobility_rescaling_factor() const
    {
      return this->electron_mobility_magnitude /
             this->get_current_equation_constant();
    }

    template <int dim>
    void
    adimensionalize_electron_mobility(
      std::vector<dealii::Tensor<2, dim>> &data) const
    {
      const double mobility_rescaling_factor =
        this->get_mobility_rescaling_factor();
      const unsigned int n_of_elements = data.size();
      for (unsigned int i = 0; i < n_of_elements; i++)
        data[i] /= mobility_rescaling_factor;
    }

    template <int dim>
    void
    adimensionalize_hole_mobility(
      std::vector<dealii::Tensor<2, dim>> &data) const
    {
      const double mobility_rescaling_factor =
        this->get_mobility_rescaling_factor();
      const unsigned int n_of_elements = data.size();
      for (unsigned int i = 0; i < n_of_elements; i++)
        data[i] /= mobility_rescaling_factor;
    }

    [[nodiscard]] inline double
    get_recombination_rescaling_factor() const
    {
      const double num = Constants::KB * this->doping_magnitude *
                         this->temperature_magnitude *
                         this->electron_mobility_magnitude;
      const double den =
        this->scale_length * this->get_current_equation_constant();
      return num / den;
    }

    void
    adimensionalize_recombination_term(std::vector<double> &r,
                                       std::vector<double> &dr_n,
                                       std::vector<double> &dr_p) const;

    template <Component c>
    [[nodiscard]] inline double
    adimensionalize_tau(const double tau) const
    {
      return tau * this->scale_length;
    }

    [[nodiscard]] double
    adimensionalize_tau(const double tau, Component c) const;

    const double scale_length;
    const double temperature_magnitude;
    const double doping_magnitude;
    const double electron_mobility_magnitude;
  };

} // namespace Ddhdg
