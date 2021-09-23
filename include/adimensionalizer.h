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
    explicit constexpr Adimensionalizer(
      double scale_length          = 1,
      double temperature_magnitude = Constants::Q / Constants::KB,
      double doping_magnitude      = 1,
      double mobility_magnitude    = 1)
      : scale_length(scale_length)
      , temperature_magnitude(temperature_magnitude)
      , doping_magnitude(doping_magnitude)
      , mobility_magnitude(mobility_magnitude)
    {}

    constexpr Adimensionalizer(const Adimensionalizer &adm) = default;

    template <Component c>
    [[nodiscard]] constexpr double
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

    template <Displacement d>
    [[nodiscard]] constexpr double
    get_displacement_rescaling_factor() const
    {
      const Component c = displacement2component(d);
      return this->get_component_rescaling_factor<c>() / this->scale_length;
    }

    [[nodiscard]] double
    get_displacement_rescaling_factor(Displacement d) const;

    template <Component c>
    [[nodiscard]] constexpr double
    get_neumann_boundary_condition_rescaling_factor() const
    {
      switch (c)
        {
          case Component::V:
            return Constants::Q;
            case Component::n: {
              const double num = Constants::KB * this->doping_magnitude *
                                 this->temperature_magnitude *
                                 this->mobility_magnitude;
              const double den = this->scale_length * this->scale_length;
              return num / den;
            }
            case Component::p: {
              const double num = Constants::KB * this->doping_magnitude *
                                 this->temperature_magnitude *
                                 this->mobility_magnitude;
              const double den = this->scale_length * this->scale_length;
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
                              std::vector<double>       &dest) const;

    template <Component c>
    void
    redimensionalize_component(const std::vector<double> &source,
                               std::vector<double>       &dest) const;

    template <Component c>
    void
    inplace_adimensionalize_component(std::vector<double> &data) const;

    template <Component c>
    void
    inplace_redimensionalize_component(std::vector<double> &data) const;

    void
    adimensionalize_component(const std::vector<double> &source,
                              Component                  c,
                              std::vector<double>       &dest) const;

    void
    redimensionalize_component(const std::vector<double> &source,
                               Component                  c,
                               std::vector<double>       &dest) const;

    void
    inplace_adimensionalize_component(std::vector<double> &data,
                                      Component            c) const;

    void
    inplace_redimensionalize_component(std::vector<double> &data,
                                       Component            c) const;

    template <int dim>
    void
    adimensionalize_displacement(
      const std::vector<dealii::Tensor<1, dim>> &source,
      Displacement                               d,
      std::vector<dealii::Tensor<1, dim>>       &dest) const;

    template <int dim>
    void
    redimensionalize_displacement(
      const std::vector<dealii::Tensor<1, dim>> &source,
      Displacement                               d,
      std::vector<dealii::Tensor<1, dim>>       &dest) const;

    template <int dim>
    void
    inplace_adimensionalize_displacement(
      std::vector<dealii::Tensor<1, dim>> &data,
      Displacement                         d) const;

    template <int dim>
    void
    inplace_redimensionalize_displacement(
      std::vector<dealii::Tensor<1, dim>> &data,
      Displacement                         d) const;

    void
    adimensionalize_dof_vector(
      const dealii::Vector<double> &dof_vector,
      const std::vector<Component> &dof_to_component_map,
      const std::vector<DofType>   &dof_to_dof_type,
      dealii::Vector<double>       &rescaled_vector) const;

    void
    redimensionalize_dof_vector(
      const dealii::Vector<double> &dof_vector,
      const std::vector<Component> &dof_to_component_map,
      const std::vector<DofType>   &dof_to_dof_type,
      dealii::Vector<double>       &rescaled_vector) const;

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
      const double den = this->temperature_magnitude * Constants::KB;
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
      return this->mobility_magnitude;
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
                         this->temperature_magnitude * this->mobility_magnitude;
      const double den = this->scale_length * this->scale_length * Constants::Q;
      return num / den;
    }

    void
    adimensionalize_recombination_term(dealii::Vector<double> &r) const;

    void
    adimensionalize_recombination_term(std::vector<double> &r,
                                       std::vector<double> &dr_n,
                                       std::vector<double> &dr_p) const;

    template <Component c>
    [[nodiscard]] inline double
    get_tau_rescaling_factor() const
    {
      double Tv = this->temperature_magnitude * Constants::KB / Constants::Q;
      const double k = this->doping_magnitude;
      switch (c)
        {
          case (Component::V):
            return Constants::Q * k / Tv;
          case (Component::n):
            return (Tv * this->mobility_magnitude) /
                   (this->scale_length * this->scale_length);
          case (Component::p):
            return (Tv * this->mobility_magnitude) /
                   (this->scale_length * this->scale_length);
            default: {
              Assert(false, InvalidComponent());
              return 1.;
            }
        }
    }

    [[nodiscard]] double
    get_tau_rescaling_factor(Component c) const;

    const double scale_length;
    const double temperature_magnitude;
    const double doping_magnitude;
    const double mobility_magnitude;
  };

} // namespace Ddhdg
