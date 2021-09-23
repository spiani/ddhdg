#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include "adimensionalizer.h"

namespace Ddhdg
{
  template <int dim>
  class HomogeneousPermittivityComputer;

  template <int dim>
  class HomogeneousPermittivity
  {
  public:
    using PermittivityComputer = HomogeneousPermittivityComputer<dim>;

    explicit HomogeneousPermittivity(double epsilon)
      : epsilon(epsilon)
    {}

    PermittivityComputer
    get_computer(const Adimensionalizer &adimensionalizer) const
    {
      return PermittivityComputer(*this, adimensionalizer);
    }

    PermittivityComputer
    get_computer(const double rescaling_factor) const
    {
      return PermittivityComputer(*this, rescaling_factor);
    }

    const double epsilon;
  };


  template <int dim>
  class HomogeneousPermittivityComputer
  {
  public:
    HomogeneousPermittivityComputer(
      const HomogeneousPermittivity<dim> &permittivity,
      const Adimensionalizer             &adimensionalizer)
      : epsilon(permittivity.epsilon)
      , rescaling_factor(adimensionalizer.get_permittivity_rescaling_factor())
      , rescaled_epsilon(epsilon / rescaling_factor)
    {}

    HomogeneousPermittivityComputer(
      const HomogeneousPermittivity<dim> &permittivity,
      const double                        rescaling_factor)
      : epsilon(permittivity.epsilon)
      , rescaling_factor(rescaling_factor)
      , rescaled_epsilon(epsilon / rescaling_factor)
    {}

    inline void
    initialize_on_cell(const std::vector<dealii::Point<dim>> &)
    {}

    inline void
    initialize_on_face(const std::vector<dealii::Point<dim>> &)
    {}

    inline void
    epsilon_operator_on_cell(const unsigned int,
                             const dealii::Tensor<1, dim> &v,
                             dealii::Tensor<1, dim>       &w)
    {
      for (unsigned int i = 0; i < dim; i++)
        w[i] = v[i] * this->rescaled_epsilon;
    }

    inline void
    epsilon_operator_on_face(const unsigned int,
                             const dealii::Tensor<1, dim> &v,
                             dealii::Tensor<1, dim>       &w)
    {
      for (unsigned int i = 0; i < dim; i++)
        w[i] = v[i] * this->rescaled_epsilon;
    }

    const double epsilon;
    const double rescaling_factor;
    const double rescaled_epsilon;
  };
} // namespace Ddhdg
