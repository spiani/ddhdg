#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include "adimensionalizer.h"

namespace Ddhdg
{
  template <int dim>
  class HomogeneousMobilityComputer;

  template <int dim>
  class HomogeneousMobility
  {
  public:
    using MobilityComputer = HomogeneousMobilityComputer<dim>;

    HomogeneousMobility(double mu, Component c)
      : mu(mu)
      , cmp(c)
    {}

    MobilityComputer
    get_computer(const Adimensionalizer &adimensionalizer) const
    {
      return MobilityComputer(*this, adimensionalizer);
    }

    MobilityComputer
    get_computer(const double rescaling_factor) const
    {
      return MobilityComputer(*this, rescaling_factor);
    }

    const double    mu;
    const Component cmp;
  };

  template <int dim>
  class HomogeneousMobilityComputer
  {
  public:
    HomogeneousMobilityComputer(const HomogeneousMobility<dim> &mobility,
                                const Adimensionalizer &adimensionalizer)
      : mu(mobility.mu)
      , rescaling_factor(adimensionalizer.get_mobility_rescaling_factor())
      , rescaled_mu(mu / rescaling_factor)
    {}

    HomogeneousMobilityComputer(const HomogeneousMobility<dim> &mobility,
                                const double rescaling_factor)
      : mu(mobility.mu)
      , rescaling_factor(rescaling_factor)
      , rescaled_mu(mu / rescaling_factor)
    {}

    inline void
    initialize_on_cell(const std::vector<dealii::Point<dim>> &)
    {}

    inline void
    initialize_on_face(const std::vector<dealii::Point<dim>> &)
    {}

    inline void
    mu_operator_on_cell(const unsigned int,
                        const dealii::Tensor<1, dim> &v,
                        dealii::Tensor<1, dim> &      w) const
    {
      for (unsigned int i = 0; i < dim; i++)
        w[i] = v[i] * this->rescaled_mu;
    }

    inline void
    mu_operator_on_face(const unsigned int,
                        const dealii::Tensor<1, dim> &v,
                        dealii::Tensor<1, dim> &      w) const
    {
      for (unsigned int i = 0; i < dim; i++)
        w[i] = v[i] * this->rescaled_mu;
    }

    const double mu;
    const double rescaling_factor;
    const double rescaled_mu;
  };
} // namespace Ddhdg
