#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace Ddhdg
{
  template <int dim>
  class HomogeneousElectronMobility
  {
  public:
    HomogeneousElectronMobility(double mu)
      : mu(mu)
    {}

    const double mu;
    double       rescaled_mu = 0;

    inline void
    initialize_on_cell(const std::vector<dealii::Point<dim>> &,
                       const double rescaling_factor)
    {
      this->rescaled_mu = this->mu / rescaling_factor;
    }

    inline void
    initialize_on_face(const std::vector<dealii::Point<dim>> &,
                       const double rescaling_factor)
    {
      this->rescaled_mu = this->mu / rescaling_factor;
    }

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
  };
} // namespace Ddhdg
