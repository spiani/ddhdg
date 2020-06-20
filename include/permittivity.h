#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace Ddhdg
{
  template <int dim>
  class HomogeneousPermittivity
  {
  public:
    explicit HomogeneousPermittivity(double epsilon)
      : epsilon0(epsilon)
    {}

    const double epsilon0;
    double       rescaled_epsilon = 0;

    inline void
    initialize_on_cell(const std::vector<dealii::Point<dim>> &,
                       const double rescaling_factor)
    {
      this->rescaled_epsilon = this->epsilon0 / rescaling_factor;
    }

    inline void
    initialize_on_face(const std::vector<dealii::Point<dim>> &,
                       const double rescaling_factor)
    {
      this->rescaled_epsilon = this->epsilon0 / rescaling_factor;
    }

    inline void
    epsilon_operator_on_cell(const unsigned int,
                             const dealii::Tensor<1, dim> &v,
                             dealii::Tensor<1, dim> &      w)
    {
      for (unsigned int i = 0; i < dim; i++)
        w[i] = v[i] * this->rescaled_epsilon;
    }

    inline void
    epsilon_operator_on_face(const unsigned int,
                             const dealii::Tensor<1, dim> &v,
                             dealii::Tensor<1, dim> &      w)
    {
      for (unsigned int i = 0; i < dim; i++)
        w[i] = v[i] * this->rescaled_epsilon;
    }

    inline double
    compute_stabilized_v_tau(const unsigned int,
                             const double v_tau,
                             const dealii::Tensor<1, dim> &) const
    {
      return v_tau * this->rescaled_epsilon;
    }
  };
} // namespace Ddhdg
