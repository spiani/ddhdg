#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace Ddhdg
{
  template <int dim>
  class Permittivity
  {
  public:
    virtual dealii::Tensor<2, dim>
    compute_absolute_permittivity(const dealii::Point<dim> &q) const = 0;

    virtual void
    compute_absolute_permittivity(
      const std::vector<dealii::Point<dim>> &P,
      std::vector<dealii::Tensor<2, dim>> &  epsilon) const;
  };

  template <int dim>
  class HomogeneousPermittivity : public Permittivity<dim>
  {
  public:
    HomogeneousPermittivity(double epsilon)
      : epsilon0(epsilon)
    {}

    const double epsilon0;

    virtual dealii::Tensor<2, dim>
    compute_absolute_permittivity(const dealii::Point<dim> &q) const;


    virtual void
    compute_absolute_permittivity(
      const std::vector<dealii::Point<dim>> &P,
      std::vector<dealii::Tensor<2, dim>> &  epsilon) const;
  };
} // namespace Ddhdg
