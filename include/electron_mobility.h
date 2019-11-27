#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace Ddhdg
{
  template <int dim>
  class ElectronMobility
  {
  public:
    virtual dealii::Tensor<2, dim>
    compute_electron_mobility(const dealii::Point<dim> &q) const = 0;

    virtual void
    compute_electron_mobility(const std::vector<dealii::Point<dim>> &P,
                              std::vector<dealii::Tensor<2, dim>> &  mu) const;
  };

  template <int dim>
  class HomogeneousElectronMobility : public ElectronMobility<dim>
  {
  public:
    HomogeneousElectronMobility(double mu)
      : mu(mu)
    {}

    const double mu;

    virtual dealii::Tensor<2, dim>
    compute_electron_mobility(const dealii::Point<dim> &q) const;


    virtual void
    compute_electron_mobility(const std::vector<dealii::Point<dim>> &P,
                              std::vector<dealii::Tensor<2, dim>> &  mu) const;
  };
} // namespace Ddhdg
