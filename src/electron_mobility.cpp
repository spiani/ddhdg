#include "electron_mobility.h"

namespace Ddhdg
{
  template <int dim>
  void
  ElectronMobility<dim>::compute_electron_mobility(
    const std::vector<dealii::Point<dim>> &P,
    std::vector<dealii::Tensor<2, dim>> &  mu) const
  {
    const std::size_t n_of_points = P.size();

    Assert(n_of_points == mu.size(),
           dealii::ExcDimensionMismatch(n_of_points, mu.size()));

    for (std::size_t q = 0; q < n_of_points; q++)
      {
        const dealii::Point<dim> p = P[q];
        mu[q]                      = compute_electron_mobility(p);
      }
  }



  template <int dim>
  dealii::Tensor<2, dim>
  HomogeneousElectronMobility<dim>::compute_electron_mobility(
    const dealii::Point<dim> &q) const
  {
    (void)q;
    dealii::Tensor<2, dim> mu;
    for (unsigned int i = 0; i < dim; i++)
      for (unsigned int j = 0; j < dim; j++)
        mu[i][j] = 0.;
    for (unsigned int i = 0; i < dim; i++)
      mu[i][i] = this->mu;
    return mu;
  }



  template <int dim>
  void
  HomogeneousElectronMobility<dim>::compute_electron_mobility(
    const std::vector<dealii::Point<dim>> &P,
    std::vector<dealii::Tensor<2, dim>> &  mu) const
  {
    const std::size_t n_of_points = P.size();

    Assert(n_of_points == mu.size(),
           dealii::ExcDimensionMismatch(n_of_points, mu.size()));

    for (std::size_t q = 0; q < n_of_points; q++)
      {
        for (unsigned int i = 0; i < dim; i++)
          for (unsigned int j = 0; j < dim; j++)
            mu[q][i][j] = 0.;
        for (unsigned int i = 0; i < dim; i++)
          mu[q][i][i] = this->mu;
      }
  }



  template class ElectronMobility<1>;
  template class ElectronMobility<2>;
  template class ElectronMobility<3>;

  template class HomogeneousElectronMobility<1>;
  template class HomogeneousElectronMobility<2>;
  template class HomogeneousElectronMobility<3>;

} // namespace Ddhdg
