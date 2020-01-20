#include "permittivity.h"

namespace Ddhdg
{
  template <int dim>
  void
  Permittivity<dim>::compute_absolute_permittivity(
    const std::vector<dealii::Point<dim>> &P,
    std::vector<dealii::Tensor<2, dim>> &  epsilon) const
  {
    const std::size_t n_of_points = P.size();

    Assert(n_of_points == epsilon.size(),
           dealii::ExcDimensionMismatch(n_of_points, epsilon.size()));

    for (std::size_t q = 0; q < n_of_points; q++)
      {
        const dealii::Point<dim> p = P[q];
        epsilon[q]                 = compute_absolute_permittivity(p);
      }
  }



  template <int dim>
  dealii::Tensor<2, dim>
  HomogeneousPermittivity<dim>::compute_absolute_permittivity(
    const dealii::Point<dim> &q) const
  {
    (void)q;
    dealii::Tensor<2, dim> epsilon;
    for (unsigned int i = 0; i < dim; i++)
      for (unsigned int j = 0; j < dim; j++)
        epsilon[i][j] = 0.;
    for (unsigned int i = 0; i < dim; i++)
      epsilon[i][i] = epsilon0;
    return epsilon;
  }



  template <int dim>
  void
  HomogeneousPermittivity<dim>::compute_absolute_permittivity(
    const std::vector<dealii::Point<dim>> &P,
    std::vector<dealii::Tensor<2, dim>> &  epsilon) const
  {
    const std::size_t n_of_points = P.size();

    Assert(n_of_points == epsilon.size(),
           dealii::ExcDimensionMismatch(n_of_points, epsilon.size()));

    for (std::size_t q = 0; q < n_of_points; q++)
      {
        for (unsigned int i = 0; i < dim; i++)
          for (unsigned int j = 0; j < dim; j++)
            epsilon[q][i][j] = 0.;
        for (unsigned int i = 0; i < dim; i++)
          epsilon[q][i][i] = epsilon0;
      }
  }



  template class Permittivity<1>;
  template class Permittivity<2>;
  template class Permittivity<3>;

  template class HomogeneousPermittivity<1>;
  template class HomogeneousPermittivity<2>;
  template class HomogeneousPermittivity<3>;

} // namespace Ddhdg
