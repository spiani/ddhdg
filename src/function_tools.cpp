#include "function_tools.h"


namespace Ddhdg
{
  template <int dim>
  double
  FunctionByComponents<dim>::value(const dealii::Point<dim> &p,
                                   const unsigned int        component) const
  {
    auto component_function = component_map.find(component);
    if (component_function != component_map.end())
      {
        return component_function->second->value(p, 0.);
      }
    else
      {
        return 0.;
      }
  }



  template <int dim>
  dealii::Tensor<1, dim>
  FunctionByComponents<dim>::gradient(const dealii::Point<dim> &p,
                                      const unsigned int        component) const
  {
    auto component_function = component_map.find(component);
    if (component_function != component_map.end())
      {
        return component_function->second->gradient(p, 0.);
      }
    else
      {
        dealii::Tensor<1, dim> zeros;
        for (int i = 0; i < dim; i++)
          zeros[i] = 0.;
        return zeros;
      }
  }


  template <int dim>
  Gradient<dim>::Gradient(const dealii::Function<dim> &function)
    : dealii::Function<dim>(dim)
    , f(function)
  {
    Assert(function.n_components == 1, FunctionMustBeScalar());
    switch (dim)
      {
          case 1: {
            const dealii::Point<dim> p0{1};
            partial_derivatives.insert(
              {0, dealii::FunctionDerivative(this->f, p0)});
            break;
          }
          case 2: {
            const dealii::Point<dim> p0{1, 0};
            const dealii::Point<dim> p1{0, 1};
            partial_derivatives.insert(
              {0, dealii::FunctionDerivative(this->f, p0)});
            partial_derivatives.insert(
              {1, dealii::FunctionDerivative(this->f, p1)});
            break;
          }
          case 3: {
            const dealii::Point<dim> p0{1, 0, 0};
            const dealii::Point<dim> p1{0, 1, 0};
            const dealii::Point<dim> p2{0, 0, 1};
            partial_derivatives.insert(
              {0, dealii::FunctionDerivative(this->f, p0)});
            partial_derivatives.insert(
              {1, dealii::FunctionDerivative(this->f, p1)});
            partial_derivatives.insert(
              {2, dealii::FunctionDerivative(this->f, p2)});
            break;
          }
        default:
          break;
      }
  }



  template <int dim>
  double
  Gradient<dim>::value(const dealii::Point<dim> &p,
                       unsigned int              component) const
  {
    return this->partial_derivatives.at(component).value(p);
  }


  template <int dim>
  dealii::Tensor<1, dim>
  Gradient<dim>::gradient(const dealii::Point<dim> &p,
    unsigned int              component) const
  {
    return this->partial_derivatives.at(component).gradient(p);
  }

  template class FunctionByComponents<1>;
  template class FunctionByComponents<2>;
  template class FunctionByComponents<3>;

  template class Gradient<1>;
  template class Gradient<2>;
  template class Gradient<3>;

} // namespace Ddhdg
