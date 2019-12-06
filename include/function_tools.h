#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/function_derivative.h>

namespace Ddhdg
{
  DeclExceptionMsg(FunctionMustBeScalar,
                   "The submitted function must be scalar");


  template <int dim>
  class FunctionByComponents : public dealii::Function<dim>
  {
  public:
    FunctionByComponents(
      int n_of_components,
      const std::map<unsigned int,
                     const std::shared_ptr<const dealii::Function<dim>>>
        components)
      : dealii::Function<dim>(n_of_components)
      , component_map(components)
    {}

    double
    value(const dealii::Point<dim> &p,
          unsigned int              component = 0) const override;

    dealii::Tensor<1, dim>
    gradient(const dealii::Point<dim> &p,
             unsigned int              component = 0) const override;


  private:
    const std::map<unsigned int,
                   const std::shared_ptr<const dealii::Function<dim>>>
      component_map;
  };


  template <int dim>
  class ComponentFunction : public dealii::Function<dim>
  {
  public:
    ComponentFunction(const dealii::Function<dim> &f, unsigned int component)
      : dealii::Function<dim>(1)
      , component(component)
      , f(f)
    {}

    double
    value(const dealii::Point<dim> &p,
          unsigned int              component = 0) const override
    {
      Assert(component == 0, dealii::ExcIndexRange(component, 0, 1));
      return f.value(p, this->component);
    }

    dealii::Tensor<1, dim>
    gradient(const dealii::Point<dim> &p,
             unsigned int              component = 0) const override
    {
      Assert(component == 0, dealii::ExcIndexRange(component, 0, 1));
      return f.gradient(p, this->component);
    }


  private:
    const unsigned int           component;
    const dealii::Function<dim> &f;
  };

  template <int dim>
  class Gradient : public dealii::Function<dim>
  {
  public:
    explicit Gradient(const dealii::Function<dim> &function);

    double
    value(const dealii::Point<dim> &p,
          unsigned int              component = 0) const override;

    dealii::Tensor<1, dim>
    gradient(const dealii::Point<dim> &p,
             unsigned int              component = 0) const override;

  private:
    const dealii::Function<dim> &                  f;
    std::map<int, dealii::FunctionDerivative<dim>> partial_derivatives;
  };
} // namespace Ddhdg
