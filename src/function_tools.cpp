#include "function_tools.h"


namespace Ddhdg
{
  template <int dim>
  FunctionByComponents<dim>::FunctionByComponents(
    int n_of_components,
    const std::map<unsigned int,
                   const std::shared_ptr<const dealii::Function<dim>>>
      components)
    : dealii::Function<dim>(n_of_components)
    , components(components)
  {
    for (auto const &x : this->components)
      {
        (void)x;
        Assert(x.first < this->n_components,
               dealii::ExcIndexRange(x.first, 0, this->n_components));
        Assert(x.second->n_components == 1, FunctionMustBeScalar());
      }
  }



  template <int dim>
  double
  FunctionByComponents<dim>::value(const dealii::Point<dim> &p,
                                   const unsigned int        component) const
  {
    Assert(component < this->n_components,
           dealii::ExcIndexRange(component, 0, this->n_components));
    auto component_function = components.find(component);
    if (component_function != components.end())
      return component_function->second->value(p, 0.);
    return 0.;
  }



  template <int dim>
  void
  FunctionByComponents<dim>::vector_value(const dealii::Point<dim> &p,
                                          dealii::Vector<double> &values) const
  {
    Assert(values.size() == this->n_components,
           dealii::ExcDimensionMismatch(values.size(), this->n_components));
    values = 0;
    for (auto const &x : this->components)
      {
        values[x.first] = x.second->value(p);
      }
  }



  template <int dim>
  void
  FunctionByComponents<dim>::value_list(
    const std::vector<dealii::Point<dim>> &p,
    std::vector<double> &                  values,
    const unsigned int                     component) const
  {
    Assert(values.size() == p.size(),
           dealii::ExcDimensionMismatch(values.size(), p.size()));
    Assert(component < this->n_components,
           dealii::ExcIndexRange(component, 0, this->n_components));
    auto component_function = components.find(component);
    if (component_function != components.end())
      component_function->second->value_list(p, values);
    else
      for (unsigned int i = 0; i < values.size(); i++)
        values[i] = 0;
  }



  template <int dim>
  dealii::Tensor<1, dim>
  FunctionByComponents<dim>::gradient(const dealii::Point<dim> &p,
                                      const unsigned int        component) const
  {
    Assert(component < this->n_components,
           dealii::ExcIndexRange(component, 0, this->n_components));
    auto component_function = components.find(component);
    if (component_function != components.end())
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
  void
  FunctionByComponents<dim>::vector_gradient(
    const dealii::Point<dim> &           p,
    std::vector<dealii::Tensor<1, dim>> &values) const
  {
    Assert(values.size() == this->n_components,
           dealii::ExcDimensionMismatch(values.size(), this->n_components));
    for (unsigned int i = 0; i < this->n_components; i++)
      {
        auto component_function = components.find(i);
        if (component_function != components.end())
          values[i] = component_function->second->gradient(p, 0.);
        else
          for (int j = 0; j < dim; j++)
            values[i][j] = 0.;
      }
  }



  template <int dim>
  void
  FunctionByComponents<dim>::gradient_list(
    const std::vector<dealii::Point<dim>> &p,
    std::vector<dealii::Tensor<1, dim>> &  values,
    const unsigned int                     component) const
  {
    Assert(values.size() == p.size(),
           dealii::ExcDimensionMismatch(values.size(), p.size()));
    Assert(component < this->n_components,
           dealii::ExcIndexRange(component, 0, this->n_components));
    auto component_function = components.find(component);
    if (component_function != components.end())
      component_function->second->gradient_list(p, values);
    else
      for (unsigned int i = 0; i < values.size(); i++)
        for (unsigned int j = 0; j < dim; j++)
          values[i][j] = 0;
  }



  template <int dim>
  ComponentFunction<dim>::ComponentFunction(
    const std::shared_ptr<const dealii::Function<dim>> f,
    const unsigned int                                 component)
    : dealii::Function<dim>(1)
    , f_component(component)
    , f(f)
  {
    Assert(component < f->n_components,
           dealii::ExcIndexRange(component, 0, f->n_components));
  }



  template <int dim>
  double
  ComponentFunction<dim>::value(const dealii::Point<dim> &p,
                                const unsigned int        component) const
  {
    (void)component;
    Assert(component == 0, dealii::ExcIndexRange(component, 0, 1));
    return f->value(p, this->f_component);
  }



  template <int dim>
  void
  ComponentFunction<dim>::vector_value(const dealii::Point<dim> &p,
                                       dealii::Vector<double> &  values) const
  {
    Assert(values.size() == this->n_components,
           dealii::ExcDimensionMismatch(values.size(), this->n_components));
    values[0] = f->value(p, this->f_component);
  }



  template <int dim>
  void
  ComponentFunction<dim>::value_list(const std::vector<dealii::Point<dim>> &p,
                                     std::vector<double> &values,
                                     const unsigned int   component) const
  {
    (void)component;
    Assert(values.size() == p.size(),
           dealii::ExcDimensionMismatch(values.size(), p.size()));
    Assert(component < this->n_components,
           dealii::ExcIndexRange(component, 0, this->n_components));
    this->f->value_list(p, values, this->f_component);
  }



  template <int dim>
  dealii::Tensor<1, dim>
  ComponentFunction<dim>::gradient(const dealii::Point<dim> &p,
                                   const unsigned int        component) const
  {
    (void)component;
    Assert(component == 0, dealii::ExcIndexRange(component, 0, 1));
    return f->gradient(p, this->f_component);
  }



  template <int dim>
  void
  ComponentFunction<dim>::vector_gradient(
    const dealii::Point<dim> &           p,
    std::vector<dealii::Tensor<1, dim>> &values) const
  {
    Assert(values.size() == this->n_components,
           dealii::ExcDimensionMismatch(values.size(), this->n_components));
    values[0] = this->f->gradient(p, this->f_component);
  }



  template <int dim>
  void
  ComponentFunction<dim>::gradient_list(
    const std::vector<dealii::Point<dim>> &p,
    std::vector<dealii::Tensor<1, dim>> &  values,
    const unsigned int                     component) const
  {
    (void)component;
    Assert(values.size() == p.size(),
           dealii::ExcDimensionMismatch(values.size(), p.size()));
    Assert(component < this->n_components,
           dealii::ExcIndexRange(component, 0, this->n_components));
    this->f->gradient_list(p, values, this->f_component);
  }



  template <int dim>
  Gradient<dim>::Gradient(
    const std::shared_ptr<const dealii::Function<dim>> function)
    : FunctionByComponents<dim>(dim, build_component_map(function))
    , f(function)
  {
    Assert(function->n_components == 1, FunctionMustBeScalar());
  }



  template <int dim>
  std::map<unsigned int, const std::shared_ptr<const dealii::Function<dim>>>
  Gradient<dim>::build_component_map(
    const std::shared_ptr<const dealii::Function<dim>> function)
  {
    std::map<unsigned int, const std::shared_ptr<const dealii::Function<dim>>>
      partial_derivatives;
    switch (dim)
      {
          case 1: {
            const dealii::Point<dim> p0{1};
            partial_derivatives.insert(
              {0,
               std::make_shared<dealii::FunctionDerivative<dim>>(*function,
                                                                 p0)});
            break;
          }
          case 2: {
            const dealii::Point<dim> p0{1, 0};
            const dealii::Point<dim> p1{0, 1};
            partial_derivatives.insert(
              {0,
               std::make_shared<dealii::FunctionDerivative<dim>>(*function,
                                                                 p0)});
            partial_derivatives.insert(
              {1,
               std::make_shared<dealii::FunctionDerivative<dim>>(*function,
                                                                 p1)});
            break;
          }
          case 3: {
            const dealii::Point<dim> p0{1, 0, 0};
            const dealii::Point<dim> p1{0, 1, 0};
            const dealii::Point<dim> p2{0, 0, 1};
            partial_derivatives.insert(
              {0,
               std::make_shared<dealii::FunctionDerivative<dim>>(*function,
                                                                 p0)});
            partial_derivatives.insert(
              {1,
               std::make_shared<dealii::FunctionDerivative<dim>>(*function,
                                                                 p1)});
            partial_derivatives.insert(
              {2,
               std::make_shared<dealii::FunctionDerivative<dim>>(*function,
                                                                 p2)});
            break;
          }
        default:
          break;
      }
    return partial_derivatives;
  }



  template <int dim>
  Opposite<dim>::Opposite(
    const std::shared_ptr<const dealii::Function<dim>> function)
    : dealii::Function<dim>(function->n_components)
    , f(function)
  {}



  template <int dim>
  double
  Opposite<dim>::value(const dealii::Point<dim> &p,
                       unsigned int              component) const
  {
    Assert(component < this->n_components,
           dealii::ExcIndexRange(component, 0, this->n_components));
    return -this->f->value(p, component);
  }



  template <int dim>
  void
  Opposite<dim>::vector_value(const dealii::Point<dim> &p,
                              dealii::Vector<double> &  values) const
  {
    Assert(values.size() == this->n_components,
           dealii::ExcDimensionMismatch(values.size(), this->n_components));
    this->f->vector_value(p, values);
    for (unsigned int i = 0; i < values.size(); i++)
      values[i] *= -1;
  }



  template <int dim>
  void
  Opposite<dim>::value_list(const std::vector<dealii::Point<dim>> &p,
                            std::vector<double> &                  values,
                            const unsigned int component) const
  {
    Assert(values.size() == p.size(),
           dealii::ExcDimensionMismatch(values.size(), p.size()));
    Assert(component < this->n_components,
           dealii::ExcIndexRange(component, 0, this->n_components));
    this->f->value_list(p, values, component);
    for (unsigned int i = 0; i < values.size(); i++)
      values[i] *= -1;
  }



  template <int dim>
  dealii::Tensor<1, dim>
  Opposite<dim>::gradient(const dealii::Point<dim> &p,
                          unsigned int              component) const
  {
    Assert(component < this->n_components,
           dealii::ExcIndexRange(component, 0, this->n_components));
    return -this->f->gradient(p, component);
  }



  template <int dim>
  void
  Opposite<dim>::vector_gradient(
    const dealii::Point<dim> &           p,
    std::vector<dealii::Tensor<1, dim>> &values) const
  {
    Assert(values.size() == this->n_components,
           dealii::ExcDimensionMismatch(values.size(), this->n_components));
    this->f->vector_gradient(p, values);
    for (unsigned int i = 0; i < values.size(); i++)
      values[i] *= -1;
  }



  template <int dim>
  void
  Opposite<dim>::gradient_list(const std::vector<dealii::Point<dim>> &p,
                               std::vector<dealii::Tensor<1, dim>> &  values,
                               const unsigned int component) const
  {
    Assert(values.size() == p.size(),
           dealii::ExcDimensionMismatch(values.size(), p.size()));
    Assert(component < this->n_components,
           dealii::ExcIndexRange(component, 0, this->n_components));
    this->f->gradient_list(p, values, component);
    for (unsigned int i = 0; i < values.size(); i++)
      values[i] *= -1;
  }



  template <int dim>
  PiecewiseFunction<dim>::PiecewiseFunction(
    const std::shared_ptr<const dealii::Function<dim>> condition,
    const std::shared_ptr<const dealii::Function<dim>> f1,
    const std::shared_ptr<const dealii::Function<dim>> f2)
    : dealii::Function<dim>(f1->n_components)
    , condition(condition)
    , f1(f1)
    , f2(f2)
  {
    Assert(this->condition->n_components == 1, FunctionMustBeScalar());
    Assert(this->f1->n_components == this->f2->n_components,
           dealii::ExcDimensionMismatch(this->f1->n_components,
                                        this->f2->n_components));
  }



  template <int dim>
  double
  PiecewiseFunction<dim>::value(const dealii::Point<dim> &p,
                                unsigned int              component) const
  {
    Assert(component < this->n_components,
           dealii::ExcIndexRange(component, 0, this->n_components));
    const double condition_evaluation = this->condition->value(p);
    if (condition_evaluation >= 0.)
      return this->f1->value(p, component);
    return -this->f2->value(p, component);
  }



  template <int dim>
  void
  PiecewiseFunction<dim>::vector_value(const dealii::Point<dim> &p,
                                       dealii::Vector<double> &  values) const
  {
    Assert(values.size() == this->n_components,
           dealii::ExcDimensionMismatch(values.size(), this->n_components));
    const double condition_evaluation = this->condition->value(p);
    if (condition_evaluation >= 0.)
      this->f1->vector_value(p, values);
    else
      this->f2->vector_value(p, values);
  }



  template <int dim>
  void
  PiecewiseFunction<dim>::value_list(const std::vector<dealii::Point<dim>> &p,
                                     std::vector<double> &values,
                                     const unsigned int   component) const
  {
    Assert(values.size() == p.size(),
           dealii::ExcDimensionMismatch(values.size(), p.size()));
    Assert(component < this->n_components,
           dealii::ExcIndexRange(component, 0, this->n_components));

    const unsigned int  n_of_points = p.size();
    std::vector<double> condition_evaluation(n_of_points);
    this->condition->value_list(p, condition_evaluation);

    for (unsigned int i = 0; i < n_of_points; i++)
      {
        if (condition_evaluation[i] > 0)
          values[i] = this->f1->value(p[i], component);
        else
          values[i] = this->f2->value(p[i], component);
      }
  }



  template <int dim>
  dealii::Tensor<1, dim>
  PiecewiseFunction<dim>::gradient(const dealii::Point<dim> &p,
                                   unsigned int              component) const
  {
    Assert(component < this->n_components,
           dealii::ExcIndexRange(component, 0, this->n_components));
    const double condition_evaluation = this->condition->value(p);
    if (condition_evaluation >= 0.)
      return this->f1->gradient(p, component);
    return -this->f2->gradient(p, component);
  }



  template <int dim>
  void
  PiecewiseFunction<dim>::vector_gradient(
    const dealii::Point<dim> &           p,
    std::vector<dealii::Tensor<1, dim>> &values) const
  {
    Assert(values.size() == this->n_components,
           dealii::ExcDimensionMismatch(values.size(), this->n_components));
    const double condition_evaluation = this->condition->value(p);
    if (condition_evaluation >= 0.)
      this->f1->vector_gradient(p, values);
    else
      this->f2->vector_gradient(p, values);
  }



  template <int dim>
  void
  PiecewiseFunction<dim>::gradient_list(
    const std::vector<dealii::Point<dim>> &p,
    std::vector<dealii::Tensor<1, dim>> &  values,
    const unsigned int                     component) const
  {
    Assert(values.size() == p.size(),
           dealii::ExcDimensionMismatch(values.size(), p.size()));
    Assert(component < this->n_components,
           dealii::ExcIndexRange(component, 0, this->n_components));

    const unsigned int  n_of_points = p.size();
    std::vector<double> condition_evaluation(n_of_points);
    this->condition->value_list(p, condition_evaluation);

    for (unsigned int i = 0; i < n_of_points; i++)
      {
        if (condition_evaluation[i] > 0)
          values[i] = this->f1->gradient(p[i], component);
        else
          values[i] = this->f2->gradient(p[i], component);
      }
  }



  template class FunctionByComponents<1>;
  template class FunctionByComponents<2>;
  template class FunctionByComponents<3>;

  template class ComponentFunction<1>;
  template class ComponentFunction<2>;
  template class ComponentFunction<3>;

  template class Gradient<1>;
  template class Gradient<2>;
  template class Gradient<3>;

  template class Opposite<1>;
  template class Opposite<2>;
  template class Opposite<3>;

  template class PiecewiseFunction<1>;
  template class PiecewiseFunction<2>;
  template class PiecewiseFunction<3>;

} // namespace Ddhdg
