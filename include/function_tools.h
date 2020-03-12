#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/function_derivative.h>

#include <deal.II/lac/vector.h>

namespace Ddhdg
{
  DeclExceptionMsg(FunctionMustBeScalar,
                   "The submitted function must be scalar");
  DeclExceptionMsg(
    FunctionMustHaveDimComponents,
    "The number of components of the submitted function must be equal to the "
    "dimension of the domain");

  template <int dim>
  using component_map =
    std::map<unsigned int, const std::shared_ptr<const dealii::Function<dim>>>;

  template <int dim>
  class FunctionByComponents : public dealii::Function<dim>
  {
  public:
    FunctionByComponents(int n_of_components, component_map<dim> components);

    double
    value(const dealii::Point<dim> &p,
          unsigned int              component = 0) const override;

    void
    vector_value(const dealii::Point<dim> &p,
                 dealii::Vector<double> &  values) const override;

    void
    value_list(const std::vector<dealii::Point<dim>> &p,
               std::vector<double> &                  values,
               unsigned int component = 0) const override;

    dealii::Tensor<1, dim>
    gradient(const dealii::Point<dim> &p,
             unsigned int              component = 0) const override;

    void
    vector_gradient(const dealii::Point<dim> &           p,
                    std::vector<dealii::Tensor<1, dim>> &values) const override;

    void
    gradient_list(const std::vector<dealii::Point<dim>> &p,
                  std::vector<dealii::Tensor<1, dim>> &  values,
                  unsigned int component = 0) const override;

  private:
    component_map<dim> components;
  };


  template <int dim>
  class ComponentFunction : public dealii::Function<dim>
  {
  public:
    ComponentFunction(std::shared_ptr<const dealii::Function<dim>> f,
                      unsigned int                                 component);

    double
    value(const dealii::Point<dim> &p,
          unsigned int              component = 0) const override;

    void
    vector_value(const dealii::Point<dim> &p,
                 dealii::Vector<double> &  values) const override;

    void
    value_list(const std::vector<dealii::Point<dim>> &p,
               std::vector<double> &                  values,
               unsigned int component = 0) const override;

    dealii::Tensor<1, dim>
    gradient(const dealii::Point<dim> &p, unsigned int cmp = 0) const override;

    void
    vector_gradient(const dealii::Point<dim> &           p,
                    std::vector<dealii::Tensor<1, dim>> &values) const override;

    void
    gradient_list(const std::vector<dealii::Point<dim>> &p,
                  std::vector<dealii::Tensor<1, dim>> &  values,
                  unsigned int component = 0) const override;

  private:
    const unsigned int                                 f_component;
    const std::shared_ptr<const dealii::Function<dim>> f;
  };

  template <int dim>
  class Gradient : public FunctionByComponents<dim>
  {
  public:
    explicit Gradient(std::shared_ptr<const dealii::Function<dim>> function);

  private:
    static std::map<unsigned int,
                    const std::shared_ptr<const dealii::Function<dim>>>
    build_component_map(std::shared_ptr<const dealii::Function<dim>> function);

    const std::shared_ptr<const dealii::Function<dim>> f;
  };

  template <int dim>
  class Opposite : public dealii::Function<dim>
  {
  public:
    explicit Opposite(std::shared_ptr<const dealii::Function<dim>> function);

    double
    value(const dealii::Point<dim> &p,
          unsigned int              component = 0) const override;

    void
    vector_value(const dealii::Point<dim> &p,
                 dealii::Vector<double> &  values) const override;

    void
    value_list(const std::vector<dealii::Point<dim>> &p,
               std::vector<double> &                  values,
               unsigned int component = 0) const override;

    dealii::Tensor<1, dim>
    gradient(const dealii::Point<dim> &p,
             unsigned int              component = 0) const override;

    void
    vector_gradient(const dealii::Point<dim> &           p,
                    std::vector<dealii::Tensor<1, dim>> &values) const override;

    void
    gradient_list(const std::vector<dealii::Point<dim>> &p,
                  std::vector<dealii::Tensor<1, dim>> &  values,
                  unsigned int component = 0) const override;

  private:
    const std::shared_ptr<const dealii::Function<dim>> f;
  };
} // namespace Ddhdg
