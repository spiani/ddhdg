#include "adimensionalizer.h"

namespace Ddhdg
{
  double
  Adimensionalizer::get_component_rescaling_factor(const Component c) const
  {
    switch (c)
      {
        case Component::V:
          return this->get_component_rescaling_factor<Component::V>();
        case Component::n:
          return this->get_component_rescaling_factor<Component::n>();
        case Component::p:
          return this->get_component_rescaling_factor<Component::p>();
        default:
          Assert(false, InvalidComponent());
          return 1;
      }
  }



  double
  Adimensionalizer::get_neumann_boundary_condition_rescaling_factor(
    const Component c) const
  {
    switch (c)
      {
        case Component::V:
          return this
            ->get_neumann_boundary_condition_rescaling_factor<Component::V>();
        case Component::n:
          return this
            ->get_neumann_boundary_condition_rescaling_factor<Component::n>();
        case Component::p:
          return this
            ->get_neumann_boundary_condition_rescaling_factor<Component::p>();
        default:
          Assert(false, InvalidComponent());
          return 1;
      }
  }



  template <Component c>
  void
  Adimensionalizer::adimensionalize_component(const std::vector<double> &source,
                                              std::vector<double> &dest) const
  {
    AssertDimension(source.size(), dest.size());
    if (c != Component::V && c != Component::n && c != Component::p)
      Assert(false, InvalidComponent());

    const unsigned int n_of_elements = source.size();
    const double       rescaling = this->get_component_rescaling_factor<c>();

    for (unsigned int i = 0; i < n_of_elements; i++)
      {
        dest[i] = source[i] / rescaling;
      }
  }



  template <Component c>
  void
  Adimensionalizer::redimensionalize_component(
    const std::vector<double> &source,
    std::vector<double> &      dest) const
  {
    AssertDimension(source.size(), dest.size());
    if (c != Component::V && c != Component::n && c != Component::p)
      Assert(false, InvalidComponent());

    const unsigned int n_of_elements = source.size();
    const double       rescaling = this->get_component_rescaling_factor<c>();

    for (unsigned int i = 0; i < n_of_elements; i++)
      {
        dest[i] = source[i] * rescaling;
      }
  }



  template <Component c>
  void
  Adimensionalizer::inplace_adimensionalize_component(
    std::vector<double> &data) const
  {
    if (c != Component::V && c != Component::n && c != Component::p)
      Assert(false, InvalidComponent());

    const unsigned int n_of_elements = data.size();
    const double       rescaling = this->get_component_rescaling_factor<c>();

    for (unsigned int i = 0; i < n_of_elements; i++)
      {
        data[i] /= rescaling;
      }
  }



  template <Component c>
  void
  Adimensionalizer::inplace_redimensionalize_component(
    std::vector<double> &data) const
  {
    if (c != Component::V && c != Component::n && c != Component::p)
      Assert(false, InvalidComponent());

    const unsigned int n_of_elements = data.size();
    const double       rescaling = this->get_component_rescaling_factor<c>();

    for (unsigned int i = 0; i < n_of_elements; i++)
      {
        data[i] *= rescaling;
      }
  }



  void
  Adimensionalizer::adimensionalize_component(const std::vector<double> &source,
                                              Component                  c,
                                              std::vector<double> &dest) const
  {
    switch (c)
      {
        case Component::V:
          this->adimensionalize_component<Component::V>(source, dest);
          break;
        case Component::n:
          this->adimensionalize_component<Component::n>(source, dest);
          break;
        case Component::p:
          this->adimensionalize_component<Component::p>(source, dest);
          break;
        default:
          Assert(false, InvalidComponent());
      }
  }


  void
  Adimensionalizer::redimensionalize_component(
    const std::vector<double> &source,
    Component                  c,
    std::vector<double> &      dest) const
  {
    switch (c)
      {
        case Component::V:
          this->redimensionalize_component<Component::V>(source, dest);
          break;
        case Component::n:
          this->redimensionalize_component<Component::n>(source, dest);
          break;
        case Component::p:
          this->redimensionalize_component<Component::p>(source, dest);
          break;
        default:
          Assert(false, InvalidComponent());
      }
  }



  void
  Adimensionalizer::inplace_adimensionalize_component(std::vector<double> &data,
                                                      Component c) const
  {
    switch (c)
      {
        case Component::V:
          this->inplace_adimensionalize_component<Component::V>(data);
          break;
        case Component::n:
          this->inplace_adimensionalize_component<Component::n>(data);
          break;
        case Component::p:
          this->inplace_adimensionalize_component<Component::p>(data);
          break;
        default:
          Assert(false, InvalidComponent());
      }
  }


  void
  Adimensionalizer::inplace_redimensionalize_component(
    std::vector<double> &data,
    Component            c) const
  {
    switch (c)
      {
        case Component::V:
          this->inplace_redimensionalize_component<Component::V>(data);
          break;
        case Component::n:
          this->inplace_redimensionalize_component<Component::n>(data);
          break;
        case Component::p:
          this->inplace_redimensionalize_component<Component::p>(data);
          break;
        default:
          Assert(false, InvalidComponent());
      }
  }



  void
  Adimensionalizer::adimensionalize_dof_vector(
    const dealii::Vector<double> &dof_vector,
    const std::vector<Component> &dof_to_component_map,
    const std::vector<DofType> &  dof_to_dof_type,
    dealii::Vector<double> &      rescaled_vector) const
  {
    AssertDimension(dof_vector.size(), dof_to_component_map.size());
    AssertDimension(dof_vector.size(), dof_to_dof_type.size());
    AssertDimension(dof_vector.size(), rescaled_vector.size());

    const double V_rescale =
      this->get_component_rescaling_factor<Component::V>();
    const double n_rescale =
      this->get_component_rescaling_factor<Component::n>();
    const double p_rescale =
      this->get_component_rescaling_factor<Component::p>();

    double gradient_rescale;
    for (unsigned int i = 0; i < dof_vector.size(); i++)
      {
        if (dof_to_dof_type[i] == DofType::DISPLACEMENT)
          gradient_rescale = this->scale_length;
        else
          gradient_rescale = 1.;

        switch (dof_to_component_map[i])
          {
            case Component::V:
              rescaled_vector[i] = dof_vector[i] / V_rescale * gradient_rescale;
              break;
            case Component::n:
              rescaled_vector[i] = dof_vector[i] / n_rescale * gradient_rescale;
              break;
            case Component::p:
              rescaled_vector[i] = dof_vector[i] / p_rescale * gradient_rescale;
              break;
            default:
              Assert(false, InvalidComponent());
          }
      }
  }



  void
  Adimensionalizer::redimensionalize_dof_vector(
    const dealii::Vector<double> &dof_vector,
    const std::vector<Component> &dof_to_component_map,
    const std::vector<DofType> &  dof_to_dof_type,
    dealii::Vector<double> &      rescaled_vector) const
  {
    AssertDimension(dof_vector.size(), dof_to_component_map.size());
    AssertDimension(dof_vector.size(), dof_to_dof_type.size());
    AssertDimension(dof_vector.size(), rescaled_vector.size());

    const double V_rescale =
      this->get_component_rescaling_factor<Component::V>();
    const double n_rescale =
      this->get_component_rescaling_factor<Component::n>();
    const double p_rescale =
      this->get_component_rescaling_factor<Component::p>();

    for (unsigned int i = 0; i < dof_vector.size(); i++)
      {
        switch (dof_to_component_map[i])
          {
            case Component::V:
              if (dof_to_dof_type[i] == DofType::DISPLACEMENT)
                rescaled_vector[i] =
                  dof_vector[i] * V_rescale / this->scale_length;
              else
                rescaled_vector[i] = dof_vector[i] * V_rescale;
              break;
            case Component::n:
              if (dof_to_dof_type[i] == DofType::DISPLACEMENT)
                rescaled_vector[i] =
                  dof_vector[i] * n_rescale / this->scale_length;
              else
                rescaled_vector[i] = dof_vector[i] * n_rescale;
              break;
            case Component::p:
              if (dof_to_dof_type[i] == DofType::DISPLACEMENT)
                rescaled_vector[i] =
                  dof_vector[i] * p_rescale / this->scale_length;
              else
                rescaled_vector[i] = dof_vector[i] * p_rescale;
              break;
            default:
              Assert(false, InvalidComponent());
          }
      }
  }



  template <int dim>
  std::shared_ptr<dealii::Function<dim>>
  Adimensionalizer::adimensionalize_component_function(
    const std::shared_ptr<const dealii::Function<dim>> f,
    const Component                                    c) const
  {
    double rescaling = 1.;

    switch (c)
      {
        case Component::V:
          rescaling =
            this->template get_component_rescaling_factor<Component::V>();
          break;
        case Component::n:
          rescaling =
            this->template get_component_rescaling_factor<Component::n>();
          break;
        case Component::p:
          rescaling =
            this->template get_component_rescaling_factor<Component::p>();
          break;
        default:
          Assert(false, InvalidComponent());
      }

    return std::make_shared<FunctionTimesScalar<dim>>(f, 1 / rescaling);
  }



  template <int dim>
  std::shared_ptr<dealii::Function<dim>>
  Adimensionalizer::adimensionalize_doping_function(
    const std::shared_ptr<const dealii::Function<dim>> doping) const
  {
    double rescaling = this->doping_magnitude;

    return std::make_shared<FunctionTimesScalar<dim>>(doping, 1 / rescaling);
  }



  void
  Adimensionalizer::adimensionalize_recombination_term(
    std::vector<double> &r,
    std::vector<double> &dr_n,
    std::vector<double> &dr_p) const
  {
    AssertDimension(r.size(), dr_n.size());
    AssertDimension(r.size(), dr_p.size());

    const double recombination_rescaling_factor =
      this->get_recombination_rescaling_factor();

    const unsigned int n_of_elements = r.size();

    for (unsigned int i = 0; i < n_of_elements; i++)
      r[i] /= recombination_rescaling_factor;

    for (unsigned int i = 0; i < n_of_elements; i++)
      dr_n[i] /= recombination_rescaling_factor;

    for (unsigned int i = 0; i < n_of_elements; i++)
      dr_p[i] /= recombination_rescaling_factor;
  }



  double
  Adimensionalizer::adimensionalize_tau(const double tau, Component c) const
  {
    switch (c)
      {
        case Component::V:
          return this->adimensionalize_tau<Component::V>(tau);
        case Component::n:
          return this->adimensionalize_tau<Component::n>(tau);
        case Component::p:
          return this->adimensionalize_tau<Component::p>(tau);
        default:
          Assert(false, InvalidComponent());
      }
    return 1.;
  }



  // Instantiation of the template methods
  template void
  Adimensionalizer::inplace_adimensionalize_component<Component::V>(
    std::vector<double> &data) const;
  template void
  Adimensionalizer::inplace_adimensionalize_component<Component::n>(
    std::vector<double> &data) const;
  template void
  Adimensionalizer::inplace_adimensionalize_component<Component::p>(
    std::vector<double> &data) const;

  template void
  Adimensionalizer::inplace_redimensionalize_component<Component::V>(
    std::vector<double> &data) const;
  template void
  Adimensionalizer::inplace_redimensionalize_component<Component::n>(
    std::vector<double> &data) const;
  template void
  Adimensionalizer::inplace_redimensionalize_component<Component::p>(
    std::vector<double> &data) const;

  template std::shared_ptr<dealii::Function<1>>
  Adimensionalizer::adimensionalize_component_function<1>(
    std::shared_ptr<const dealii::Function<1>> f,
    Component                                  c) const;
  template std::shared_ptr<dealii::Function<2>>
  Adimensionalizer::adimensionalize_component_function<2>(
    std::shared_ptr<const dealii::Function<2>> f,
    Component                                  c) const;
  template std::shared_ptr<dealii::Function<3>>
  Adimensionalizer::adimensionalize_component_function<3>(
    std::shared_ptr<const dealii::Function<3>> f,
    Component                                  c) const;

  template std::shared_ptr<dealii::Function<1>>
  Adimensionalizer::adimensionalize_doping_function<1>(
    std::shared_ptr<const dealii::Function<1>> doping) const;
  template std::shared_ptr<dealii::Function<2>>
  Adimensionalizer::adimensionalize_doping_function<2>(
    std::shared_ptr<const dealii::Function<2>> doping) const;
  template std::shared_ptr<dealii::Function<3>>
  Adimensionalizer::adimensionalize_doping_function<3>(
    std::shared_ptr<const dealii::Function<3>> doping) const;

} // namespace Ddhdg
