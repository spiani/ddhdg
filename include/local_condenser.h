#pragma once

#include <deal.II/fe/fe.h>

#include "components.h"

namespace Ddhdg
{
  template <int dim>
  class LocalCondenser
  {
  public:
    static std::vector<Component>
    generate_enabled_components_vector(
      const std::map<Component, const dealii::FiniteElement<dim> &> &fe_map);

    static std::vector<dealii::LAPACKFullMatrix<double>>
    generate_proj_matrix_vector(
      const std::map<Component, const dealii::FiniteElement<dim> &> &fe_map);

    static std::vector<dealii::Vector<double>>
    generate_proj_rhs_vector(
      const std::map<Component, const dealii::FiniteElement<dim> &> &fe_map);

    template <typename dtype>
    static std::vector<std::vector<dtype>>
    generate_dof_vector(
      const std::map<Component, const dealii::FiniteElement<dim> &> &fe_map);

    template <typename dtype>
    static std::vector<dtype>
    generate_cell_dof_vector(
      const std::map<Component, const dealii::FiniteElement<dim> &> &fe_map);

    explicit LocalCondenser(
      const std::map<Component, const dealii::FiniteElement<dim> &> &fe_map);

    LocalCondenser(const LocalCondenser<dim> &other) = default;

    LocalCondenser(LocalCondenser<dim> &&other) = default;

    void
    condense_ct_matrix(dealii::FullMatrix<double> &ct_matrix,
                       dealii::Vector<double>     &cc_rhs);

    unsigned int
    get_component_index(Component c);

    const unsigned int           n_enabled_components;
    const std::vector<Component> enabled_components;

    std::vector<dealii::LAPACKFullMatrix<double>> proj_matrix;
    std::vector<dealii::Vector<double>>           proj_rhs;
    std::vector<std::vector<unsigned int>>        current_dofs;
    std::vector<std::vector<double>>              bf_values;

    unsigned int              n_cell_constrained_dofs;
    std::vector<unsigned int> cell_constrained_dofs;
    std::vector<double>       constrained_dofs_values;
  };
} // namespace Ddhdg
