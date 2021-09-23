#include "local_condenser.h"

namespace Ddhdg
{
  template <int dim>
  std::vector<Component>
  LocalCondenser<dim>::generate_enabled_components_vector(
    const std::map<Component, const dealii::FiniteElement<dim> &> &fe_map)
  {
    std::vector<Component> enabled_c_vector;
    enabled_c_vector.reserve(fe_map.size());
    for (const auto &[c, fe] : fe_map)
      {
        (void)fe;
        enabled_c_vector.push_back(c);
      }
    return enabled_c_vector;
  }



  template <int dim>
  std::vector<dealii::LAPACKFullMatrix<double>>
  LocalCondenser<dim>::generate_proj_matrix_vector(
    const std::map<Component, const dealii::FiniteElement<dim> &> &fe_map)
  {
    std::vector<dealii::LAPACKFullMatrix<double>> proj_matrix_vector;
    proj_matrix_vector.reserve(fe_map.size());

    unsigned int dofs_per_face;
    for (const auto &[c, fe] : fe_map)
      {
        (void)c;
        dofs_per_face = 0;
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          if (fe.has_support_on_face(i, 0))
            ++dofs_per_face;
        proj_matrix_vector.push_back(
          dealii::LAPACKFullMatrix<double>(dofs_per_face, dofs_per_face));
      }

    return proj_matrix_vector;
  }



  template <int dim>
  std::vector<dealii::Vector<double>>
  LocalCondenser<dim>::generate_proj_rhs_vector(
    const std::map<Component, const dealii::FiniteElement<dim> &> &fe_map)
  {
    std::vector<dealii::Vector<double>> proj_rhs_vector;
    proj_rhs_vector.reserve(fe_map.size());

    unsigned int dofs_per_face;
    for (const auto &[c, fe] : fe_map)
      {
        (void)c;
        dofs_per_face = 0;
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          if (fe.has_support_on_face(i, 0))
            ++dofs_per_face;
        proj_rhs_vector.push_back(dealii::Vector<double>(dofs_per_face));
      }
    return proj_rhs_vector;
  }



  template <int dim>
  template <typename dtype>
  std::vector<std::vector<dtype>>
  LocalCondenser<dim>::generate_dof_vector(
    const std::map<Component, const dealii::FiniteElement<dim> &> &fe_map)
  {
    std::vector<std::vector<dtype>> proj_rhs_vector;
    proj_rhs_vector.reserve(fe_map.size());

    unsigned int dofs_per_face;
    for (const auto &[c, fe] : fe_map)
      {
        (void)c;
        dofs_per_face = 0;
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          if (fe.has_support_on_face(i, 0))
            ++dofs_per_face;
        proj_rhs_vector.push_back(std::vector<dtype>(dofs_per_face));
      }
    return proj_rhs_vector;
  }



  template <int dim>
  template <typename dtype>
  std::vector<dtype>
  LocalCondenser<dim>::generate_cell_dof_vector(
    const std::map<Component, const dealii::FiniteElement<dim> &> &fe_map)
  {
    unsigned int dofs_per_cell = 0;
    for (const auto &[c, fe] : fe_map)
      {
        (void)c;
        dofs_per_cell += fe.dofs_per_cell;
      }
    return std::vector<dtype>(dofs_per_cell);
  }



  template <int dim>
  LocalCondenser<dim>::LocalCondenser(
    const std::map<Component, const dealii::FiniteElement<dim> &> &fe_map)
    : n_enabled_components(fe_map.size())
    , enabled_components(generate_enabled_components_vector(fe_map))
    , proj_matrix(generate_proj_matrix_vector(fe_map))
    , proj_rhs(generate_proj_rhs_vector(fe_map))
    , current_dofs(generate_dof_vector<unsigned int>(fe_map))
    , bf_values(generate_dof_vector<double>(fe_map))
    , n_cell_constrained_dofs(0)
    , cell_constrained_dofs(generate_cell_dof_vector<unsigned int>(fe_map))
    , constrained_dofs_values(generate_cell_dof_vector<double>(fe_map))
  {}



  template <int dim>
  unsigned int
  LocalCondenser<dim>::get_component_index(Component c)
  {
    for (unsigned int i = 0; i < this->n_enabled_components; i++)
      if (this->enabled_components[i] == c)
        return i;
    Assert(false, InvalidComponent());
    return dealii::numbers::invalid_unsigned_int;
  }



  template <int dim>
  void
  LocalCondenser<dim>::condense_ct_matrix(dealii::FullMatrix<double> &ct_matrix,
                                          dealii::Vector<double>     &cc_rhs)
  {
    const unsigned int n_rows = ct_matrix.n_rows();

    unsigned int jj;
    double       j_value;
    double       matrix_value;
    for (unsigned int i = 0; i < n_rows; ++i)
      for (unsigned int j = 0; j < this->n_cell_constrained_dofs; ++j)
        {
          jj      = cell_constrained_dofs[j];
          j_value = constrained_dofs_values[j];

          matrix_value     = ct_matrix(i, jj);
          ct_matrix(i, jj) = 0;
          cc_rhs[i] -= matrix_value * j_value;
        }
  }

  template class LocalCondenser<1>;
  template class LocalCondenser<2>;
  template class LocalCondenser<3>;

} // namespace Ddhdg
