#include "np_solver.h"

namespace Ddhdg
{
  template <int dim>
  double
  NPSolver<dim>::estimate_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                                        expected_solution,
    const Ddhdg::Component              c,
    const dealii::VectorTools::NormType norm) const
  {
    Assert(expected_solution->n_components == 1, FunctionMustBeScalar());
    Vector<double> difference_per_cell(triangulation->n_active_cells());

    std::shared_ptr<dealii::Function<dim>> expected_solution_rescaled =
      this->adimensionalizer->template adimensionalize_component_function<dim>(
        expected_solution, c);

    unsigned int component_index = get_component_index(c) * (1 + dim) + dim;

    const unsigned int n_of_components = (dim + 1) * all_components().size();
    const auto         component_selection =
      dealii::ComponentSelectFunction<dim>(component_index, n_of_components);

    std::map<unsigned int, const std::shared_ptr<const dealii::Function<dim>>>
      c_map;
    c_map.insert({component_index, expected_solution_rescaled});
    FunctionByComponents<dim> expected_solution_multidim =
      FunctionByComponents<dim>(n_of_components, c_map);

    VectorTools::integrate_difference(
      this->dof_handler_cell,
      this->current_solution_cell,
      expected_solution_multidim,
      difference_per_cell,
      QGauss<dim>(this->get_number_of_quadrature_points() + 2),
      norm,
      &component_selection);

    const double global_error =
      VectorTools::compute_global_error(*(this->triangulation),
                                        difference_per_cell,
                                        norm);
    return global_error;
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                                        expected_solution,
    const Ddhdg::Displacement           d,
    const dealii::VectorTools::NormType norm) const
  {
    Assert(expected_solution->n_components == dim,
           FunctionMustHaveDimComponents());
    Vector<double> difference_per_cell(triangulation->n_active_cells());

    Component c = displacement2component(d);

    std::shared_ptr<dealii::Function<dim>> expected_solution_rescaled =
      this->adimensionalizer->template adimensionalize_component_function<dim>(
        expected_solution, c);

    unsigned int       component_index = get_component_index(c);
    const unsigned int n_of_components = (dim + 1) * all_components().size();

    const std::pair<const unsigned int, const unsigned int> selection_interval =
      {component_index * (dim + 1), component_index * (dim + 1) + dim};
    const auto component_selection =
      dealii::ComponentSelectFunction<dim>(selection_interval, n_of_components);

    std::map<unsigned int, const std::shared_ptr<const dealii::Function<dim>>>
      c_map;
    for (unsigned int i = 0; i < dim; i++)
      {
        c_map.insert(
          {component_index * (dim + 1) + i,
           std::make_shared<ComponentFunction<dim>>(expected_solution_rescaled,
                                                    i)});
      }
    FunctionByComponents<dim> expected_solution_multidim =
      FunctionByComponents<dim>(n_of_components, c_map);

    VectorTools::integrate_difference(
      this->dof_handler_cell,
      this->current_solution_cell,
      expected_solution_multidim,
      difference_per_cell,
      QGauss<dim>(this->get_number_of_quadrature_points() + 2),
      norm,
      &component_selection);

    const double global_error =
      VectorTools::compute_global_error(*(this->triangulation),
                                        difference_per_cell,
                                        norm);
    return global_error;
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_error_on_trace(
    const std::shared_ptr<const dealii::Function<dim, double>>
                                        expected_solution,
    const Ddhdg::Component              c,
    const dealii::VectorTools::NormType norm) const
  {
    Assert(expected_solution->n_components == 1, FunctionMustBeScalar());
    Assert(norm == VectorTools::L2_norm || norm == VectorTools::Linfty_norm,
           ExcNotImplemented())

      const unsigned int           c_index = get_component_index(c);
    std::map<unsigned int, double> difference_per_face;

    std::shared_ptr<dealii::Function<dim>> expected_solution_rescaled =
      this->adimensionalizer->template adimensionalize_component_function<dim>(
        expected_solution, c);

    const QGauss<dim - 1> face_quadrature_formula(
      this->get_number_of_quadrature_points() + 2);
    const UpdateFlags flags(update_values | update_quadrature_points |
                            update_JxW_values);

    const unsigned int n_face_q_points = face_quadrature_formula.size();
    const unsigned int dofs_per_cell   = this->fe_trace->dofs_per_cell;

    FEFaceValues<dim> fe_face_trace_values(*(this->fe_trace),
                                           face_quadrature_formula,
                                           flags);

    const FEValuesExtractors::Scalar c_extractor =
      this->get_trace_component_extractor(c);

    // Check which dofs are related to our component
    std::vector<unsigned int> on_current_component;
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const unsigned int current_index =
          this->fe_trace->system_to_block_index(i).first;
        if (current_index == c_index)
          on_current_component.push_back(i);
      }
    const unsigned int dofs_per_component = on_current_component.size();

    // Check which component related dofs are relevant for each face
    std::vector<std::vector<unsigned int>> component_support_on_face(
      GeometryInfo<dim>::faces_per_cell);
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
         ++face)
      for (unsigned int i = 0; i < dofs_per_component; ++i)
        {
          const unsigned int dof_index = on_current_component[i];
          if (this->fe_trace->has_support_on_face(dof_index, face))
            component_support_on_face[face].push_back(i);
        }
    const unsigned int dofs_per_face_on_component =
      component_support_on_face[0].size();

    Vector<double> current_solution_node_values(dofs_per_cell);

    std::vector<Point<dim>> face_quadrature_points(n_face_q_points);
    std::vector<double>     expected_solution_on_q(n_face_q_points);
    std::vector<double>     current_solution_on_q(n_face_q_points);

    for (const auto &cell : dof_handler_trace.active_cell_iterators())
      {
        cell->get_dof_values(this->current_solution_trace,
                             current_solution_node_values);

        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            unsigned int face_uid = cell->face_index(face);

            // If this face has already been examined from another cell, skip
            // it
            if (difference_per_face.find(face_uid) != difference_per_face.end())
              continue;

            fe_face_trace_values.reinit(cell, face);

            // Compute the position of the quadrature points
            for (unsigned int q = 0; q < n_face_q_points; ++q)
              face_quadrature_points[q] =
                fe_face_trace_values.quadrature_point(q);

            // Compute the value of the current solution on the quadrature
            // points
            for (unsigned int q = 0; q < n_face_q_points; ++q)
              {
                double solution_on_q = 0;
                for (unsigned int i = 0; i < dofs_per_face_on_component; i++)
                  {
                    const unsigned int ii =
                      on_current_component[component_support_on_face[face][i]];
                    solution_on_q +=
                      fe_face_trace_values[c_extractor].value(ii, q) *
                      current_solution_node_values[ii];
                  }
                current_solution_on_q[q] = solution_on_q;
              }

            // Compute the value of the expected solution on the quadrature
            // points
            expected_solution_rescaled->value_list(face_quadrature_points,
                                                   expected_solution_on_q);

            // Now it's time to perform the integration
            double difference_norm = 0;
            if (norm == VectorTools::L2_norm)
              {
                for (unsigned int q = 0; q < n_face_q_points; ++q)
                  {
                    const double JxW = fe_face_trace_values.JxW(q);
                    double       difference_on_q =
                      current_solution_on_q[q] - expected_solution_on_q[q];
                    difference_norm += difference_on_q * difference_on_q * JxW;
                  }
              }
            if (norm == VectorTools::Linfty_norm)
              {
                for (unsigned int q = 0; q < n_face_q_points; ++q)
                  {
                    double difference_on_q =
                      current_solution_on_q[q] - expected_solution_on_q[q];
                    double abs_difference = (difference_on_q > 0) ?
                                              difference_on_q :
                                              -difference_on_q;
                    if (abs_difference > difference_norm)
                      difference_norm = abs_difference;
                  }
              }

            // We save the result in the map for each face
            difference_per_face.insert({face_uid, difference_norm});
          }
      }

    // Now we need to aggregate data
    switch (norm)
      {
          case VectorTools::L2_norm: {
            double norm_sum = 0;
            for (const auto &nrm : difference_per_face)
              {
                norm_sum += nrm.second;
              }
            return sqrt(norm_sum);
          }
          case VectorTools::Linfty_norm: {
            double max_value = 0;
            for (const auto &nrm : difference_per_face)
              {
                max_value = (max_value > nrm.second) ? max_value : nrm.second;
              }
            return max_value;
          }
        default:
          break;
      }
    return 1e99;
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                           expected_solution,
    const Ddhdg::Component c) const
  {
    return this->estimate_error(expected_solution,
                                c,
                                dealii::VectorTools::L2_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                              expected_solution,
    const Ddhdg::Displacement d) const
  {
    return this->estimate_error(expected_solution,
                                d,
                                dealii::VectorTools::L2_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error_on_trace(
    const std::shared_ptr<const dealii::Function<dim, double>>
                           expected_solution,
    const Ddhdg::Component c) const
  {
    return this->estimate_error_on_trace(expected_solution,
                                         c,
                                         dealii::VectorTools::L2_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                           expected_solution,
    const Ddhdg::Component c) const
  {
    return this->estimate_error(expected_solution,
                                c,
                                dealii::VectorTools::H1_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                              expected_solution,
    const Ddhdg::Displacement d) const
  {
    return this->estimate_error(expected_solution,
                                d,
                                dealii::VectorTools::H1_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                           expected_solution,
    const Ddhdg::Component c) const
  {
    return this->estimate_error(expected_solution,
                                c,
                                dealii::VectorTools::Linfty_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(
    const std::shared_ptr<const dealii::Function<dim, double>>
                              expected_solution,
    const Ddhdg::Displacement d) const
  {
    return this->estimate_error(expected_solution,
                                d,
                                dealii::VectorTools::Linfty_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error_on_trace(
    const std::shared_ptr<const dealii::Function<dim, double>>
                           expected_solution,
    const Ddhdg::Component c) const
  {
    return this->estimate_error_on_trace(expected_solution,
                                         c,
                                         dealii::VectorTools::Linfty_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_error(const NPSolver<dim> &         other,
                                Component                     c,
                                dealii::VectorTools::NormType norm) const
  {
    Vector<double> difference_per_cell(triangulation->n_active_cells());

    auto tmp(this->current_solution_cell);

    dealii::VectorTools::interpolate_to_different_mesh(
      other.dof_handler_cell,
      other.current_solution_cell,
      this->dof_handler_cell,
      tmp);

    tmp -= this->current_solution_cell;

    unsigned int component_index = get_component_index(c) * (1 + dim) + dim;

    const unsigned int n_of_components = (dim + 1) * all_components().size();
    const auto         component_selection =
      dealii::ComponentSelectFunction<dim>(component_index, n_of_components);

    VectorTools::integrate_difference(
      this->dof_handler_cell,
      tmp,
      dealii::Functions::ZeroFunction<dim>(n_of_components),
      difference_per_cell,
      QGauss<dim>(this->get_number_of_quadrature_points() + 2),
      norm,
      &component_selection);

    const double global_error =
      VectorTools::compute_global_error(*(this->triangulation),
                                        difference_per_cell,
                                        norm);
    return global_error;
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_error(const NPSolver<dim> &         other,
                                Displacement                  d,
                                dealii::VectorTools::NormType norm) const
  {
    Vector<double> difference_per_cell(triangulation->n_active_cells());

    auto tmp(this->current_solution_cell);

    dealii::VectorTools::interpolate_to_different_mesh(
      other.dof_handler_cell,
      other.current_solution_cell,
      this->dof_handler_cell,
      tmp);

    tmp -= this->current_solution_cell;

    Component c = displacement2component(d);

    unsigned int       component_index = get_component_index(c);
    const unsigned int n_of_components = (dim + 1) * all_components().size();

    const std::pair<const unsigned int, const unsigned int> selection_interval =
      {component_index * (dim + 1), component_index * (dim + 1) + dim};
    const auto component_selection =
      dealii::ComponentSelectFunction<dim>(selection_interval, n_of_components);

    VectorTools::integrate_difference(
      this->dof_handler_cell,
      tmp,
      dealii::Functions::ZeroFunction<dim>(3 * (dim + 1)),
      difference_per_cell,
      QGauss<dim>(this->get_number_of_quadrature_points() + 2),
      norm,
      &component_selection);

    const double global_error =
      VectorTools::compute_global_error(*(this->triangulation),
                                        difference_per_cell,
                                        norm);
    return global_error;
  }



  template class NPSolver<1>;
  template class NPSolver<2>;
  template class NPSolver<3>;

} // namespace Ddhdg