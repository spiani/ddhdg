#include <deal.II/dofs/dof_tools.h>

#include <cmath>

#include "np_solver.h"

namespace Ddhdg
{
  template <int dim, class Permittivity>
  void
  NPSolver<dim, Permittivity>::compute_local_charge_neutrality_on_a_point(
    const std::vector<double> &evaluated_doping,
    const std::vector<double> &evaluated_temperature,
    std::vector<double> &      evaluated_potentials)
  {
    AssertDimension(evaluated_doping.size(), evaluated_temperature.size());
    AssertDimension(evaluated_doping.size(), evaluated_potentials.size());

    const unsigned int n_of_points = evaluated_doping.size();

    const double rescale_factor =
      this->adimensionalizer
        ->template get_component_rescaling_factor<Component::V>();

    constexpr double ev = Ddhdg::Constants::EV;
    const double     Nc = this->problem->band_density.at(Component::n);
    const double     Ec = this->problem->band_edge_energy.at(Component::n) * ev;
    const double     Nv = this->problem->band_density.at(Component::p);
    const double     Ev = this->problem->band_edge_energy.at(Component::p) * ev;

    for (unsigned int i = 0; i < n_of_points; i++)
      {
        const double U_T =
          evaluated_temperature[i] * Constants::KB / Constants::Q;
        const double N_intr_square =
          Nc * Nv * exp((Ev - Ec) / (evaluated_temperature[i] * Constants::KB));
        const double twice_N_intr = 2 * sqrt(N_intr_square);
        evaluated_potentials[i] =
          ((Ec + Ev) / (2 * Constants::Q) + U_T / 2 * log(Nv / Nc) +
           U_T * asinh(evaluated_doping[i] / twice_N_intr)) /
          rescale_factor;
      }
  }



  template <int dim, class Permittivity>
  void
  NPSolver<dim, Permittivity>::compute_local_charge_neutrality_on_cells()
  {
    const QGauss<dim> quadrature_formula(
      this->get_number_of_quadrature_points());
    const QGauss<dim - 1> face_quadrature_formula(
      this->get_number_of_quadrature_points());

    const unsigned int V_index = get_component_index(Component::V);

    const UpdateFlags flags_cell(update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points);
    const UpdateFlags flags_trace(update_values | update_normal_vectors |
                                  update_quadrature_points | update_JxW_values);

    FEValues<dim>     fe_values_cell(*(this->fe_cell),
                                 quadrature_formula,
                                 flags_cell);
    FEFaceValues<dim> fe_face_values(*(this->fe_cell),
                                     face_quadrature_formula,
                                     flags_trace);

    const unsigned int n_q_points      = fe_values_cell.get_quadrature().size();
    const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();

    // Now we need to map the dofs that are related to the current component
    std::vector<unsigned int> on_current_component;
    for (unsigned int i = 0; i < this->fe_cell->dofs_per_cell; ++i)
      {
        const unsigned int current_index =
          this->fe_cell->system_to_block_index(i).first;
        if (current_index == V_index)
          on_current_component.push_back(i);
      }
    const unsigned int dofs_per_component = on_current_component.size();

    std::vector<std::vector<unsigned int>> component_support_on_face(
      GeometryInfo<dim>::faces_per_cell);
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
         ++face)
      for (unsigned int i = 0; i < dofs_per_component; ++i)
        {
          const unsigned int dof_index = on_current_component[i];
          if (this->fe_cell->has_support_on_face(dof_index, face))
            component_support_on_face[face].push_back(i);
        }
    const unsigned int dofs_per_face_on_component =
      component_support_on_face[0].size();

    const FEValuesExtractors::Vector E_extractor =
      this->get_displacement_extractor(Displacement::E);
    const FEValuesExtractors::Scalar V_extractor =
      this->get_component_extractor(Component::V);

    LAPACKFullMatrix<double> local_matrix(dofs_per_component,
                                          dofs_per_component);
    Vector<double>           local_residual(dofs_per_component);
    Vector<double>           local_values(fe_values_cell.dofs_per_cell);

    // Temporary buffer for the values of the local base function on a
    // quadrature point
    std::vector<double>         c_bf(dofs_per_component);
    std::vector<Tensor<1, dim>> d_bf(dofs_per_component);
    std::vector<double>         d_div_bf(dofs_per_component);

    std::vector<Point<dim>> cell_quadrature_points(n_q_points);
    std::vector<Point<dim>> face_quadrature_points(n_face_q_points);

    std::vector<double> evaluated_doping(n_q_points);
    std::vector<double> evaluated_temperature(n_q_points);
    std::vector<double> evaluated_potentials(n_q_points);

    std::vector<double> evaluated_doping_face(n_face_q_points);
    std::vector<double> evaluated_temperature_face(n_face_q_points);
    std::vector<double> evaluated_potentials_face(n_face_q_points);

    for (const auto &cell : this->dof_handler_cell.active_cell_iterators())
      {
        local_matrix   = 0.;
        local_residual = 0.;

        fe_values_cell.reinit(cell);

        // Get the position of the quadrature points
        for (unsigned int q = 0; q < n_q_points; ++q)
          cell_quadrature_points[q] = fe_values_cell.quadrature_point(q);

        // Evaluated the analytic functions over the quadrature points
        this->problem->doping->value_list(cell_quadrature_points,
                                          evaluated_doping);
        this->problem->temperature->value_list(cell_quadrature_points,
                                               evaluated_temperature);

        this->compute_local_charge_neutrality_on_a_point(evaluated_doping,
                                                         evaluated_temperature,
                                                         evaluated_potentials);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            // Copy data of the shape function
            for (unsigned int k = 0; k < dofs_per_component; ++k)
              {
                const unsigned int i = on_current_component[k];
                c_bf[k]              = fe_values_cell[V_extractor].value(i, q);
                d_bf[k]              = fe_values_cell[E_extractor].value(i, q);
                d_div_bf[k] = fe_values_cell[E_extractor].divergence(i, q);
              }

            const double JxW = fe_values_cell.JxW(q);

            for (unsigned int i = 0; i < dofs_per_component; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_component; ++j)
                  {
                    local_matrix(i, j) +=
                      (c_bf[j] * c_bf[i] + d_bf[i] * d_bf[j]) * JxW;
                  }
                local_residual[i] +=
                  (evaluated_potentials[q] * (c_bf[i] + d_div_bf[i])) * JxW;
              }
          }
        for (unsigned int face_number = 0;
             face_number < GeometryInfo<dim>::faces_per_cell;
             ++face_number)
          {
            fe_face_values.reinit(cell, face_number);

            for (unsigned int q = 0; q < n_face_q_points; ++q)
              face_quadrature_points[q] = fe_face_values.quadrature_point(q);

            this->problem->doping->value_list(face_quadrature_points,
                                              evaluated_doping_face);
            this->problem->temperature->value_list(face_quadrature_points,
                                                   evaluated_temperature_face);

            this->compute_local_charge_neutrality_on_a_point(
              evaluated_doping_face,
              evaluated_temperature_face,
              evaluated_potentials_face);

            for (unsigned int q = 0; q < n_face_q_points; ++q)
              {
                const double JxW    = fe_face_values.JxW(q);
                const auto   normal = fe_face_values.normal_vector(q);

                for (unsigned int k = 0; k < dofs_per_face_on_component; ++k)
                  {
                    const auto kk  = component_support_on_face[face_number][k];
                    const auto kkk = on_current_component[kk];
                    const auto f_bf_face =
                      fe_face_values[E_extractor].value(kkk, q);
                    local_residual[kk] +=
                      (-evaluated_potentials_face[q] * (f_bf_face * normal)) *
                      JxW;
                  }
              }
          }
        local_matrix.compute_lu_factorization();
        local_matrix.solve(local_residual);

        cell->get_dof_values(this->current_solution_cell, local_values);
        for (unsigned int i = 0; i < dofs_per_component; i++)
          local_values[on_current_component[i]] = local_residual[i];
        cell->set_dof_values(local_values, this->current_solution_cell);
      }
  }



  template <int dim, class Permittivity>
  void
  NPSolver<dim, Permittivity>::compute_local_charge_neutrality_on_trace(
    const bool only_at_boundary)
  {
    const QGauss<dim> quadrature_formula(
      this->get_number_of_quadrature_points());
    const QGauss<dim - 1> face_quadrature_formula(
      this->get_number_of_quadrature_points());

    const unsigned int V_index = get_component_index(Component::V);
    const UpdateFlags  flags(update_values | update_normal_vectors |
                            update_quadrature_points | update_JxW_values);

    FEFaceValues<dim>  fe_face_trace_values(*(this->fe_trace),
                                           face_quadrature_formula,
                                           flags);
    const unsigned int n_face_q_points =
      fe_face_trace_values.get_quadrature().size();

    const unsigned int dofs_per_cell = this->fe_trace->dofs_per_cell;

    const FEValuesExtractors::Scalar V_extractor =
      this->get_trace_component_extractor(Component::V);

    // Again, we map the dofs that are related to the current component
    std::vector<unsigned int> on_current_component;
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const unsigned int current_index =
          this->fe_trace->system_to_block_index(i).first;
        if (current_index == V_index)
          on_current_component.push_back(i);
      }
    const unsigned int dofs_per_component = on_current_component.size();

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

    std::vector<double> c_bf(dofs_per_face_on_component);

    std::vector<Point<dim>> face_quadrature_points(n_face_q_points);

    std::vector<double> evaluated_doping(n_face_q_points);
    std::vector<double> evaluated_temperature(n_face_q_points);
    std::vector<double> evaluated_potentials(n_face_q_points);

    LAPACKFullMatrix<double> local_trace_matrix(dofs_per_face_on_component,
                                                dofs_per_face_on_component);
    Vector<double>           local_trace_residual(dofs_per_face_on_component);
    Vector<double>           local_trace_values(dofs_per_cell);

    for (const auto &cell : dof_handler_trace.active_cell_iterators())
      {
        cell->get_dof_values(this->current_solution_trace, local_trace_values);
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            if (only_at_boundary && !cell->face(face)->at_boundary())
              continue;

            if (only_at_boundary)
              {
                const auto &boundary_handler = this->problem->boundary_handler;
                const types::boundary_id face_boundary_id =
                  cell->face(face)->boundary_id();
                if (!boundary_handler->has_dirichlet_boundary_conditions(
                      face_boundary_id, Component::V))
                  continue;
              }

            local_trace_matrix   = 0;
            local_trace_residual = 0;

            fe_face_trace_values.reinit(cell, face);
            for (unsigned int q = 0; q < n_face_q_points; ++q)
              face_quadrature_points[q] =
                fe_face_trace_values.quadrature_point(q);

            this->problem->doping->value_list(face_quadrature_points,
                                              evaluated_doping);
            this->problem->temperature->value_list(face_quadrature_points,
                                                   evaluated_temperature);

            this->compute_local_charge_neutrality_on_a_point(
              evaluated_doping, evaluated_temperature, evaluated_potentials);

            for (unsigned int q = 0; q < n_face_q_points; ++q)
              {
                // Copy data of the shape function
                for (unsigned int k = 0; k < dofs_per_face_on_component; ++k)
                  {
                    const unsigned int component_index =
                      component_support_on_face[face][k];
                    const unsigned int local_index =
                      on_current_component[component_index];
                    c_bf[k] =
                      fe_face_trace_values[V_extractor].value(local_index, q);
                  }

                const double JxW = fe_face_trace_values.JxW(q);

                for (unsigned int i = 0; i < dofs_per_face_on_component; ++i)
                  {
                    for (unsigned int j = 0; j < dofs_per_face_on_component;
                         ++j)
                      local_trace_matrix(i, j) += (c_bf[j] * c_bf[i]) * JxW;
                    local_trace_residual[i] +=
                      (evaluated_potentials[q] * c_bf[i]) * JxW;
                  }
              }
            local_trace_matrix.compute_lu_factorization();
            local_trace_matrix.solve(local_trace_residual);

            for (unsigned int i = 0; i < dofs_per_face_on_component; i++)
              {
                const unsigned int ii = component_support_on_face[face][i];
                local_trace_values[on_current_component[ii]] =
                  local_trace_residual[i];
              }
          }
        cell->set_dof_values(local_trace_values, this->current_solution_trace);
      }
    this->constraints.distribute(this->current_solution_trace);
  }



  template <int dim, class Permittivity>
  void
  NPSolver<dim, Permittivity>::compute_local_charge_neutrality()
  {
    this->compute_local_charge_neutrality_on_cells();
    this->compute_local_charge_neutrality_on_trace(false);
  }



  template <int dim, class Permittivity>
  NonlinearIterationResults
  NPSolver<dim, Permittivity>::compute_thermodynamic_equilibrium(
    const double absolute_tol,
    const double relative_tol,
    const int    max_number_of_iterations,
    const bool   generate_first_guess)
  {
    std::map<Component, bool> current_active_components;
    for (Component c : all_components())
      {
        current_active_components[c] = this->is_enabled(c);
      }

    this->set_enabled_components(true, false, false);

    if (!this->initialized)
      this->setup_overall_system();

    if (generate_first_guess)
      this->compute_local_charge_neutrality();

    NonlinearIterationResults iterations = this->private_run(
      absolute_tol, relative_tol, max_number_of_iterations, true);

    // Now we need to set the values of n and p
    // First of all we check which components are related with n and p
    const unsigned int        dofs_per_cell = this->fe_cell->dofs_per_cell;
    const unsigned int        n_index       = get_component_index(Component::n);
    const unsigned int        p_index       = get_component_index(Component::p);
    std::vector<unsigned int> n_components;
    std::vector<unsigned int> p_components;
    std::vector<unsigned int> Wn_components;
    std::vector<unsigned int> Wp_components;

    const unsigned int n_faces = dealii::GeometryInfo<dim>::faces_per_cell;
    std::vector<std::vector<unsigned int>> Wn_components_on_face(n_faces);
    std::vector<std::vector<unsigned int>> Wp_components_on_face(n_faces);

    // Here we want the subfe systems related with n and p
    const dealii::ComponentMask n_mask = this->get_component_mask(Component::n);
    const dealii::ComponentMask p_mask = this->get_component_mask(Component::p);
    const dealii::ComponentMask Wn_mask =
      this->get_component_mask(Displacement::Wn);
    const dealii::ComponentMask Wp_mask =
      this->get_component_mask(Displacement::Wp);
    const dealii::ComponentMask       total_n_mask = n_mask | Wn_mask;
    const dealii::ComponentMask       total_p_mask = p_mask | Wp_mask;
    const dealii::FiniteElement<dim> &n_fe_system =
      this->fe_cell->get_sub_fe(total_n_mask.first_selected_component(),
                                total_n_mask.n_selected_components());
    const dealii::FiniteElement<dim> &p_fe_system =
      this->fe_cell->get_sub_fe(total_p_mask.first_selected_component(),
                                total_p_mask.n_selected_components());

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const auto global_fe_indices = this->fe_cell->system_to_block_index(i);
        const unsigned int current_block       = global_fe_indices.first;
        const unsigned int current_block_index = global_fe_indices.second;
        if (current_block == n_index)
          switch (n_fe_system.system_to_block_index(current_block_index).first)
            {
              case 0:
                Wn_components.push_back(i);
                break;
              case 1:
                n_components.push_back(i);
                break;
              default:
                Assert(false, ExcInternalError("Unexpected index value"));
                break;
            }

        if (current_block == p_index)
          switch (p_fe_system.system_to_block_index(current_block_index).first)
            {
              case 0:
                Wp_components.push_back(i);
                break;
              case 1:
                p_components.push_back(i);
                break;
              default:
                break;
            }
      }

    for (unsigned int face = 0; face < n_faces; ++face)
      {
        for (unsigned int i = 0; i < n_components.size(); ++i)
          {
            const unsigned int ii = n_components[i];
            if (this->fe_cell->has_support_on_face(ii, face))
              Wn_components_on_face[face].push_back(i);
          }
        for (unsigned int i = 0; i < p_components.size(); ++i)
          {
            const unsigned int ii = p_components[i];
            if (this->fe_cell->has_support_on_face(ii, face))
              Wp_components_on_face[face].push_back(i);
          }
      }

    // Now we prepare the space for the projection
    const QGauss<dim> quadrature_formula(
      this->get_number_of_quadrature_points());
    const UpdateFlags flags_cell(update_values | update_JxW_values |
                                 update_gradients | update_quadrature_points);
    FEValues<dim>     fe_values_cell(*(this->fe_cell),
                                 quadrature_formula,
                                 flags_cell);

    const QGauss<dim - 1> face_quadrature_formula(
      this->get_number_of_quadrature_points());
    const UpdateFlags flags_face(update_values | update_JxW_values |
                                 update_quadrature_points |
                                 update_normal_vectors);
    FEFaceValues<dim> fe_face_values(*(this->fe_cell),
                                     face_quadrature_formula,
                                     flags_face);


    LAPACKFullMatrix<double> n_matrix(n_components.size(), n_components.size());
    LAPACKFullMatrix<double> p_matrix(p_components.size(), p_components.size());
    LAPACKFullMatrix<double> Wn_matrix(Wn_components.size(),
                                       Wn_components.size());
    LAPACKFullMatrix<double> Wp_matrix(Wp_components.size(),
                                       Wp_components.size());

    const unsigned int     n_q_points = quadrature_formula.size();
    dealii::Vector<double> n_rhs(n_components.size());
    dealii::Vector<double> p_rhs(p_components.size());
    dealii::Vector<double> Wn_rhs(Wn_components.size());
    dealii::Vector<double> Wp_rhs(Wp_components.size());

    std::vector<dealii::Point<dim>> cell_quadrature_points(n_q_points);
    std::vector<double>             V_values(n_q_points);
    std::vector<double>             temperature(n_q_points);

    const unsigned int n_face_q_points = face_quadrature_formula.size();
    std::vector<dealii::Point<dim>> face_quadrature_points(n_face_q_points);
    std::vector<double>             V_values_face(n_face_q_points);
    std::vector<double>             temperature_face(n_face_q_points);

    Vector<double> local_values(fe_values_cell.dofs_per_cell);

    const auto V_extractor = this->get_component_extractor(Component::V);
    const auto n_extractor = this->get_component_extractor(Component::n);
    const auto p_extractor = this->get_component_extractor(Component::p);
    const auto Wn_extractor =
      this->get_displacement_extractor(Displacement::Wn);
    const auto Wp_extractor =
      this->get_displacement_extractor(Displacement::Wp);

    const double V_rescale =
      this->adimensionalizer
        ->template get_component_rescaling_factor<Component::V>();
    const double n_rescale =
      this->adimensionalizer
        ->template get_component_rescaling_factor<Component::n>();
    const double p_rescale =
      this->adimensionalizer
        ->template get_component_rescaling_factor<Component::p>();

    for (const auto &cell : this->dof_handler_cell.active_cell_iterators())
      {
        fe_values_cell.reinit(cell);

        n_matrix = 0.;
        p_matrix = 0.;
        n_rhs    = 0.;
        p_rhs    = 0.;

        Wn_matrix = 0.;
        Wp_matrix = 0.;
        Wn_rhs    = 0.;
        Wp_rhs    = 0.;

        cell->get_dof_values(this->current_solution_cell, local_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
          cell_quadrature_points[q] = fe_values_cell.quadrature_point(q);

        this->problem->temperature->value_list(cell_quadrature_points,
                                               temperature);

        fe_values_cell[V_extractor].get_function_values(
          this->current_solution_cell, V_values);
        for (unsigned int q = 0; q < n_q_points; q++)
          V_values[q] *= V_rescale;

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const double JxW = fe_values_cell.JxW(q);

            const double n_value =
              this->template compute_density<Component::n>(0.,
                                                           V_values[q],
                                                           temperature[q]) /
              n_rescale;
            const double p_value =
              this->template compute_density<Component::p>(0.,
                                                           V_values[q],
                                                           temperature[q]) /
              p_rescale;

            for (unsigned int i = 0; i < n_components.size(); ++i)
              {
                const unsigned int ii = n_components[i];
                const double       v = fe_values_cell[n_extractor].value(ii, q);
                for (unsigned int j = 0; j < n_components.size(); ++j)
                  {
                    const unsigned int jj = n_components[j];
                    const double u = fe_values_cell[n_extractor].value(jj, q);
                    n_matrix(i, j) += u * v * JxW;
                  }
                n_rhs[i] += v * n_value * JxW;
              }

            for (unsigned int i = 0; i < p_components.size(); ++i)
              {
                const unsigned int ii = p_components[i];
                const double       v = fe_values_cell[p_extractor].value(ii, q);
                for (unsigned int j = 0; j < p_components.size(); ++j)
                  {
                    const unsigned int jj = p_components[j];
                    const double u = fe_values_cell[p_extractor].value(jj, q);
                    p_matrix(i, j) += u * v * JxW;
                  }
                p_rhs[i] += v * p_value * JxW;
              }

            for (unsigned int i = 0; i < Wn_components.size(); ++i)
              {
                const unsigned int ii = Wn_components[i];
                const auto v = fe_values_cell[Wn_extractor].value(ii, q);
                for (unsigned int j = 0; j < Wn_components.size(); ++j)
                  {
                    const unsigned int jj = Wn_components[j];
                    const auto u = fe_values_cell[Wn_extractor].value(jj, q);
                    Wn_matrix(i, j) += v * u * JxW;
                  }
                const double v_div =
                  fe_values_cell[Wn_extractor].divergence(ii, q);
                Wn_rhs[i] += v_div * n_value * JxW;
              }

            for (unsigned int i = 0; i < Wp_components.size(); ++i)
              {
                const unsigned int ii = Wp_components[i];
                const auto v = fe_values_cell[Wp_extractor].value(ii, q);
                for (unsigned int j = 0; j < Wp_components.size(); ++j)
                  {
                    const unsigned int jj = Wp_components[j];
                    const auto u = fe_values_cell[Wp_extractor].value(jj, q);
                    Wp_matrix(i, j) += u * v * JxW;
                  }
                const double v_div =
                  fe_values_cell[Wp_extractor].divergence(ii, q);
                Wp_rhs[i] += v_div * p_value * JxW;
              }
          }

        // For Wn and Wp the equation is a little bit more complicated and
        // involves the values on the boundary of the cell
        for (unsigned int face = 0; face < n_faces; ++face)
          {
            fe_face_values.reinit(cell, face);

            for (unsigned int q = 0; q < n_face_q_points; ++q)
              face_quadrature_points[q] = fe_face_values.quadrature_point(q);

            this->problem->temperature->value_list(face_quadrature_points,
                                                   temperature_face);

            fe_face_values[V_extractor].get_function_values(
              this->current_solution_cell, V_values_face);
            for (unsigned int q = 0; q < n_face_q_points; q++)
              V_values_face[q] *= V_rescale;

            for (unsigned int q = 0; q < n_face_q_points; ++q)
              {
                const double JxW    = fe_face_values.JxW(q);
                const auto   normal = fe_face_values.normal_vector(q);

                const double n_value =
                  this->template compute_density<Component::n>(
                    0., V_values_face[q], temperature_face[q]) /
                  n_rescale;
                const double p_value =
                  this->template compute_density<Component::p>(
                    0., V_values_face[q], temperature_face[q]) /
                  p_rescale;

                for (unsigned int i : Wn_components_on_face[face])
                  {
                    const unsigned int ii = Wn_components[i];
                    const auto v = fe_face_values[Wn_extractor].value(ii, q);
                    Wn_rhs[i] += -(v * normal) * n_value * JxW;
                  }

                for (unsigned int i : Wp_components_on_face[face])
                  {
                    const unsigned int ii = Wp_components[i];
                    const auto v = fe_face_values[Wp_extractor].value(ii, q);
                    Wp_rhs[i] += -(v * normal) * p_value * JxW;
                  }
              }
          }

        n_matrix.compute_lu_factorization();
        n_matrix.solve(n_rhs);
        p_matrix.compute_lu_factorization();
        p_matrix.solve(p_rhs);

        Wn_matrix.compute_lu_factorization();
        Wn_matrix.solve(Wn_rhs);
        Wp_matrix.compute_lu_factorization();
        Wp_matrix.solve(Wp_rhs);

        // Now we copy back the computed values into the global system
        for (unsigned int i = 0; i < n_components.size(); ++i)
          local_values[n_components[i]] = n_rhs[i];

        for (unsigned int i = 0; i < p_components.size(); ++i)
          local_values[p_components[i]] = p_rhs[i];

        for (unsigned int i = 0; i < Wn_components.size(); ++i)
          local_values[Wn_components[i]] = Wn_rhs[i];

        for (unsigned int i = 0; i < Wp_components.size(); ++i)
          local_values[Wp_components[i]] = Wp_rhs[i];

        cell->set_dof_values(local_values, this->current_solution_cell);
      }

    this->project_cell_function_on_trace(
      {Component::n, Component::p},
      TraceProjectionStrategy::reconstruct_problem_solution);

    this->set_enabled_components(current_active_components[Component::V],
                                 current_active_components[Component::n],
                                 current_active_components[Component::p]);

    return iterations;
  }



  template <int dim, class Permittivity>
  NonlinearIterationResults
  NPSolver<dim, Permittivity>::compute_thermodynamic_equilibrium(
    bool generate_first_guess)
  {
    return this->compute_thermodynamic_equilibrium(
      this->parameters->nonlinear_solver_absolute_tolerance,
      this->parameters->nonlinear_solver_relative_tolerance,
      this->parameters->nonlinear_solver_max_number_of_iterations,
      generate_first_guess);
  }


  template class NPSolver<1, HomogeneousPermittivity<1>>;
  template class NPSolver<2, HomogeneousPermittivity<2>>;
  template class NPSolver<3, HomogeneousPermittivity<3>>;
} // namespace Ddhdg