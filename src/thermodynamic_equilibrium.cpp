#include <deal.II/dofs/dof_tools.h>

#include <cmath>

#include "np_solver.h"

namespace Ddhdg
{
  template <int dim>
  void
  NPSolver<dim>::compute_local_charge_neutrality_first_guess(
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

    const double doping_threshold =
      (this->adimensionalizer->doping_magnitude * 1e-5 < 1e2) ?
        this->adimensionalizer->doping_magnitude * 1e-5 :
        1e2;

    const double doping_threshold_square = doping_threshold * doping_threshold;

    constexpr double ev = Ddhdg::Constants::EV;
    const double     Nc = this->problem->band_density.at(Component::n);
    const double     Ec = this->problem->band_edge_energy.at(Component::n) * ev;
    const double     Nv = this->problem->band_density.at(Component::p);
    const double     Ev = this->problem->band_edge_energy.at(Component::p) * ev;

    for (unsigned int i = 0; i < n_of_points; i++)
      {
        if (evaluated_doping[i] * evaluated_doping[i] < doping_threshold_square)
          {
            const double U_T =
              evaluated_temperature[i] * Constants::KB / Constants::Q;
            evaluated_potentials[i] =
              ((Ec + Ev) / (2 * Constants::Q) + U_T / 2 * log(Nv / Nc)) /
              rescale_factor;
          }
        else
          {
            if (evaluated_doping[i] > 0)
              evaluated_potentials[i] =
                this->template compute_potential<Component::n>(
                  evaluated_doping[i], 0, evaluated_temperature[i]) /
                rescale_factor;
            else
              evaluated_potentials[i] =
                this->template compute_potential<Component::p>(
                  -evaluated_doping[i], 0, evaluated_temperature[i]) /
                rescale_factor;
          }
      }
  }



  template <int dim>
  void
  NPSolver<dim>::set_local_charge_neutrality_first_guess()
  {
    const QGauss<dim> quadrature_formula(
      this->get_number_of_quadrature_points());
    const QGauss<dim - 1> face_quadrature_formula(
      this->get_number_of_quadrature_points());

    const unsigned int V_index = get_component_index(Component::V);

    // This is for the cell
    {
      const UpdateFlags flags_cell(update_values | update_gradients |
                                   update_JxW_values |
                                   update_quadrature_points);
      const UpdateFlags flags_trace(update_values | update_normal_vectors |
                                    update_quadrature_points |
                                    update_JxW_values);

      FEValues<dim>     fe_values_cell(*(this->fe_cell),
                                   quadrature_formula,
                                   flags_cell);
      FEFaceValues<dim> fe_face_values(*(this->fe_cell),
                                       face_quadrature_formula,
                                       flags_trace);

      const unsigned int n_q_points = fe_values_cell.get_quadrature().size();
      const unsigned int n_face_q_points =
        fe_face_values.get_quadrature().size();

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

          this->compute_local_charge_neutrality_first_guess(
            evaluated_doping, evaluated_temperature, evaluated_potentials);

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              // Copy data of the shape function
              for (unsigned int k = 0; k < dofs_per_component; ++k)
                {
                  const unsigned int i = on_current_component[k];
                  c_bf[k]     = fe_values_cell[V_extractor].value(i, q);
                  d_bf[k]     = fe_values_cell[E_extractor].value(i, q);
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
              this->problem->temperature->value_list(
                face_quadrature_points, evaluated_temperature_face);

              this->compute_local_charge_neutrality_first_guess(
                evaluated_doping_face,
                evaluated_temperature_face,
                evaluated_potentials_face);

              for (unsigned int q = 0; q < n_face_q_points; ++q)
                {
                  const double JxW    = fe_face_values.JxW(q);
                  const auto   normal = fe_face_values.normal_vector(q);

                  for (unsigned int k = 0; k < dofs_per_face_on_component; ++k)
                    {
                      const auto kk = component_support_on_face[face_number][k];
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

    // This is the part for the trace
    {
      const UpdateFlags flags(update_values | update_normal_vectors |
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

      LAPACKFullMatrix<double> local_trace_matrix(dofs_per_component,
                                                  dofs_per_component);
      Vector<double>           local_trace_residual(dofs_per_component);
      Vector<double>           local_trace_values(dofs_per_cell);

      for (const auto &cell : dof_handler_trace.active_cell_iterators())
        {
          local_trace_matrix   = 0;
          local_trace_residual = 0;
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              fe_face_trace_values.reinit(cell, face);
              for (unsigned int q = 0; q < n_face_q_points; ++q)
                face_quadrature_points[q] =
                  fe_face_trace_values.quadrature_point(q);

              this->problem->doping->value_list(face_quadrature_points,
                                                evaluated_doping);
              this->problem->temperature->value_list(face_quadrature_points,
                                                     evaluated_temperature);

              this->compute_local_charge_neutrality_first_guess(
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
                      const unsigned int ii =
                        component_support_on_face[face][i];
                      for (unsigned int j = 0; j < dofs_per_face_on_component;
                           ++j)
                        {
                          const unsigned int jj =
                            component_support_on_face[face][j];
                          local_trace_matrix(ii, jj) +=
                            (c_bf[j] * c_bf[i]) * JxW;
                        }
                      local_trace_residual[ii] +=
                        (evaluated_potentials[q] * c_bf[i]) * JxW;
                    }
                }
            }
          local_trace_matrix.compute_lu_factorization();
          local_trace_matrix.solve(local_trace_residual);

          cell->get_dof_values(this->current_solution_trace,
                               local_trace_values);
          for (unsigned int i = 0; i < dofs_per_component; i++)
            local_trace_values[on_current_component[i]] =
              local_trace_residual[i];
          cell->set_dof_values(local_trace_values,
                               this->current_solution_trace);
        }
    }
  }



  template <int dim>
  void
  NPSolver<dim>::compute_local_charge_neutrality_copy_solution(
    const dealii::FiniteElement<dim> &V_fe,
    const dealii::DoFHandler<dim> &   V_dof_handler,
    Vector<double> &                  current_solution)
  {
    std::vector<dealii::types::global_dof_index> global_indices(
      V_fe.n_dofs_per_cell());
    std::vector<dealii::types::global_dof_index> all_system_global_indices(
      this->fe_cell->n_dofs_per_cell());

    const unsigned int V_index = get_component_index(Component::V);

    const dealii::ComponentMask V_mask = this->get_component_mask(Component::V);
    const dealii::ComponentMask E_mask =
      this->get_component_mask(Displacement::E);
    const dealii::ComponentMask       total_mask = V_mask | E_mask;
    const dealii::FiniteElement<dim> &all_system_V_fe_system =
      this->fe_cell->get_sub_fe(total_mask.first_selected_component(),
                                total_mask.n_selected_components());

    for (const auto &cell : V_dof_handler.active_cell_iterators())
      {
        typename DoFHandler<dim>::active_cell_iterator all_system_cell(
          &(*triangulation), cell->level(), cell->index(), &dof_handler_cell);

        cell->get_dof_indices(global_indices);
        all_system_cell->get_dof_indices(all_system_global_indices);

        for (unsigned int i = 0; i < this->fe_cell->n_dofs_per_cell(); i++)
          {
            const dealii::types::global_dof_index as_gi =
              all_system_global_indices[i];

            const auto block_position = this->fe_cell->system_to_block_index(i);

            const unsigned int current_block       = block_position.first;
            const unsigned int current_block_index = block_position.second;

            if (current_block != V_index)
              continue;

            const auto V_block_position =
              all_system_V_fe_system.system_to_block_index(current_block_index);

            const unsigned int V_current_block       = V_block_position.first;
            const unsigned int V_current_block_index = V_block_position.second;

            if (V_current_block != 1)
              continue;

            current_solution[global_indices[V_current_block_index]] =
              this->current_solution_cell[as_gi];
          }
      }
  }



  template <int dim>
  void
  NPSolver<dim>::compute_local_charge_neutrality_nonlinear_solver(
    const dealii::DoFHandler<dim> &V_dof_handler,
    Vector<double> &               current_solution)
  {
    constexpr double       ABS_TOL                  = 1e-14;
    constexpr double       REL_TOL                  = 1e-10;
    constexpr unsigned int MAX_NUMBER_OF_ITERATIONS = 10000;

    dealii::Vector<double> doping_values(V_dof_handler.n_dofs());
    dealii::Vector<double> temperature_values(V_dof_handler.n_dofs());

    dealii::VectorTools::interpolate(V_dof_handler,
                                     *(this->problem->doping),
                                     doping_values);
    dealii::VectorTools::interpolate(V_dof_handler,
                                     *(this->problem->temperature),
                                     temperature_values);

    const double Nc = this->problem->band_density.at(Component::n);
    const double Nv = this->problem->band_density.at(Component::p);
    const double Ec =
      this->problem->band_edge_energy.at(Component::n) * Constants::EV;
    const double Ev =
      this->problem->band_edge_energy.at(Component::p) * Constants::EV;

    const double V_rescale =
      this->adimensionalizer
        ->template get_component_rescaling_factor<Component::V>();

    const unsigned int n_dofs = V_dof_handler.n_dofs();
    for (unsigned int i = 0; i < n_dofs; i++)
      {
        const double T   = temperature_values[i];
        const double c   = doping_values[i];
        const double KbT = Constants::KB * T;

        const double current_c_rescale = (std::abs(c) < 1) ? 1 : std::abs(c);

        const double coeff =
          (V_rescale * Constants::Q) / (Constants::KB * T * current_c_rescale);

        std::function<double(const double)> residual = [=](const double v) {
          // Compute the exponents that appears in the function
          const double exp_n = (V_rescale * v * Constants::Q - Ec) / KbT;

          const double exp_p = (Ev - V_rescale * v * Constants::Q) / KbT;

          const double residual =
            (c - Nc * exp(exp_n) + Nv * exp(exp_p)) / current_c_rescale;
          return residual;
        };

        std::function<double(const double, const double)>
          solve_jacobian_system = [=](const double v, const double rhs) {
            const double exp_n = (V_rescale * v * Constants::Q - Ec) / KbT;

            const double exp_p = (Ev - V_rescale * v * Constants::Q) / KbT;

            const double dF =
              -Nc * coeff * exp(exp_n) - Nv * coeff * exp(exp_p);

            return -rhs / dF;
          };

        unsigned int current_iteration = 0;
        bool         converged         = false;
        double       current_v         = current_solution[i];
        double       rhs               = residual(current_v);
        double       temp_rhs;
        double       delta_v;
        int          float_type;

        while (current_iteration < MAX_NUMBER_OF_ITERATIONS)
          {
            delta_v = solve_jacobian_system(current_v, rhs);

            // Avoid movements bigger than 1
            if (std::abs(delta_v) > 1.)
              delta_v = std::abs(delta_v) / delta_v;

            temp_rhs = residual(current_v + delta_v);

            // Do not move in place that increase the residual
            while (std::fpclassify(temp_rhs) == FP_INFINITE ||
                   std::abs(rhs) < std::abs(temp_rhs))
              {
                delta_v /= 2.;
                temp_rhs = residual(current_v + delta_v);
                if (std::abs(delta_v) < ABS_TOL)
                  break;
              }
            current_v += delta_v;
            rhs = temp_rhs;

            if (std::abs(delta_v) < ABS_TOL)
              {
                converged = true;
                break;
              }

            if (std::abs(delta_v / current_v) < REL_TOL)
              {
                converged = true;
                break;
              }

            float_type = std::fpclassify(rhs);
            if (float_type == FP_NAN || float_type == FP_INFINITE)
              break;

            ++current_iteration;
          }
        float_type = std::fpclassify(current_v);

        if (!converged || float_type == FP_NAN || float_type == FP_INFINITE)
          AssertThrow(
            false,
            ExcMessage(
              "Newton method has not converged during computation of local "
              "charge neutrality"));

        current_solution[i] = current_v;
      }
  }



  template <int dim>
  void
  NPSolver<dim>::compute_local_charge_neutrality_set_solution(
    const dealii::FiniteElement<dim> &V_fe,
    const dealii::DoFHandler<dim> &   V_dof_handler,
    Vector<double> &                  current_solution)
  {
    const UpdateFlags flags_cell(update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points);
    const UpdateFlags flags_face(update_values | update_normal_vectors |
                                 update_quadrature_points | update_JxW_values);
    const UpdateFlags V_flags_cell(update_values | update_quadrature_points);
    const UpdateFlags V_flags_face(update_values | update_quadrature_points);

    const QGauss<dim> quadrature_formula(
      this->get_number_of_quadrature_points());
    const QGauss<dim - 1> face_quadrature_formula(
      this->get_number_of_quadrature_points());

    FEValues<dim>     fe_values_cell(*(this->fe_cell),
                                 quadrature_formula,
                                 flags_cell);
    FEFaceValues<dim> fe_face_values(*(this->fe_cell),
                                     face_quadrature_formula,
                                     flags_face);
    FEValues<dim>     V_fe_values(V_fe, quadrature_formula, V_flags_cell);
    FEFaceValues<dim> V_fe_face_values(V_fe,
                                       face_quadrature_formula,
                                       V_flags_face);

    const unsigned int n_q_points      = fe_values_cell.get_quadrature().size();
    const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();

    // Now we need to map the dofs that are related to the component V
    const unsigned int        V_index = get_component_index(Component::V);
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

    // Temporary buffers for the values of the local base function on a
    // quadrature point
    std::vector<double>         V_bf(dofs_per_component);
    std::vector<Tensor<1, dim>> E_bf(dofs_per_component);
    std::vector<double>         E_div_bf(dofs_per_component);

    std::vector<double> charge_neutrality_values(n_q_points);
    std::vector<double> charge_neutrality_values_face(n_face_q_points);

    for (const auto &cell : this->dof_handler_cell.active_cell_iterators())
      {
        local_matrix   = 0.;
        local_residual = 0.;

        fe_values_cell.reinit(cell);

        typename DoFHandler<dim>::active_cell_iterator V_cell(&(*triangulation),
                                                              cell->level(),
                                                              cell->index(),
                                                              &V_dof_handler);
        V_fe_values.reinit(V_cell);

        // Get the values of the charge neutrality for this cell
        V_fe_values.get_function_values(current_solution,
                                        charge_neutrality_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            // Copy data of the shape function
            for (unsigned int k = 0; k < dofs_per_component; ++k)
              {
                const unsigned int i = on_current_component[k];
                V_bf[k]              = fe_values_cell[V_extractor].value(i, q);
                E_bf[k]              = fe_values_cell[E_extractor].value(i, q);
                E_div_bf[k] = fe_values_cell[E_extractor].divergence(i, q);
              }

            const double JxW = fe_values_cell.JxW(q);

            for (unsigned int i = 0; i < dofs_per_component; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_component; ++j)
                  {
                    local_matrix(i, j) +=
                      (V_bf[j] * V_bf[i] + E_bf[i] * E_bf[j]) * JxW;
                  }
                local_residual[i] +=
                  (charge_neutrality_values[q] * (V_bf[i] + E_div_bf[i])) * JxW;
              }
          }
        for (unsigned int face_number = 0;
             face_number < GeometryInfo<dim>::faces_per_cell;
             ++face_number)
          {
            fe_face_values.reinit(cell, face_number);
            V_fe_face_values.reinit(V_cell, face_number);

            V_fe_face_values.get_function_values(current_solution,
                                                 charge_neutrality_values_face);

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
                    local_residual[kk] += (-charge_neutrality_values_face[q] *
                                           (f_bf_face * normal)) *
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



  template <int dim>
  void
  NPSolver<dim>::compute_local_charge_neutrality()
  {
    const dealii::ComponentMask V_component_mask =
      this->get_component_mask(Component::V);

    const unsigned int fsc = V_component_mask.first_selected_component();
    const unsigned int nsc = V_component_mask.n_selected_components();

    const dealii::FiniteElement<dim> &V_fe =
      this->fe_cell->get_sub_fe(fsc, nsc);

    dealii::DoFHandler<dim> V_dof_handler(*(this->triangulation));
    V_dof_handler.distribute_dofs(V_fe);

    Vector<double> current_solution(V_dof_handler.n_dofs());

    // Copy the current solution (which has all the components) to a smaller
    // system only in V
    this->compute_local_charge_neutrality_copy_solution(V_fe,
                                                        V_dof_handler,
                                                        current_solution);

    // Solve the problem of the local charge neutrality using Newton
    this->compute_local_charge_neutrality_nonlinear_solver(V_dof_handler,
                                                           current_solution);

    // Copy back the local charge neutrality function in the overall system
    this->compute_local_charge_neutrality_set_solution(V_fe,
                                                       V_dof_handler,
                                                       current_solution);

    // So far we have fixed only the cell values. Now we copy the values for the
    // cells on the trace
    this->project_cell_function_on_trace();
  }



  template <int dim>
  NonlinearIterationResults
  NPSolver<dim>::compute_thermodynamic_equilibrium(
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
      {
        this->set_local_charge_neutrality_first_guess();
        this->compute_local_charge_neutrality();
      }
    NonlinearIterationResults iterations = this->private_run(
      absolute_tol, relative_tol, max_number_of_iterations, true);

    this->set_enabled_components(current_active_components[Component::V],
                                 current_active_components[Component::n],
                                 current_active_components[Component::p]);

    return iterations;
  }



  template <int dim>
  NonlinearIterationResults
  NPSolver<dim>::compute_thermodynamic_equilibrium(bool generate_first_guess)
  {
    return this->compute_thermodynamic_equilibrium(
      this->parameters->nonlinear_solver_absolute_tolerance,
      this->parameters->nonlinear_solver_relative_tolerance,
      this->parameters->nonlinear_solver_max_number_of_iterations,
      generate_first_guess);
  }


  template class NPSolver<1>;
  template class NPSolver<2>;
  template class NPSolver<3>;
} // namespace Ddhdg