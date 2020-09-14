#include <deal.II/dofs/dof_tools.h>

#include <cmath>

#include "np_solver.h"

namespace Ddhdg
{
  template <int dim, typename Problem>
  void
  NPSolver<dim, Problem>::compute_local_charge_neutrality_on_a_point(
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



  template <int dim, typename Problem>
  void
  NPSolver<dim, Problem>::compute_local_charge_neutrality_on_cells()
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



  template <int dim, typename Problem>
  void
  NPSolver<dim, Problem>::compute_local_charge_neutrality_on_trace(
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
    this->global_constraints.distribute(this->current_solution_trace);
  }



  template <int dim, typename Problem>
  void
  NPSolver<dim, Problem>::compute_local_charge_neutrality()
  {
    this->compute_local_charge_neutrality_on_cells();
    this->compute_local_charge_neutrality_on_trace(false);
  }



  template <int dim, typename Problem>
  NonlinearIterationResults
  NPSolver<dim, Problem>::compute_thermodynamic_equilibrium(
    const double absolute_tol,
    const double relative_tol,
    const int    max_number_of_iterations,
    const bool   generate_first_guess)
  {
    std::map<Component, bool> current_active_components;
    for (Component c : all_primary_components())
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

    // Right now I will disable this code because it seems that it compute
    // values that are not real (like inf or nan). This requires further
    // investigation
    /*this->project_component(
      Component::phi_n,
      std::make_shared<dealii::Functions::ZeroFunction<dim>>());
    this->project_component(
      Component::phi_p,
      std::make_shared<dealii::Functions::ZeroFunction<dim>>());*/

    this->set_enabled_components(current_active_components[Component::V],
                                 current_active_components[Component::n],
                                 current_active_components[Component::p]);

    return iterations;
  }



  template <int dim, typename Problem>
  NonlinearIterationResults
  NPSolver<dim, Problem>::compute_thermodynamic_equilibrium(
    bool generate_first_guess)
  {
    return this->compute_thermodynamic_equilibrium(
      this->parameters->nonlinear_solver_absolute_tolerance,
      this->parameters->nonlinear_solver_relative_tolerance,
      this->parameters->nonlinear_solver_max_number_of_iterations,
      generate_first_guess);
  }


  template class NPSolver<1, HomogeneousProblem<1>>;
  template class NPSolver<2, HomogeneousProblem<2>>;
  template class NPSolver<3, HomogeneousProblem<3>>;
} // namespace Ddhdg