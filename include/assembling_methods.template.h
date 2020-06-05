#include <deal.II/base/work_stream.h>

#include "constants.h"
#include "templatized_parameters.h"

namespace Ddhdg
{
  template <int dim, class Permittivity>
  void
  NPSolver<dim, Permittivity>::assemble_system_multithreaded(
    const bool trace_reconstruct,
    const bool compute_thermodynamic_equilibrium)
  {
    Assert(this->initialized, dealii::ExcNotInitialized());

    const QGauss<dim> quadrature_formula(
      this->get_number_of_quadrature_points());
    const QGauss<dim - 1> face_quadrature_formula(
      this->get_number_of_quadrature_points());

    const UpdateFlags flags_cell(update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points);
    const UpdateFlags face_flags_cell(update_values);
    const UpdateFlags flags_trace(update_values);
    const UpdateFlags flags_trace_restricted(
      update_values | update_normal_vectors | update_quadrature_points |
      update_JxW_values);
    PerTaskData task_data(this->fe_trace_restricted->dofs_per_cell,
                          trace_reconstruct);
    ScratchData scratch(*(this->fe_trace_restricted),
                        *(this->fe_trace),
                        *(this->fe_cell),
                        quadrature_formula,
                        face_quadrature_formula,
                        flags_cell,
                        face_flags_cell,
                        flags_trace,
                        flags_trace_restricted,
                        *(this->problem->permittivity),
                        this->enabled_components);
    WorkStream::run(dof_handler_trace_restricted.begin_active(),
                    dof_handler_trace_restricted.end(),
                    *this,
                    this->get_assemble_system_one_cell_function(
                      compute_thermodynamic_equilibrium),
                    &NPSolver<dim, Permittivity>::copy_local_to_global,
                    scratch,
                    task_data);
  }

  template <int dim, class Permittivity>
  void
  NPSolver<dim, Permittivity>::assemble_system(
    const bool trace_reconstruct,
    const bool compute_thermodynamic_equilibrium)
  {
    Assert(this->initialized, dealii::ExcNotInitialized());

    const QGauss<dim> quadrature_formula(
      this->get_number_of_quadrature_points());
    const QGauss<dim - 1> face_quadrature_formula(
      this->get_number_of_quadrature_points());

    const UpdateFlags flags_cell(update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points);
    const UpdateFlags face_flags_cell(update_values);
    const UpdateFlags flags_trace(update_values);
    const UpdateFlags flags_trace_restricted(
      update_values | update_normal_vectors | update_quadrature_points |
      update_JxW_values);

    PerTaskData task_data(this->fe_trace_restricted->dofs_per_cell,
                          trace_reconstruct);
    ScratchData scratch(*(this->fe_trace_restricted),
                        *(this->fe_trace),
                        *(this->fe_cell),
                        quadrature_formula,
                        face_quadrature_formula,
                        flags_cell,
                        face_flags_cell,
                        flags_trace,
                        flags_trace_restricted,
                        *(this->problem->permittivity),
                        this->enabled_components);

    for (const auto &cell :
         this->dof_handler_trace_restricted.active_cell_iterators())
      {
        (this->*get_assemble_system_one_cell_function(
                  compute_thermodynamic_equilibrium))(cell, scratch, task_data);
        copy_local_to_global(task_data);
      }
  }



  template <int dim, class Permittivity>
  typename NPSolver<dim, Permittivity>::assemble_system_one_cell_pointer
  NPSolver<dim, Permittivity>::get_assemble_system_one_cell_function(
    const bool compute_thermodynamic_equilibrium)
  {
    // The last three bits of parameter_mask are the values of the flags
    // this->is_enabled(Component::V), this->is_enabled(Component::n) and
    // this->is_enabled(Component::p); as usual 1 is for TRUE, 0 is FALSE.
    unsigned int parameter_mask = 0;
    if (compute_thermodynamic_equilibrium)
      parameter_mask += 8;
    if (this->is_enabled(Component::V))
      parameter_mask += 4;
    if (this->is_enabled(Component::n))
      parameter_mask += 2;
    if (this->is_enabled(Component::p))
      parameter_mask += 1;

    TemplatizedParametersInterface<dim, Permittivity> *p1;
    TemplatizedParametersInterface<dim, Permittivity> *p2;

    p1 = new TemplatizedParameters<dim, Permittivity, 15>();
    while (p1->get_parameter_mask() != parameter_mask)
      {
        p2 = p1->get_previous();
        delete p1;
        p1 = p2;
      }

    typename NPSolver<dim, Permittivity>::assemble_system_one_cell_pointer f =
      p1->get_assemble_system_one_cell_function();
    delete p1;
    return f;
  }



  template <int dim, class Permittivity>
  void
  NPSolver<dim, Permittivity>::prepare_data_on_cell_quadrature_points(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch)
  {
    const unsigned int n_q_points =
      scratch.fe_values_cell.get_quadrature().size();

    // Copy the values of the previous solution regarding the previous cell in
    // the scratch. This must be done for every component because, for
    // example, the equation in n requires the data from p and so on
    for (const auto c : Ddhdg::all_components())
      {
        const Displacement d = component2displacement(c);

        const FEValuesExtractors::Scalar c_extractor =
          this->get_component_extractor(c);
        const FEValuesExtractors::Vector d_extractor =
          this->get_displacement_extractor(d);

        scratch.fe_values_cell[c_extractor].get_function_values(
          current_solution_cell, scratch.previous_c_cell.at(c));
        scratch.fe_values_cell[d_extractor].get_function_values(
          current_solution_cell, scratch.previous_d_cell.at(c));
      }

    // Get the position of the quadrature points
    for (unsigned int q = 0; q < n_q_points; ++q)
      scratch.cell_quadrature_points[q] =
        scratch.fe_values_cell.quadrature_point(q);

    // Compute the value of epsilon
    scratch.permittivity.initialize_on_cell(
      scratch.cell_quadrature_points,
      this->adimensionalizer->get_permittivity_rescaling_factor());

    // Compute the value of mu
    if (this->is_enabled(Component::n))
      {
        this->problem->n_electron_mobility->compute_electron_mobility(
          scratch.cell_quadrature_points, scratch.mu_n_cell);
        this->adimensionalizer->template adimensionalize_electron_mobility<dim>(
          scratch.mu_n_cell);
      }

    if (this->is_enabled(Component::p))
      {
        this->problem->p_electron_mobility->compute_electron_mobility(
          scratch.cell_quadrature_points, scratch.mu_p_cell);
        this->adimensionalizer->template adimensionalize_hole_mobility<dim>(
          scratch.mu_p_cell);
      }

    // Compute the value of T
    this->problem->temperature->value_list(scratch.cell_quadrature_points,
                                           scratch.T_cell);

    // Compute the thermal voltage
    const double thermal_voltage_rescaling_factor =
      this->adimensionalizer->get_thermal_voltage_rescaling_factor();
    for (unsigned int q = 0; q < n_q_points; ++q)
      scratch.U_T_cell[q] = (Constants::KB * scratch.T_cell[q] / Constants::Q) /
                            thermal_voltage_rescaling_factor;


    // Compute the value of the doping
    if (this->is_enabled(Component::V))
      this->rescaled_doping->value_list(scratch.cell_quadrature_points,
                                        scratch.doping_cell);

    // Compute the value of the recombination term and its derivative respect
    // to n and p
    if (this->is_enabled(Component::n) || this->is_enabled(Component::p))
      {
        auto &dr_n = scratch.dr_cell.at(Component::n);
        auto &dr_p = scratch.dr_cell.at(Component::p);

        this->problem->recombination_term->compute_multiple_recombination_terms(
          scratch.previous_c_cell.at(Component::n),
          scratch.previous_c_cell.at(Component::p),
          scratch.cell_quadrature_points,
          scratch.r_cell);
        this->problem->recombination_term
          ->compute_multiple_derivatives_of_recombination_terms(
            scratch.previous_c_cell.at(Component::n),
            scratch.previous_c_cell.at(Component::p),
            scratch.cell_quadrature_points,
            Component::n,
            dr_n);
        this->problem->recombination_term
          ->compute_multiple_derivatives_of_recombination_terms(
            scratch.previous_c_cell.at(Component::n),
            scratch.previous_c_cell.at(Component::p),
            scratch.cell_quadrature_points,
            Component::p,
            dr_p);

        this->adimensionalizer->adimensionalize_recombination_term(
          scratch.r_cell, dr_n, dr_p);
      }
  }



  template <int dim, class Permittivity>
  template <typename prm>
  void
  NPSolver<dim, Permittivity>::add_cell_products_to_cc_matrix(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch)
  {
    const unsigned int n_q_points =
      scratch.fe_values_cell.get_quadrature().size();
    const unsigned int loc_dofs_per_cell =
      scratch.fe_values_cell.get_fe().dofs_per_cell;

    const FEValuesExtractors::Vector electric_field =
      this->get_displacement_extractor(Displacement::E);
    const FEValuesExtractors::Scalar electric_potential =
      this->get_component_extractor(Component::V);
    const FEValuesExtractors::Vector electron_displacement =
      this->get_displacement_extractor(Displacement::Wn);
    const FEValuesExtractors::Scalar electron_density =
      this->get_component_extractor(Component::n);
    const FEValuesExtractors::Vector hole_displacement =
      this->get_displacement_extractor(Displacement::Wp);
    const FEValuesExtractors::Scalar hole_density =
      this->get_component_extractor(Component::p);

    // The following are just aliases (and some of them refer to the same
    // vector). These may seem useless, but the make the code that assemble
    // the matrix a lot more understandable
    auto &E = scratch.d.at(Component::V);
    auto &V = scratch.c.at(Component::V);

    auto &Wn = scratch.d.at(Component::n);
    auto &n  = scratch.c.at(Component::n);

    auto &Wp = scratch.d.at(Component::p);
    auto &p  = scratch.c.at(Component::p);

    auto &q1      = scratch.d.at(Component::V);
    auto &q1_div  = scratch.d_div.at(Component::V);
    auto &z1      = scratch.c.at(Component::V);
    auto &z1_grad = scratch.c_grad.at(Component::V);

    auto &q2      = scratch.d.at(Component::n);
    auto &q2_div  = scratch.d_div.at(Component::n);
    auto &z2      = scratch.c.at(Component::n);
    auto &z2_grad = scratch.c_grad.at(Component::n);

    auto &q3      = scratch.d.at(Component::p);
    auto &q3_div  = scratch.d_div.at(Component::p);
    auto &z3      = scratch.c.at(Component::p);
    auto &z3_grad = scratch.c_grad.at(Component::p);

    const auto &V0 = scratch.previous_c_cell.at(Component::V);
    const auto &n0 = scratch.previous_c_cell.at(Component::n);
    const auto &p0 = scratch.previous_c_cell.at(Component::p);
    const auto &E0 = scratch.previous_d_cell.at(Component::V);

    const unsigned int dofs_per_component =
      scratch.enabled_component_indices.size();

    dealii::Tensor<1, dim> epsilon_times_E;

    dealii::Tensor<1, dim> mu_n_times_previous_E;
    dealii::Tensor<1, dim> mu_p_times_previous_E;

    dealii::Tensor<2, dim> n_einstein_diffusion_coefficient;
    dealii::Tensor<2, dim> p_einstein_diffusion_coefficient;

    const std::vector<double> &dr_n = scratch.dr_cell.at(Component::n);
    const std::vector<double> &dr_p = scratch.dr_cell.at(Component::p);

    const double nc = this->problem->band_density.at(Component::n);
    const double nv = this->problem->band_density.at(Component::p);
    const double ec =
      this->problem->band_edge_energy.at(Component::n) * Constants::EV;
    const double ev =
      this->problem->band_edge_energy.at(Component::p) * Constants::EV;

    const double V_rescale =
      this->adimensionalizer
        ->template get_component_rescaling_factor<Component::V>();
    const double n_rescale =
      this->adimensionalizer
        ->template get_component_rescaling_factor<Component::n>();

    double thermodynamic_equilibrium_der = 0.;

    const double Q =
      this->adimensionalizer->get_poisson_equation_density_constant();

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double JxW = scratch.fe_values_cell.JxW(q);
        for (unsigned int k = 0; k < loc_dofs_per_cell; ++k)
          {
            if (prm::is_V_enabled)
              {
                q1[k] = scratch.fe_values_cell[electric_field].value(k, q);
                q1_div[k] =
                  scratch.fe_values_cell[electric_field].divergence(k, q);
                z1[k] = scratch.fe_values_cell[electric_potential].value(k, q);
                z1_grad[k] =
                  scratch.fe_values_cell[electric_potential].gradient(k, q);
              }

            if (prm::is_n_enabled)
              {
                q2[k] =
                  scratch.fe_values_cell[electron_displacement].value(k, q);
                q2_div[k] =
                  scratch.fe_values_cell[electron_displacement].divergence(k,
                                                                           q);
                z2[k] = scratch.fe_values_cell[electron_density].value(k, q);
                z2_grad[k] =
                  scratch.fe_values_cell[electron_density].gradient(k, q);
              }

            if (prm::is_p_enabled)
              {
                q3[k] = scratch.fe_values_cell[hole_displacement].value(k, q);
                q3_div[k] =
                  scratch.fe_values_cell[hole_displacement].divergence(k, q);
                z3[k] = scratch.fe_values_cell[hole_density].value(k, q);
                z3_grad[k] =
                  scratch.fe_values_cell[hole_density].gradient(k, q);
              }
          }

        if (prm::is_n_enabled)
          mu_n_times_previous_E = scratch.mu_n_cell[q] * E0[q];

        if (prm::is_p_enabled)
          mu_p_times_previous_E = scratch.mu_p_cell[q] * E0[q];

        if (prm::is_n_enabled)
          n_einstein_diffusion_coefficient =
            scratch
              .template compute_einstein_diffusion_coefficient<Component::n,
                                                               false>(q);
        if (prm::is_p_enabled)
          p_einstein_diffusion_coefficient =
            scratch
              .template compute_einstein_diffusion_coefficient<Component::p,
                                                               false>(q);

        if (prm::thermodyn_eq)
          {
            const double KbT_over_q =
              Constants::KB * scratch.T_cell[q] / (Constants::Q * V_rescale);
            const double ev_rescaled = ev / (Constants::Q * V_rescale);
            const double ec_rescaled = ec / (Constants::Q * V_rescale);

            thermodynamic_equilibrium_der =
              -Q / n_rescale *
              (nv / KbT_over_q * exp((ev_rescaled - V0[q]) / KbT_over_q) +
               nc / KbT_over_q * exp((V0[q] - ec_rescaled) / KbT_over_q));
          }

        for (unsigned int i = 0; i < dofs_per_component; ++i)
          {
            const unsigned int ii = scratch.enabled_component_indices[i];
            for (unsigned int j = 0; j < dofs_per_component; ++j)
              {
                const unsigned int jj = scratch.enabled_component_indices[j];
                if (prm::is_V_enabled)
                  {
                    scratch.permittivity.epsilon_operator_on_cell(
                      q, E[jj], epsilon_times_E);
                    scratch.cc_matrix(i, j) +=
                      (-V[jj] * q1_div[ii] + E[jj] * q1[ii] -
                       epsilon_times_E * z1_grad[ii]) *
                      JxW;

                    if (prm::thermodyn_eq)
                      scratch.cc_matrix(i, j) +=
                        -thermodynamic_equilibrium_der * V[jj] * z1[ii] * JxW;
                    else
                      scratch.cc_matrix(i, j) +=
                        Q * (n[jj] - p[jj]) * z1[ii] * JxW;
                  }
                if (prm::is_n_enabled)
                  scratch.cc_matrix(i, j) +=
                    (-n[jj] * q2_div[ii] + Wn[jj] * q2[ii] -
                     n[jj] * (mu_n_times_previous_E * z2_grad[ii]) +
                     n0[q] * ((scratch.mu_n_cell[q] * E[jj]) * z2_grad[ii]) +
                     (n_einstein_diffusion_coefficient * Wn[jj]) * z2_grad[ii] -
                     (dr_n[q] * n[jj] + dr_p[q] * p[jj]) * z2[ii]) *
                    JxW;
                if (prm::is_p_enabled)
                  scratch.cc_matrix(i, j) +=
                    (-p[jj] * q3_div[ii] + Wp[jj] * q3[ii] +
                     p[jj] * (mu_p_times_previous_E * z3_grad[ii]) -
                     p0[q] * ((scratch.mu_p_cell[q] * E[jj]) * z3_grad[ii]) +
                     (p_einstein_diffusion_coefficient * Wp[jj]) * z3_grad[ii] -
                     (dr_n[q] * n[jj] + dr_p[q] * p[jj]) * z3[ii]) *
                    JxW;
              }
          }
      }
  }



  template <int dim, class Permittivity>
  template <typename prm>
  void
  NPSolver<dim, Permittivity>::add_cell_products_to_cc_rhs(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch)
  {
    const unsigned int n_q_points =
      scratch.fe_values_cell.get_quadrature().size();
    const unsigned int dofs_per_component =
      scratch.enabled_component_indices.size();

    const FEValuesExtractors::Vector electric_field =
      this->get_displacement_extractor(Displacement::E);
    const FEValuesExtractors::Scalar electric_potential =
      this->get_component_extractor(Component::V);
    const FEValuesExtractors::Vector electron_displacement =
      this->get_displacement_extractor(Displacement::Wn);
    const FEValuesExtractors::Scalar electron_density =
      this->get_component_extractor(Component::n);
    const FEValuesExtractors::Vector hole_displacement =
      this->get_displacement_extractor(Displacement::Wp);
    const FEValuesExtractors::Scalar hole_density =
      this->get_component_extractor(Component::p);

    const auto &V0  = scratch.previous_c_cell.at(Component::V);
    const auto &E0  = scratch.previous_d_cell.at(Component::V);
    const auto &n0  = scratch.previous_c_cell.at(Component::n);
    const auto &Wn0 = scratch.previous_d_cell.at(Component::n);
    const auto &p0  = scratch.previous_c_cell.at(Component::p);
    const auto &Wp0 = scratch.previous_d_cell.at(Component::p);

    const auto &c0 = scratch.doping_cell;

    dealii::Tensor<1, dim> epsilon_times_E;

    dealii::Tensor<2, dim> n_einstein_diffusion_coefficient;
    dealii::Tensor<2, dim> p_einstein_diffusion_coefficient;

    dealii::Tensor<1, dim> Jn;
    dealii::Tensor<1, dim> Jp;

    const double nc = this->problem->band_density.at(Component::n);
    const double nv = this->problem->band_density.at(Component::p);
    const double ec =
      this->problem->band_edge_energy.at(Component::n) * Constants::EV;
    const double ev =
      this->problem->band_edge_energy.at(Component::p) * Constants::EV;

    double thermodynamic_equilibrium_rhs = 0.;

    const double V_rescale =
      this->adimensionalizer
        ->template get_component_rescaling_factor<Component::V>();
    const double n_rescale =
      this->adimensionalizer
        ->template get_component_rescaling_factor<Component::n>();
    const double Q =
      this->adimensionalizer->get_poisson_equation_density_constant();

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double JxW = scratch.fe_values_cell.JxW(q);

        if (prm::is_V_enabled)
          scratch.permittivity.epsilon_operator_on_cell(q,
                                                        E0[q],
                                                        epsilon_times_E);

        if (prm::is_n_enabled)
          n_einstein_diffusion_coefficient =
            scratch
              .template compute_einstein_diffusion_coefficient<Component::n,
                                                               false>(q);
        if (prm::is_p_enabled)
          p_einstein_diffusion_coefficient =
            scratch
              .template compute_einstein_diffusion_coefficient<Component::p,
                                                               false>(q);

        if (prm::is_n_enabled)
          Jn = n0[q] * (scratch.mu_n_cell[q] * E0[q]) -
               (n_einstein_diffusion_coefficient * Wn0[q]);
        if (prm::is_p_enabled)
          Jp = -p0[q] * (scratch.mu_p_cell[q] * E0[q]) -
               (p_einstein_diffusion_coefficient * Wp0[q]);

        if (prm::thermodyn_eq)
          {
            const double KbT_over_q =
              Constants::KB * scratch.T_cell[q] / (Constants::Q * V_rescale);
            const double ev_rescaled = ev / (Constants::Q * V_rescale);
            const double ec_rescaled = ec / (Constants::Q * V_rescale);
            thermodynamic_equilibrium_rhs =
              Q / n_rescale *
              (nv * exp((ev_rescaled - V0[q]) / KbT_over_q) -
               nc * exp((V0[q] - ec_rescaled) / KbT_over_q));
          }

        for (unsigned int i = 0; i < dofs_per_component; ++i)
          {
            const unsigned int ii = scratch.enabled_component_indices[i];
            if (prm::is_V_enabled)
              {
                const dealii::Tensor<1, dim> q1 =
                  scratch.fe_values_cell[electric_field].value(ii, q);
                const double q1_div =
                  scratch.fe_values_cell[electric_field].divergence(ii, q);
                const double z1 =
                  scratch.fe_values_cell[electric_potential].value(ii, q);
                const dealii::Tensor<1, dim> z1_grad =
                  scratch.fe_values_cell[electric_potential].gradient(ii, q);

                scratch.cc_rhs[i] +=
                  (V0[q] * q1_div - E0[q] * q1 + epsilon_times_E * z1_grad +
                   Q * c0[q] * z1) *
                  JxW;

                if (prm::thermodyn_eq)
                  scratch.cc_rhs[i] += thermodynamic_equilibrium_rhs * z1 * JxW;
                else
                  scratch.cc_rhs[i] += Q * (-n0[q] + p0[q]) * z1 * JxW;
              }

            if (prm::is_n_enabled)
              {
                const dealii::Tensor<1, dim> q2 =
                  scratch.fe_values_cell[electron_displacement].value(ii, q);
                const double q2_div =
                  scratch.fe_values_cell[electron_displacement].divergence(ii,
                                                                           q);
                const double z2 =
                  scratch.fe_values_cell[electron_density].value(ii, q);
                const dealii::Tensor<1, dim> z2_grad =
                  scratch.fe_values_cell[electron_density].gradient(ii, q);

                scratch.cc_rhs[i] += (n0[q] * q2_div - Wn0[q] * q2 +
                                      scratch.r_cell[q] * z2 + Jn * z2_grad) *
                                     JxW;
              }

            if (prm::is_p_enabled)
              {
                const dealii::Tensor<1, dim> q3 =
                  scratch.fe_values_cell[hole_displacement].value(ii, q);
                const double q3_div =
                  scratch.fe_values_cell[hole_displacement].divergence(ii, q);
                const double z3 =
                  scratch.fe_values_cell[hole_density].value(ii, q);
                const dealii::Tensor<1, dim> z3_grad =
                  scratch.fe_values_cell[hole_density].gradient(ii, q);

                scratch.cc_rhs[i] += (p0[q] * q3_div - Wp0[q] * q3 +
                                      +scratch.r_cell[q] * z3 + Jp * z3_grad) *
                                     JxW;
              }
          }
      }
  }



  template <int dim, class Permittivity>
  void
  NPSolver<dim, Permittivity>::prepare_data_on_face_quadrature_points(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      scratch.face_quadrature_points[q] =
        scratch.fe_face_values_trace_restricted.quadrature_point(q);

    for (const auto c : Ddhdg::all_components())
      {
        const Displacement d = component2displacement(c);

        const FEValuesExtractors::Scalar c_extractor =
          this->get_component_extractor(c);
        const FEValuesExtractors::Vector d_extractor =
          this->get_displacement_extractor(d);
        const FEValuesExtractors::Scalar tr_c_extractor =
          this->get_trace_component_extractor(c);

        scratch.fe_face_values_cell[c_extractor].get_function_values(
          current_solution_cell, scratch.previous_c_face[c]);
        scratch.fe_face_values_cell[d_extractor].get_function_values(
          current_solution_cell, scratch.previous_d_face[c]);
        scratch.fe_face_values_trace[tr_c_extractor].get_function_values(
          current_solution_trace, scratch.previous_tr_c_face[c]);
      }

    scratch.permittivity.initialize_on_face(
      scratch.face_quadrature_points,
      this->adimensionalizer->get_permittivity_rescaling_factor());

    if (this->is_enabled(Component::n))
      {
        this->problem->n_electron_mobility->compute_electron_mobility(
          scratch.face_quadrature_points, scratch.mu_n_face);
        this->adimensionalizer->template adimensionalize_electron_mobility<dim>(
          scratch.mu_n_face);
      }

    if (this->is_enabled(Component::p))
      {
        this->problem->p_electron_mobility->compute_electron_mobility(
          scratch.face_quadrature_points, scratch.mu_p_face);
        this->adimensionalizer->template adimensionalize_electron_mobility<dim>(
          scratch.mu_p_face);
      }

    this->problem->temperature->value_list(scratch.face_quadrature_points,
                                           scratch.T_face);

    const double thermal_voltage_rescaling_factor =
      this->adimensionalizer->get_thermal_voltage_rescaling_factor();
    for (unsigned int q = 0; q < n_face_q_points; ++q)
      scratch.U_T_face[q] = (Constants::KB * scratch.T_face[q] / Constants::Q) /
                            thermal_voltage_rescaling_factor;
  }



  template <int dim, class Permittivity>
  inline void
  NPSolver<dim, Permittivity>::copy_fe_values_on_scratch(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch,
    const unsigned int                               face,
    const unsigned int                               q)
  {
    for (const auto c : this->enabled_components)
      {
        const Displacement d = component2displacement(c);

        const FEValuesExtractors::Scalar c_extractor =
          this->get_component_extractor(c);
        const FEValuesExtractors::Vector d_extractor =
          this->get_displacement_extractor(d);

        auto &f  = scratch.d.at(c);
        auto &c_ = scratch.c.at(c);

        const unsigned int cell_dofs_on_face =
          scratch.fe_cell_support_on_face[face].size();
        for (unsigned int k = 0; k < cell_dofs_on_face; ++k)
          {
            const unsigned int component_dof_index =
              scratch.fe_cell_support_on_face[face][k];
            const unsigned int kk =
              scratch.enabled_component_indices[component_dof_index];
            f[k]  = scratch.fe_face_values_cell[d_extractor].value(kk, q);
            c_[k] = scratch.fe_face_values_cell[c_extractor].value(kk, q);
          }
      }
  }



  template <int dim, class Permittivity>
  inline void
  NPSolver<dim, Permittivity>::copy_fe_values_for_trace(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch,
    const unsigned int                               face,
    const unsigned int                               q)
  {
    for (const auto c : this->enabled_components)
      {
        const FEValuesExtractors::Scalar extractor =
          this->get_trace_component_extractor(c, true);

        auto &tr_c = scratch.tr_c.at(c);

        const unsigned int dofs_on_face =
          scratch.fe_trace_support_on_face[face].size();
        for (unsigned int k = 0; k < dofs_on_face; ++k)
          {
            tr_c[k] = scratch.fe_face_values_trace_restricted[extractor].value(
              scratch.fe_trace_support_on_face[face][k], q);
          }
      }
  }



  template <int dim, class Permittivity>
  template <typename prm>
  inline void
  NPSolver<dim, Permittivity>::assemble_ct_matrix(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch,
    const unsigned int                               face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    const unsigned int cell_dofs_on_face =
      scratch.fe_cell_support_on_face[face].size();
    const unsigned int trace_dofs_on_face =
      scratch.fe_trace_support_on_face[face].size();

    auto &q1   = scratch.d.at(Component::V);
    auto &z1   = scratch.c.at(Component::V);
    auto &tr_V = scratch.tr_c.at(Component::V);

    auto &q2   = scratch.d.at(Component::n);
    auto &z2   = scratch.c.at(Component::n);
    auto &tr_n = scratch.tr_c.at(Component::n);

    auto &q3   = scratch.d.at(Component::p);
    auto &z3   = scratch.c.at(Component::p);
    auto &tr_p = scratch.tr_c.at(Component::p);

    const double V_tau = this->parameters->tau.at(Component::V);
    const double n_tau = this->parameters->tau.at(Component::n);
    const double p_tau = this->parameters->tau.at(Component::p);

    double V_tau_stabilized = 0.;
    double n_tau_stabilized = 0.;
    double p_tau_stabilized = 0.;

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);
        copy_fe_values_for_trace(scratch, face, q);

        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        if (prm::is_V_enabled)
          V_tau_stabilized = scratch.permittivity.compute_stabilized_v_tau(
            q,
            this->adimensionalizer->template adimensionalize_tau<Component::V>(
              V_tau),
            normal);
        if (prm::is_n_enabled)
          n_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::n>(
              this->adimensionalizer
                ->template adimensionalize_tau<Component::n>(n_tau),
              normal,
              q);
        if (prm::is_p_enabled)
          p_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::p>(
              this->adimensionalizer
                ->template adimensionalize_tau<Component::p>(p_tau),
              normal,
              q);

        for (unsigned int i = 0; i < cell_dofs_on_face; ++i)
          {
            const unsigned int ii = scratch.fe_cell_support_on_face[face][i];
            for (unsigned int j = 0; j < trace_dofs_on_face; ++j)
              {
                const unsigned int jj =
                  scratch.fe_trace_support_on_face[face][j];

                // Integrals of trace functions using as test function
                // the restriction of cell test function on the border
                // i is the index of the test function
                if (prm::is_V_enabled)
                  scratch.ct_matrix(ii, jj) +=
                    (tr_V[j] * (q1[i] * normal) -
                     V_tau_stabilized * tr_V[j] * z1[i]) *
                    JxW;
                if (prm::is_n_enabled)
                  scratch.ct_matrix(ii, jj) +=
                    (tr_n[j] * (q2[i] * normal) +
                     n_tau_stabilized * (tr_n[j] * z2[i])) *
                    JxW;
                if (prm::is_p_enabled)
                  scratch.ct_matrix(ii, jj) +=
                    (tr_p[j] * (q3[i] * normal) +
                     p_tau_stabilized * (tr_p[j] * z3[i])) *
                    JxW;
              }
          }
      }
  }



  template <int dim, class Permittivity>
  template <typename prm>
  inline void
  NPSolver<dim, Permittivity>::add_ct_matrix_terms_to_cc_rhs(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch,
    const unsigned int                               face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    const FEValuesExtractors::Vector electric_field =
      this->get_displacement_extractor(Displacement::E);
    const FEValuesExtractors::Scalar electric_potential =
      this->get_component_extractor(Component::V);
    const FEValuesExtractors::Vector electron_displacement =
      this->get_displacement_extractor(Displacement::Wn);
    const FEValuesExtractors::Scalar electron_density =
      this->get_component_extractor(Component::n);
    const FEValuesExtractors::Vector hole_displacement =
      this->get_displacement_extractor(Displacement::Wp);
    const FEValuesExtractors::Scalar hole_density =
      this->get_component_extractor(Component::p);

    const auto &tr_V0 = scratch.previous_tr_c_face.at(Component::V);
    const auto &tr_n0 = scratch.previous_tr_c_face.at(Component::n);
    const auto &tr_p0 = scratch.previous_tr_c_face.at(Component::p);

    const double V_tau = this->parameters->tau.at(Component::V);
    const double n_tau = this->parameters->tau.at(Component::n);
    const double p_tau = this->parameters->tau.at(Component::p);

    double V_tau_stabilized = 0.;
    double n_tau_stabilized = 0.;
    double p_tau_stabilized = 0.;

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        if (prm::is_V_enabled)
          V_tau_stabilized = scratch.permittivity.compute_stabilized_v_tau(
            q,
            this->adimensionalizer->template adimensionalize_tau<Component::V>(
              V_tau),
            normal);
        if (prm::is_n_enabled)
          n_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::n>(
              this->adimensionalizer
                ->template adimensionalize_tau<Component::n>(n_tau),
              normal,
              q);
        if (prm::is_p_enabled)
          p_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::p>(
              this->adimensionalizer
                ->template adimensionalize_tau<Component::p>(p_tau),
              normal,
              q);

        for (unsigned int i = 0;
             i < scratch.fe_cell_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii  = scratch.fe_cell_support_on_face[face][i];
            const unsigned int iii = scratch.enabled_component_indices[ii];

            if (prm::is_V_enabled)
              {
                const dealii::Tensor<1, dim> q1 =
                  scratch.fe_face_values_cell[electric_field].value(iii, q);
                const double z1 =
                  scratch.fe_face_values_cell[electric_potential].value(iii, q);

                scratch.cc_rhs(ii) += (-tr_V0[q] * (q1 * normal) +
                                       V_tau_stabilized * tr_V0[q] * z1) *
                                      JxW;
              }

            if (prm::is_n_enabled)
              {
                const dealii::Tensor<1, dim> q2 =
                  scratch.fe_face_values_cell[electron_displacement].value(iii,
                                                                           q);
                const double z2 =
                  scratch.fe_face_values_cell[electron_density].value(iii, q);

                scratch.cc_rhs(ii) += (-tr_n0[q] * (q2 * normal) -
                                       n_tau_stabilized * tr_n0[q] * z2) *
                                      JxW;
              }

            if (prm::is_p_enabled)
              {
                const dealii::Tensor<1, dim> q3 =
                  scratch.fe_face_values_cell[hole_displacement].value(iii, q);
                const double z3 =
                  scratch.fe_face_values_cell[hole_density].value(iii, q);

                scratch.cc_rhs(ii) += (-tr_p0[q] * (q3 * normal) -
                                       p_tau_stabilized * tr_p0[q] * z3) *
                                      JxW;
              }
          }
      }
  }



  template <int dim, class Permittivity>
  template <typename prm, Component c>
  inline void
  NPSolver<dim, Permittivity>::assemble_tc_matrix(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch,
    const unsigned int                               face)
  {
    if (c != V && c != n && c != p)
      AssertThrow(false, InvalidComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    auto &c_ = scratch.c.at(c);
    auto &f  = scratch.d.at(c);
    auto &xi = scratch.tr_c.at(c);

    const auto &E = scratch.previous_d_face.at(Component::V);

    dealii::Tensor<1, dim> epsilon_times_E;

    dealii::Tensor<1, dim> mu_n_times_previous_E;
    dealii::Tensor<1, dim> mu_p_times_previous_E;

    dealii::Tensor<2, dim> n_einstein_diffusion_coefficient;
    dealii::Tensor<2, dim> p_einstein_diffusion_coefficient;

    const std::vector<double> n0 = scratch.previous_c_face.at(Component::n);
    const std::vector<double> p0 = scratch.previous_c_face.at(Component::p);

    const double tau = this->parameters->tau.at(c);

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);
        copy_fe_values_for_trace(scratch, face, q);

        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        if (c == Component::n)
          {
            mu_n_times_previous_E = scratch.mu_n_face[q] * E[q];
            n_einstein_diffusion_coefficient =
              scratch
                .template compute_einstein_diffusion_coefficient<Component::n>(
                  q);
          }
        if (c == Component::p)
          {
            mu_p_times_previous_E = scratch.mu_p_face[q] * E[q];
            p_einstein_diffusion_coefficient =
              scratch
                .template compute_einstein_diffusion_coefficient<Component::p>(
                  q);
          }

        const double rescaled_tau =
          this->adimensionalizer->template adimensionalize_tau<c>(tau);
        const double tau_stabilized =
          (c == Component::V) ?
            scratch.permittivity.compute_stabilized_v_tau(q,
                                                          rescaled_tau,
                                                          normal) :
            scratch.template compute_stabilized_tau<c>(rescaled_tau, normal, q);

        const unsigned int trace_dofs_per_face =
          scratch.fe_trace_support_on_face[face].size();
        const unsigned int cell_dofs_per_face =
          scratch.fe_cell_support_on_face[face].size();

        for (unsigned int i = 0; i < trace_dofs_per_face; ++i)
          {
            const unsigned int ii = scratch.fe_trace_support_on_face[face][i];
            for (unsigned int j = 0; j < cell_dofs_per_face; ++j)
              {
                const unsigned int jj =
                  scratch.fe_cell_support_on_face[face][j];

                // Integrals of the cell functions restricted on the
                // border. The sign is reversed to be used in the
                // Schur complement (and therefore this is the
                // opposite of the right matrix that describes the
                // problem on this cell)
                if (c == V)
                  {
                    scratch.permittivity.epsilon_operator_on_face(
                      q, f[j], epsilon_times_E);
                    scratch.tc_matrix(ii, jj) -=
                      (epsilon_times_E * normal + tau_stabilized * c_[j]) *
                      xi[i] * JxW;
                  }
                if (c == n)
                  {
                    const dealii::Tensor<1, dim> mu_n_times_E =
                      scratch.mu_n_face[q] * scratch.d[Component::V][j];

                    scratch.tc_matrix(ii, jj) -=
                      ((c_[j] * mu_n_times_previous_E + n0[q] * mu_n_times_E) *
                         normal -
                       (n_einstein_diffusion_coefficient * f[j]) * normal -
                       tau_stabilized * c_[j]) *
                      xi[i] * JxW;
                  }
                if (c == p)
                  {
                    const dealii::Tensor<1, dim> mu_p_times_E =
                      scratch.mu_p_face[q] * scratch.d[Component::V][j];

                    scratch.tc_matrix(ii, jj) -=
                      (-(c_[j] * mu_p_times_previous_E + p0[q] * mu_p_times_E) *
                         normal -
                       (p_einstein_diffusion_coefficient * f[j]) * normal -
                       tau_stabilized * c_[j]) *
                      xi[i] * JxW;
                  }
              }
          }
      }
  }



  template <int dim, class Permittivity>
  template <typename prm, Component c>
  inline void
  NPSolver<dim, Permittivity>::add_tc_matrix_terms_to_tt_rhs(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch,
    Ddhdg::NPSolver<dim, Permittivity>::PerTaskData &task_data,
    const unsigned int                               face)
  {
    if (c != V && c != n && c != p)
      AssertThrow(false, InvalidComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    const FEValuesExtractors::Scalar tr_c_extractor =
      this->get_trace_component_extractor(c, true);

    auto &c0 = scratch.previous_c_face.at(c);
    auto &f0 = scratch.previous_d_face.at(c);

    const auto &E0 = scratch.previous_d_face[Component::V];

    dealii::Tensor<1, dim> epsilon_times_E;
    double                 epsilon_times_previous_E_times_normal = 0.;
    double                 mu_n_times_previous_E_times_normal    = 0.;
    double                 mu_p_times_previous_E_times_normal    = 0.;

    dealii::Tensor<2, dim> n_einstein_diffusion_coefficient;
    dealii::Tensor<2, dim> p_einstein_diffusion_coefficient;

    const double tau = this->parameters->tau.at(c);

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        if (c == Component::V)
          {
            scratch.permittivity.epsilon_operator_on_face(q,
                                                          E0[q],
                                                          epsilon_times_E);
            epsilon_times_previous_E_times_normal = epsilon_times_E * normal;
          }
        if (c == Component::n)
          {
            mu_n_times_previous_E_times_normal =
              scratch.mu_n_face[q] * E0[q] * normal;
            n_einstein_diffusion_coefficient =
              scratch
                .template compute_einstein_diffusion_coefficient<Component::n>(
                  q);
          }
        if (c == Component::p)
          {
            mu_p_times_previous_E_times_normal =
              scratch.mu_p_face[q] * E0[q] * normal;
            p_einstein_diffusion_coefficient =
              scratch
                .template compute_einstein_diffusion_coefficient<Component::p>(
                  q);
          }

        const double tau_rescaled =
          this->adimensionalizer->template adimensionalize_tau<c>(tau);
        const double tau_stabilized =
          (c == Component::V) ?
            scratch.permittivity.compute_stabilized_v_tau(q,
                                                          tau_rescaled,
                                                          normal) :
            scratch.template compute_stabilized_tau<c>(tau_rescaled, normal, q);

        for (unsigned int i = 0;
             i < scratch.fe_trace_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii = scratch.fe_trace_support_on_face[face][i];
            const double       xi =
              scratch.fe_face_values_trace_restricted[tr_c_extractor].value(ii,
                                                                            q);
            if (c == V)
              {
                task_data.tt_vector[ii] +=
                  (-epsilon_times_previous_E_times_normal -
                   tau_stabilized * c0[q]) *
                  xi * JxW;
              }
            if (c == n)
              {
                task_data.tt_vector[ii] +=
                  (-c0[q] * mu_n_times_previous_E_times_normal +
                   n_einstein_diffusion_coefficient * f0[q] * normal +
                   tau_stabilized * c0[q]) *
                  xi * JxW;
              }
            if (c == p)
              {
                task_data.tt_vector[ii] +=
                  (c0[q] * mu_p_times_previous_E_times_normal +
                   p_einstein_diffusion_coefficient * f0[q] * normal +
                   tau_stabilized * c0[q]) *
                  xi * JxW;
              }
          }
      }
  }



  template <int dim, class Permittivity>
  template <typename prm, Component c>
  inline void
  NPSolver<dim, Permittivity>::assemble_tt_matrix(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch,
    Ddhdg::NPSolver<dim, Permittivity>::PerTaskData &task_data,
    const unsigned int                               face)
  {
    if (c != V && c != n && c != p)
      AssertThrow(false, InvalidComponent());

    auto &tr_c = scratch.tr_c.at(c);
    auto &xi   = scratch.tr_c.at(c);

    const double tau  = this->parameters->tau.at(c);
    const double sign = (c == V) ? -1 : 1;

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();
    const unsigned int trace_dofs_per_face =
      scratch.fe_trace_support_on_face[face].size();

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);
        copy_fe_values_for_trace(scratch, face, q);

        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        const double tau_rescaled =
          this->adimensionalizer->template adimensionalize_tau<c>(tau);
        const double tau_stabilized =
          (c == Component::V) ?
            scratch.permittivity.compute_stabilized_v_tau(q,
                                                          tau_rescaled,
                                                          normal) :
            scratch.template compute_stabilized_tau<c>(tau_rescaled, normal, q);

        // Integrals of trace functions (both test and trial)
        for (unsigned int i = 0; i < trace_dofs_per_face; ++i)
          {
            const unsigned int ii = scratch.fe_trace_support_on_face[face][i];
            for (unsigned int j = 0; j < trace_dofs_per_face; ++j)
              {
                const unsigned int jj =
                  scratch.fe_trace_support_on_face[face][j];
                task_data.tt_matrix(ii, jj) +=
                  sign * tau_stabilized * tr_c[j] * xi[i] * JxW;
              }
          }
      }
  }



  template <int dim, class Permittivity>
  template <typename prm, Component c>
  inline void
  NPSolver<dim, Permittivity>::add_tt_matrix_terms_to_tt_rhs(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch,
    Ddhdg::NPSolver<dim, Permittivity>::PerTaskData &task_data,
    const unsigned int                               face)
  {
    if (c != V && c != n && c != p)
      AssertThrow(false, InvalidComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();
    const unsigned int trace_dofs_per_face =
      scratch.fe_trace_support_on_face[face].size();

    const double tau  = this->parameters->tau.at(c);
    const double sign = (c == Component::V) ? -1 : 1;

    const FEValuesExtractors::Scalar tr_c_extractor =
      this->get_trace_component_extractor(c, true);

    const auto &tr_c0 = scratch.previous_tr_c_face.at(c);

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        const double tau_rescaled =
          this->adimensionalizer->template adimensionalize_tau<c>(tau);
        const double tau_stabilized =
          (c == Component::V) ?
            scratch.permittivity.compute_stabilized_v_tau(q,
                                                          tau_rescaled,
                                                          normal) :
            scratch.template compute_stabilized_tau<c>(tau_rescaled, normal, q);

        for (unsigned int i = 0; i < trace_dofs_per_face; ++i)
          {
            const unsigned int ii = scratch.fe_trace_support_on_face[face][i];
            const double       xi =
              scratch.fe_face_values_trace_restricted[tr_c_extractor].value(ii,
                                                                            q);
            task_data.tt_vector[ii] +=
              -sign * tau_stabilized * tr_c0[q] * xi * JxW;
          }
      }
  }



  template <int dim, class Permittivity>
  template <typename prm, Ddhdg::Component c>
  inline void
  NPSolver<dim, Permittivity>::apply_dbc_on_face(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch,
    Ddhdg::NPSolver<dim, Permittivity>::PerTaskData &task_data,
    const Ddhdg::DirichletBoundaryCondition<dim> &   dbc,
    unsigned int                                     face)
  {
    if (c != V && c != n && c != p)
      Assert(false, InvalidComponent());

    auto &tr_c  = scratch.tr_c.at(c);
    auto &tr_c0 = scratch.previous_tr_c_face.at(c);

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();
    const unsigned int trace_dofs_per_face =
      scratch.fe_trace_support_on_face[face].size();

    const double rescaling_factor =
      this->adimensionalizer->template get_component_rescaling_factor<c>();

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_for_trace(scratch, face, q);

        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);

        const Point<dim> quadrature_point =
          scratch.fe_face_values_trace_restricted.quadrature_point(q);
        const double dbc_value =
          (prm::thermodyn_eq) ?
            0 :
            dbc.evaluate(quadrature_point) / rescaling_factor - tr_c0[q];

        for (unsigned int i = 0; i < trace_dofs_per_face; ++i)
          {
            const unsigned int ii = scratch.fe_trace_support_on_face[face][i];
            for (unsigned int j = 0; j < trace_dofs_per_face; ++j)
              {
                const unsigned int jj =
                  scratch.fe_trace_support_on_face[face][j];
                task_data.tt_matrix(ii, jj) += tr_c[i] * tr_c[j] * JxW;
              }

            task_data.tt_vector[ii] += tr_c[i] * dbc_value * JxW;
          }
      }
  }



  template <int dim, class Permittivity>
  template <Ddhdg::Component c>
  void
  NPSolver<dim, Permittivity>::apply_nbc_on_face(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch,
    Ddhdg::NPSolver<dim, Permittivity>::PerTaskData &task_data,
    const Ddhdg::NeumannBoundaryCondition<dim> &     nbc,
    unsigned int                                     face)
  {
    if (c != Component::V && c != Component::n && c != Component::p)
      Assert(false, InvalidComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();
    const unsigned int trace_dofs_per_face =
      scratch.fe_trace_support_on_face[face].size();

    auto &tr_c = scratch.tr_c.at(c);

    const double rescaling_factor =
      this->adimensionalizer
        ->template get_neumann_boundary_condition_rescaling_factor<c>();

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_for_trace(scratch, face, q);

        const double     JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Point<dim> quadrature_point =
          scratch.fe_face_values_trace_restricted.quadrature_point(q);
        const double nbc_value =
          nbc.evaluate(quadrature_point) / rescaling_factor;

        for (unsigned int i = 0; i < trace_dofs_per_face; ++i)
          {
            const unsigned int ii = scratch.fe_trace_support_on_face[face][i];
            task_data.tt_vector(ii) += tr_c[i] * nbc_value * JxW;
          }
      }
  }


  template <int dim, class Permittivity>
  template <typename prm>
  inline void
  NPSolver<dim, Permittivity>::add_border_products_to_cc_matrix(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch,
    const unsigned int                               face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();
    const unsigned int cell_dofs_per_face =
      scratch.fe_cell_support_on_face[face].size();

    auto &E  = scratch.d.at(Component::V);
    auto &V  = scratch.c.at(Component::V);
    auto &Wn = scratch.d.at(Component::n);
    auto &n  = scratch.c.at(Component::n);
    auto &Wp = scratch.d.at(Component::p);
    auto &p  = scratch.c.at(Component::p);

    auto &z1 = scratch.c.at(Component::V);
    auto &z2 = scratch.c.at(Component::n);
    auto &z3 = scratch.c.at(Component::p);

    const auto  n0 = scratch.previous_c_face.at(Component::n);
    const auto  p0 = scratch.previous_c_face.at(Component::p);
    const auto &E0 = scratch.previous_d_face.at(Component::V);

    const double V_tau = this->parameters->tau.at(Component::V);
    const double n_tau = this->parameters->tau.at(Component::n);
    const double p_tau = this->parameters->tau.at(Component::p);

    double V_tau_stabilized = 0;
    double n_tau_stabilized = 0;
    double p_tau_stabilized = 0;

    dealii::Tensor<1, dim> epsilon_times_E;

    dealii::Tensor<1, dim> mu_n_times_previous_E;
    dealii::Tensor<1, dim> mu_p_times_previous_E;

    dealii::Tensor<2, dim> n_einstein_diffusion_coefficient;
    dealii::Tensor<2, dim> p_einstein_diffusion_coefficient;

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);

        // Integrals on the border of the cell of the cell functions
        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        if (prm::is_n_enabled)
          mu_n_times_previous_E = scratch.mu_n_face[q] * E0[q];

        if (prm::is_p_enabled)
          mu_p_times_previous_E = scratch.mu_p_face[q] * E0[q];

        if (prm::is_n_enabled)
          n_einstein_diffusion_coefficient =
            scratch
              .template compute_einstein_diffusion_coefficient<Component::n>(q);

        if (prm::is_p_enabled)
          p_einstein_diffusion_coefficient =
            scratch
              .template compute_einstein_diffusion_coefficient<Component::p>(q);

        if (prm::is_V_enabled)
          V_tau_stabilized = scratch.permittivity.compute_stabilized_v_tau(
            q,
            this->adimensionalizer->template adimensionalize_tau<Component::V>(
              V_tau),
            normal);
        if (prm::is_n_enabled)
          n_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::n>(
              this->adimensionalizer
                ->template adimensionalize_tau<Component::n>(n_tau),
              normal,
              q);
        if (prm::is_p_enabled)
          p_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::p>(
              this->adimensionalizer
                ->template adimensionalize_tau<Component::p>(p_tau),
              normal,
              q);

        for (unsigned int i = 0; i < cell_dofs_per_face; ++i)
          {
            const unsigned int ii = scratch.fe_cell_support_on_face[face][i];
            for (unsigned int j = 0; j < cell_dofs_per_face; ++j)
              {
                const unsigned int jj =
                  scratch.fe_cell_support_on_face[face][j];

                if (prm::is_V_enabled)
                  {
                    scratch.permittivity.epsilon_operator_on_face(
                      q, E[j], epsilon_times_E);
                    scratch.cc_matrix(ii, jj) +=
                      (epsilon_times_E * normal * z1[i] +
                       V_tau_stabilized * V[j] * z1[i]) *
                      JxW;
                  }
                if (prm::is_n_enabled)
                  scratch.cc_matrix(ii, jj) +=
                    ((n[j] * mu_n_times_previous_E +
                      scratch.mu_n_face[q] * E[j] * n0[q]) *
                       normal * z2[i] -
                     n_einstein_diffusion_coefficient * Wn[j] * normal * z2[i] -
                     n_tau_stabilized * n[j] * z2[i]) *
                    JxW;
                if (prm::is_p_enabled)
                  scratch.cc_matrix(ii, jj) +=
                    (-(p[j] * mu_p_times_previous_E +
                       scratch.mu_p_face[q] * E[j] * p0[q]) *
                       normal * z3[i] -
                     p_einstein_diffusion_coefficient * Wp[j] * normal * z3[i] -
                     p_tau_stabilized * p[j] * z3[i]) *
                    JxW;
              }
          }
      }
  }



  template <int dim, class Permittivity>
  template <typename prm>
  inline void
  NPSolver<dim, Permittivity>::add_border_products_to_cc_rhs(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch,
    const unsigned int                               face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    const FEValuesExtractors::Scalar electric_potential =
      this->get_component_extractor(Component::V);
    const FEValuesExtractors::Scalar electron_density =
      this->get_component_extractor(Component::n);
    const FEValuesExtractors::Scalar hole_density =
      this->get_component_extractor(Component::p);

    auto &V0  = scratch.previous_c_face.at(Component::V);
    auto &E0  = scratch.previous_d_face.at(Component::V);
    auto &n0  = scratch.previous_c_face.at(Component::n);
    auto &Wn0 = scratch.previous_d_face.at(Component::n);
    auto &p0  = scratch.previous_c_face.at(Component::p);
    auto &Wp0 = scratch.previous_d_face.at(Component::p);

    const double V_tau = this->parameters->tau.at(Component::V);
    const double n_tau = this->parameters->tau.at(Component::n);
    const double p_tau = this->parameters->tau.at(Component::p);

    double V_tau_stabilized = 0.;
    double n_tau_stabilized = 0.;
    double p_tau_stabilized = 0.;

    dealii::Tensor<1, dim> epsilon_times_E;
    double                 epsilon_times_E0_times_normal = 0.;

    dealii::Tensor<2, dim> n_einstein_diffusion_coefficient;
    dealii::Tensor<2, dim> p_einstein_diffusion_coefficient;

    double Jn_flux = 0.;
    double Jp_flux = 0.;

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        if (prm::is_V_enabled)
          {
            scratch.permittivity.epsilon_operator_on_face(q,
                                                          E0[q],
                                                          epsilon_times_E);
            epsilon_times_E0_times_normal = epsilon_times_E * normal;
          }

        if (prm::is_n_enabled)
          n_einstein_diffusion_coefficient =
            scratch
              .template compute_einstein_diffusion_coefficient<Component::n>(q);

        if (prm::is_p_enabled)
          p_einstein_diffusion_coefficient =
            scratch
              .template compute_einstein_diffusion_coefficient<Component::p>(q);

        if (prm::is_n_enabled)
          Jn_flux = (n0[q] * (scratch.mu_n_face[q] * E0[q]) -
                     (n_einstein_diffusion_coefficient * Wn0[q])) *
                    normal;

        if (prm::is_p_enabled)
          Jp_flux = (-p0[q] * (scratch.mu_p_face[q] * E0[q]) -
                     (p_einstein_diffusion_coefficient * Wp0[q])) *
                    normal;

        if (prm::is_V_enabled)
          V_tau_stabilized = scratch.permittivity.compute_stabilized_v_tau(
            q,
            this->adimensionalizer->template adimensionalize_tau<Component::V>(
              V_tau),
            normal);
        if (prm::is_n_enabled)
          n_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::n>(
              this->adimensionalizer
                ->template adimensionalize_tau<Component::n>(n_tau),
              normal,
              q);
        if (prm::is_p_enabled)
          p_tau_stabilized =
            scratch.template compute_stabilized_tau<Component::p>(
              this->adimensionalizer
                ->template adimensionalize_tau<Component::p>(p_tau),
              normal,
              q);

        for (unsigned int i = 0;
             i < scratch.fe_cell_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii  = scratch.fe_cell_support_on_face[face][i];
            const unsigned int iii = scratch.enabled_component_indices[ii];

            if (prm::is_V_enabled)
              {
                const double z1 =
                  scratch.fe_face_values_cell[electric_potential].value(iii, q);
                scratch.cc_rhs[ii] += (-epsilon_times_E0_times_normal * z1 -
                                       V_tau_stabilized * V0[q] * z1) *
                                      JxW;
              }
            if (prm::is_n_enabled)
              {
                const double z2 =
                  scratch.fe_face_values_cell[electron_density].value(iii, q);
                scratch.cc_rhs[ii] +=
                  (-Jn_flux * z2 + n_tau_stabilized * n0[q] * z2) * JxW;
              }
            if (prm::is_p_enabled)
              {
                const double z3 =
                  scratch.fe_face_values_cell[hole_density].value(iii, q);
                scratch.cc_rhs[ii] +=
                  (-Jp_flux * z3 + p_tau_stabilized * p0[q] * z3) * JxW;
              }
          }
      }
  }



  template <int dim, class Permittivity>
  template <typename prm>
  inline void
  NPSolver<dim, Permittivity>::add_trace_terms_to_cc_rhs(
    Ddhdg::NPSolver<dim, Permittivity>::ScratchData &scratch,
    const unsigned int                               face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    double tau_stabilized;

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);

        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        for (const auto c : this->enabled_components)
          {
            auto &f                    = scratch.d.at(c);
            auto &c_                   = scratch.c.at(c);
            auto &tr_c_solution_values = scratch.tr_c_solution_values.at(c);

            const double tau  = this->parameters->tau.at(c);
            const double sign = (c == Component::V) ? 1 : -1;

            switch (c)
              {
                case Component::V:
                  tau_stabilized =
                    scratch.permittivity.compute_stabilized_v_tau(
                      q,
                      this->adimensionalizer
                        ->template adimensionalize_tau<Component::V>(tau),
                      normal);
                  break;
                case Component::n:
                  tau_stabilized =
                    scratch.template compute_stabilized_tau<Component::n>(
                      this->adimensionalizer
                        ->template adimensionalize_tau<Component::n>(tau),
                      normal,
                      q);
                  break;
                case Component::p:
                  tau_stabilized =
                    scratch.template compute_stabilized_tau<Component::p>(
                      this->adimensionalizer
                        ->template adimensionalize_tau<Component::p>(tau),
                      normal,
                      q);
                  break;
                default:
                  Assert(false, InvalidComponent());
                  tau_stabilized = 1.;
              }

            for (unsigned int i = 0;
                 i < scratch.fe_cell_support_on_face[face].size();
                 ++i)
              {
                const unsigned int ii =
                  scratch.fe_cell_support_on_face[face][i];
                scratch.cc_rhs(ii) +=
                  (-f[i] * normal + sign * tau_stabilized * c_[i]) *
                  tr_c_solution_values[q] * JxW;
              }
          }
      }
  }



  template <int dim, class Permittivity>
  template <typename prm, Component c>
  void
  NPSolver<dim, Permittivity>::assemble_flux_conditions(
    ScratchData &            scratch,
    PerTaskData &            task_data,
    const bool               has_dirichlet_conditions,
    const bool               has_neumann_conditions,
    const types::boundary_id face_boundary_id,
    const unsigned int       face)
  {
    if (has_dirichlet_conditions)
      {
        const DirichletBoundaryCondition<dim> dbc =
          (prm::thermodyn_eq) ?
            DirichletBoundaryCondition<dim>(
              std::make_shared<dealii::Functions::ZeroFunction<dim>>(),
              Component::V) :
            this->problem->boundary_handler->get_dirichlet_conditions_for_id(
              face_boundary_id, c);
        this->apply_dbc_on_face<prm, c>(scratch, task_data, dbc, face);
      }
    else
      {
        this->assemble_tc_matrix<prm, c>(scratch, face);
        this->add_tc_matrix_terms_to_tt_rhs<prm, c>(scratch, task_data, face);
        this->assemble_tt_matrix<prm, c>(scratch, task_data, face);
        this->add_tt_matrix_terms_to_tt_rhs<prm, c>(scratch, task_data, face);
        if (has_neumann_conditions)
          {
            const NeumannBoundaryCondition<dim> nbc =
              (prm::thermodyn_eq) ?
                NeumannBoundaryCondition<dim>(
                  std::make_shared<dealii::Functions::ZeroFunction<dim>>(), c) :
                this->problem->boundary_handler->get_neumann_conditions_for_id(
                  face_boundary_id, c);
            this->apply_nbc_on_face<c>(scratch, task_data, nbc, face);
          }
      }
  }



  template <int dim, class Permittivity>
  template <typename prm>
  void
  NPSolver<dim, Permittivity>::assemble_flux_conditions_wrapper(
    const Component                         c,
    ScratchData &                           scratch,
    PerTaskData &                           task_data,
    const std::map<Ddhdg::Component, bool> &has_dirichlet_conditions,
    const std::map<Ddhdg::Component, bool> &has_neumann_conditions,
    const types::boundary_id                face_boundary_id,
    const unsigned int                      face)
  {
    switch (c)
      {
        case Component::V:
          assemble_flux_conditions<prm, Component::V>(
            scratch,
            task_data,
            has_dirichlet_conditions.at(Component::V),
            has_neumann_conditions.at(Component::V),
            face_boundary_id,
            face);
          break;
        case Component::n:
          assemble_flux_conditions<prm, Component::n>(
            scratch,
            task_data,
            has_dirichlet_conditions.at(Component::n),
            has_neumann_conditions.at(Component::n),
            face_boundary_id,
            face);
          break;
        case Component::p:
          assemble_flux_conditions<prm, Component::p>(
            scratch,
            task_data,
            has_dirichlet_conditions.at(Component::p),
            has_neumann_conditions.at(Component::p),
            face_boundary_id,
            face);
          break;
        default:
          Assert(false, InvalidComponent());
          break;
      }
  }



  template <int dim, class Permittivity>
  template <typename prm>
  void
  NPSolver<dim, Permittivity>::assemble_system_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData &                                         scratch,
    PerTaskData &                                         task_data)
  {
    typename DoFHandler<dim>::active_cell_iterator loc_cell(&(*triangulation),
                                                            cell->level(),
                                                            cell->index(),
                                                            &dof_handler_cell);
    typename DoFHandler<dim>::active_cell_iterator trace_cell(
      &(*triangulation), cell->level(), cell->index(), &dof_handler_trace);

    // Reset every value that could be related to the previous cell
    scratch.cc_matrix = 0;
    scratch.cc_rhs    = 0;
    if (!task_data.trace_reconstruct)
      {
        scratch.ct_matrix   = 0;
        scratch.tc_matrix   = 0;
        task_data.tt_matrix = 0;
        task_data.tt_vector = 0;
      }
    scratch.fe_values_cell.reinit(loc_cell);

    // We use the following function to copy every value that we need from the
    // fe_values objects into the scratch. This also compute the physical
    // parameters (like permittivity or the recombination term) on the
    // quadrature points of the cell
    this->prepare_data_on_cell_quadrature_points(scratch);

    // Integrals on the overall cell
    // This function computes every L2 product that we need to compute in
    // order to assemble the matrix for the current cell.
    this->add_cell_products_to_cc_matrix<prm>(scratch);

    // This function, instead, computes the l2 products that are needed for
    // the right hand term
    this->add_cell_products_to_cc_rhs<prm>(scratch);

    // Now we must perform the L2 product on the boundary, i.e. for each face
    // of the cell
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
         ++face)
      {
        scratch.fe_face_values_cell.reinit(loc_cell, face);
        scratch.fe_face_values_trace.reinit(trace_cell, face);
        scratch.fe_face_values_trace_restricted.reinit(cell, face);

        // If we have already solved the system for the trace, copy the values
        // of the global solution in the scratch that stores the values for
        // this cell
        if (task_data.trace_reconstruct)
          {
            for (const auto c : Ddhdg::all_components())
              {
                const FEValuesExtractors::Scalar extractor =
                  this->get_trace_component_extractor(c);
                scratch.fe_face_values_trace[extractor].get_function_values(
                  this->update_trace, scratch.tr_c_solution_values[c]);
              }
          }

        // Now I create some maps to store, for each component, if there is a
        // boundary condition for it
        const types::boundary_id face_boundary_id =
          cell->face(face)->boundary_id();
        std::map<Ddhdg::Component, bool> has_dirichlet_conditions;
        std::map<Ddhdg::Component, bool> has_neumann_conditions;

        // Now we populate the previous maps

        for (const auto c : this->enabled_components)
          {
            has_dirichlet_conditions.insert(
              {c,
               this->problem->boundary_handler
                 ->has_dirichlet_boundary_conditions(face_boundary_id, c)});
            has_neumann_conditions.insert(
              {c,
               this->problem->boundary_handler->has_neumann_boundary_conditions(
                 face_boundary_id, c)});
          }


        // Before assembling the other parts of the matrix, we need the values
        // of epsilon, mu and D_n on the quadrature points of the current
        // face. Moreover, we want to populate the scratch object with the
        // values of the solution at the previous step. All this jobs are
        // accomplished by the next function
        this->prepare_data_on_face_quadrature_points(scratch);

        // The following function adds some terms to cc_rhs. Usually, all the
        // functions are coupled (like assemble_matrix_XXX and
        // add_matrix_XXX_terms_to_l_rhs) because the first function is the
        // function that assembles the matrix and the second one is the
        // function that adds the corresponding terms to cc_rhs (i.e. the
        // product of the matrix with the previous solution, so that the
        // solution of the system is the update from the previous solution to
        // the new one). The following function is the only exception: indeed,
        // the terms that it generates are always needed while the terms that
        // its corresponding function generates are useful only if we are not
        // reconstructing the solution from the trace
        this->add_ct_matrix_terms_to_cc_rhs<prm>(scratch, face);

        // Assembly the other matrices (the cc_matrix has been assembled
        // calling the add_cell_products_to_cc_matrix method)
        if (!task_data.trace_reconstruct)
          {
            this->assemble_ct_matrix<prm>(scratch, face);
            for (const Component c : this->enabled_components)
              {
                this->assemble_flux_conditions_wrapper<prm>(
                  c,
                  scratch,
                  task_data,
                  has_dirichlet_conditions,
                  has_neumann_conditions,
                  face_boundary_id,
                  face);
              }
          }

        // These are the last terms of the ll matrix, the ones that are
        // generated by L2 products only on the boundary of the cell
        this->add_border_products_to_cc_matrix<prm>(scratch, face);
        this->add_border_products_to_cc_rhs<prm>(scratch, face);

        if (task_data.trace_reconstruct)
          this->add_trace_terms_to_cc_rhs<prm>(scratch, face);
      }

    inversion_mutex.lock();
    scratch.cc_matrix.gauss_jordan();
    inversion_mutex.unlock();

    if (!task_data.trace_reconstruct)
      {
        scratch.tc_matrix.mmult(scratch.tmp_matrix, scratch.cc_matrix);
        scratch.tmp_matrix.vmult_add(task_data.tt_vector, scratch.cc_rhs);
        scratch.tmp_matrix.mmult(task_data.tt_matrix, scratch.ct_matrix, true);
        cell->get_dof_indices(task_data.dof_indices);
      }
    else
      {
        scratch.cc_matrix.vmult(scratch.restricted_tmp_rhs, scratch.cc_rhs);
        scratch.tmp_rhs = 0;
        for (unsigned int i = 0; i < scratch.enabled_component_indices.size();
             i++)
          scratch.tmp_rhs[scratch.enabled_component_indices[i]] =
            scratch.restricted_tmp_rhs[i];
        loc_cell->set_dof_values(scratch.tmp_rhs, update_cell);
      }
  }

} // namespace Ddhdg