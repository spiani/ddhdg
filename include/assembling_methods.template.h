#include <deal.II/base/work_stream.h>

#include "constants.h"
#include "templatized_parameters.h"

namespace Ddhdg
{
  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::assemble_system(
    const bool trace_reconstruct,
    const bool compute_thermodynamic_equilibrium,
    const bool multithreading)
  {
    Assert(this->initialized, dealii::ExcNotInitialized());

    // The number of the Dirichlet constraints will increase as soon as we found
    // them on the cells; for this reason, right now must be 0
    if (!trace_reconstruct)
      this->n_dirichlet_constraints = 0;

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

    // Create a map that associate to each active component on the trace its fe
    std::map<Component, const dealii::FiniteElement<dim> &> component_to_fe;
    for (const auto c : this->enabled_components)
      {
        const dealii::ComponentMask c_mask =
          this->get_trace_component_mask(c, true);
        const dealii::FiniteElement<dim> &fe =
          this->fe_trace_restricted->get_sub_fe(c_mask);
        component_to_fe.insert({c, fe});
      }

    PerTaskData task_data(this->fe_trace_restricted->dofs_per_cell,
                          trace_reconstruct);
    ScratchData scratch(
      *(this->fe_trace_restricted),
      *(this->fe_trace),
      *(this->fe_cell),
      quadrature_formula,
      face_quadrature_formula,
      flags_cell,
      face_flags_cell,
      flags_trace,
      flags_trace_restricted,
      this->problem->permittivity->get_computer(*(this->adimensionalizer)),
      this->problem->n_mobility->get_computer(*(this->adimensionalizer)),
      this->problem->p_mobility->get_computer(*(this->adimensionalizer)),
      this->parameters->get_tau_computer(*(this->adimensionalizer)),
      this->enabled_components,
      component_to_fe);

    if (multithreading)
      {
        WorkStream::run(dof_handler_trace_restricted.begin_active(),
                        dof_handler_trace_restricted.end(),
                        *this,
                        this->get_assemble_system_one_cell_function(
                          compute_thermodynamic_equilibrium),
                        &NPSolver<dim, ProblemType>::copy_local_to_global,
                        scratch,
                        task_data);
      }
    else
      {
        for (const auto &cell :
             this->dof_handler_trace_restricted.active_cell_iterators())
          {
            (this->*get_assemble_system_one_cell_function(
                      compute_thermodynamic_equilibrium))(cell,
                                                          scratch,
                                                          task_data);
            copy_local_to_global(task_data);
          }
      }
  }



  template <int dim, typename ProblemType>
  typename NPSolver<dim, ProblemType>::assemble_system_one_cell_pointer
  NPSolver<dim, ProblemType>::get_assemble_system_one_cell_function(
    const bool compute_thermodynamic_equilibrium)
  {
    // The last three bits of parameter_mask are the values of the flags
    // this->is_enabled(Component::V), this->is_enabled(Component::n) and
    // this->is_enabled(Component::p); as usual 1 is for TRUE, 0 is FALSE.
    unsigned int parameter_mask =
      TemplatizedParametersInterface<dim, ProblemType>::get_parameter_mask(
        this->parameters->phi_linearize,
        compute_thermodynamic_equilibrium,
        this->is_enabled(Component::V),
        this->is_enabled(Component::n),
        this->is_enabled(Component::p));

    TemplatizedParametersInterface<dim, ProblemType> *p1;
    TemplatizedParametersInterface<dim, ProblemType> *p2;

    p1 = new TemplatizedParameters<
      dim,
      ProblemType,
      TemplatizedParametersInterface<dim, ProblemType>::mask_max_value>();
    while (p1->get_parameter_mask() != parameter_mask)
      {
        p2 = p1->get_previous();
        delete p1;
        p1 = p2;
      }

    typename NPSolver<dim, ProblemType>::assemble_system_one_cell_pointer f =
      p1->get_assemble_system_one_cell_function(
        this->parameters->get_tau_computer(*(this->adimensionalizer)).get());
    delete p1;
    return f;
  }



  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::prepare_data_on_cell_quadrature_points(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &scratch)
  {
    const unsigned int n_q_points =
      scratch.fe_values_cell.get_quadrature().size();

    // Copy the values of the previous solution regarding the previous cell in
    // the scratch. This must be done for every component because, for
    // example, the equation in n requires the data from p and so on
    for (const auto c : Ddhdg::all_primary_components())
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
    scratch.permittivity.initialize_on_cell(scratch.cell_quadrature_points);

    // Compute the value of mu
    if (this->is_enabled(Component::n))
      scratch.n_mobility.initialize_on_cell(scratch.cell_quadrature_points);
    if (this->is_enabled(Component::p))
      scratch.p_mobility.initialize_on_cell(scratch.cell_quadrature_points);

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

        this->recombination_term->compute_multiple_recombination_terms(
          scratch.previous_c_cell.at(Component::n),
          scratch.previous_c_cell.at(Component::p),
          scratch.cell_quadrature_points,
          this->adimensionalizer->get_component_rescaling_factor(Component::n),
          true,
          scratch.r_cell);
        this->recombination_term
          ->compute_multiple_derivatives_of_recombination_terms(
            scratch.previous_c_cell.at(Component::n),
            scratch.previous_c_cell.at(Component::p),
            scratch.cell_quadrature_points,
            this->adimensionalizer->get_component_rescaling_factor(
              Component::n),
            Component::n,
            true,
            dr_n);
        this->recombination_term
          ->compute_multiple_derivatives_of_recombination_terms(
            scratch.previous_c_cell.at(Component::n),
            scratch.previous_c_cell.at(Component::p),
            scratch.cell_quadrature_points,
            this->adimensionalizer->get_component_rescaling_factor(
              Component::n),
            Component::p,
            true,
            dr_p);

        this->adimensionalizer->adimensionalize_recombination_term(
          scratch.r_cell, dr_n, dr_p);
      }

    if (this->parameters->phi_linearize)
      {
        const auto & V_values = scratch.previous_c_cell.at(Component::V);
        const double V_rescaling_factor =
          this->adimensionalizer
            ->template get_component_rescaling_factor<Component::V>();

        if (this->is_enabled(Component::n))
          {
            auto &      phi_n    = scratch.phi.at(Component::n);
            const auto &n_values = scratch.previous_c_cell.at(Component::n);
            double      n_rescaling_factor =
              this->adimensionalizer
                ->template get_component_rescaling_factor<Component::n>();
            for (unsigned int q = 0; q < n_q_points; ++q)
              phi_n[q] =
                this->template compute_quasi_fermi_potential<Component::n>(
                  n_values[q] * n_rescaling_factor,
                  V_values[q] * V_rescaling_factor,
                  scratch.T_cell[q]);
          }
        if (this->is_enabled(Component::p))
          {
            auto &      phi_p    = scratch.phi.at(Component::p);
            const auto &p_values = scratch.previous_c_cell.at(Component::p);
            double      p_rescaling_factor =
              this->adimensionalizer
                ->template get_component_rescaling_factor<Component::p>();
            for (unsigned int q = 0; q < n_q_points; ++q)
              phi_p[q] =
                this->template compute_quasi_fermi_potential<Component::p>(
                  p_values[q] * p_rescaling_factor,
                  V_values[q] * V_rescaling_factor,
                  scratch.T_cell[q]);
          }
      }
  }



  template <int dim, typename ProblemType>
  template <typename prm>
  void
  NPSolver<dim, ProblemType>::add_cell_products_to_cc_matrix(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &scratch)
  {
    constexpr bool something_enabled =
      prm::is_V_enabled || prm::is_n_enabled || prm::is_p_enabled;

    // I had to use this if to prevent a lot of "unused variables" warnings
    if constexpr (something_enabled)
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

        // Prevent warnings about unused variables
        if constexpr (!prm::is_V_enabled)
          {
            (void)electric_potential;
            (void)electric_field;
          }
        if constexpr (!prm::is_n_enabled)
          {
            (void)electron_density;
            (void)electron_displacement;
          }
        if constexpr (!prm::is_p_enabled)
          {
            (void)hole_density;
            (void)hole_displacement;
          }

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

        const auto &phi_n_vector = scratch.phi.at(Component::n);
        const auto &phi_p_vector = scratch.phi.at(Component::p);

        if constexpr (!prm::phi_linearize)
          {
            (void)phi_n_vector;
            (void)phi_p_vector;
          }

        const unsigned int dofs_per_component =
          scratch.enabled_component_indices.size();

        dealii::Tensor<1, dim> epsilon_times_E;

        dealii::Tensor<1, dim> mu_n_times_previous_E;
        dealii::Tensor<1, dim> mu_p_times_previous_E;

        dealii::Tensor<1, dim> mu_n_times_E;
        dealii::Tensor<1, dim> mu_p_times_E;

        dealii::Tensor<1, dim> n_einstein_diffusion_coefficient_times_Wn;
        dealii::Tensor<1, dim> p_einstein_diffusion_coefficient_times_Wp;

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

        double phi_V_der = 0.;

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const double JxW = scratch.fe_values_cell.JxW(q);
            for (unsigned int k = 0; k < loc_dofs_per_cell; ++k)
              {
                if constexpr (prm::is_V_enabled)
                  {
                    q1[k] = scratch.fe_values_cell[electric_field].value(k, q);
                    q1_div[k] =
                      scratch.fe_values_cell[electric_field].divergence(k, q);
                    z1[k] =
                      scratch.fe_values_cell[electric_potential].value(k, q);
                    z1_grad[k] =
                      scratch.fe_values_cell[electric_potential].gradient(k, q);
                  }

                if constexpr (prm::is_n_enabled)
                  {
                    q2[k] =
                      scratch.fe_values_cell[electron_displacement].value(k, q);
                    q2_div[k] =
                      scratch.fe_values_cell[electron_displacement].divergence(
                        k, q);
                    z2[k] =
                      scratch.fe_values_cell[electron_density].value(k, q);
                    z2_grad[k] =
                      scratch.fe_values_cell[electron_density].gradient(k, q);
                  }

                if constexpr (prm::is_p_enabled)
                  {
                    q3[k] =
                      scratch.fe_values_cell[hole_displacement].value(k, q);
                    q3_div[k] =
                      scratch.fe_values_cell[hole_displacement].divergence(k,
                                                                           q);
                    z3[k] = scratch.fe_values_cell[hole_density].value(k, q);
                    z3_grad[k] =
                      scratch.fe_values_cell[hole_density].gradient(k, q);
                  }
              }

            if constexpr (prm::is_n_enabled)
              scratch.n_mobility.mu_operator_on_cell(q,
                                                     E0[q],
                                                     mu_n_times_previous_E);

            if constexpr (prm::is_p_enabled)
              scratch.p_mobility.mu_operator_on_cell(q,
                                                     E0[q],
                                                     mu_p_times_previous_E);

            if constexpr (prm::phi_linearize || prm::thermodyn_eq)
              {
                const double KbT_over_q = Constants::KB * scratch.T_cell[q] /
                                          (Constants::Q * V_rescale);
                const double ev_rescaled = ev / (Constants::Q * V_rescale);
                const double ec_rescaled = ec / (Constants::Q * V_rescale);

                const double phi_n =
                  (prm::thermodyn_eq) ? 0 : phi_n_vector[q] / V_rescale;
                const double phi_p =
                  (prm::thermodyn_eq) ? 0 : phi_p_vector[q] / V_rescale;

                phi_V_der = -1 / n_rescale *
                            (nv / KbT_over_q *
                               exp((ev_rescaled - V0[q] + phi_p) / KbT_over_q) +
                             nc / KbT_over_q *
                               exp((V0[q] - phi_n - ec_rescaled) / KbT_over_q));
              }

            for (unsigned int i = 0; i < dofs_per_component; ++i)
              {
                const unsigned int ii = scratch.enabled_component_indices[i];
                for (unsigned int j = 0; j < dofs_per_component; ++j)
                  {
                    const unsigned int jj =
                      scratch.enabled_component_indices[j];
                    if constexpr (prm::is_V_enabled)
                      {
                        scratch.permittivity.epsilon_operator_on_cell(
                          q, E[jj], epsilon_times_E);
                        scratch.cc_matrix(i, j) +=
                          (-V[jj] * q1_div[ii] + E[jj] * q1[ii] -
                           epsilon_times_E * z1_grad[ii]) *
                          JxW;

                        if constexpr (prm::phi_linearize || prm::thermodyn_eq)
                          scratch.cc_matrix(i, j) +=
                            -phi_V_der * V[jj] * z1[ii] * JxW;
                        else
                          scratch.cc_matrix(i, j) +=
                            (n[jj] - p[jj]) * z1[ii] * JxW;
                      }
                    if constexpr (prm::is_n_enabled)
                      {
                        scratch.n_mobility.mu_operator_on_cell(q,
                                                               E[jj],
                                                               mu_n_times_E);

                        this->template apply_einstein_diffusion_coefficient<
                          Component::n,
                          false>(scratch,
                                 q,
                                 Wn[jj],
                                 n_einstein_diffusion_coefficient_times_Wn);

                        scratch.cc_matrix(i, j) +=
                          (-n[jj] * q2_div[ii] + Wn[jj] * q2[ii] -
                           n[jj] * (mu_n_times_previous_E * z2_grad[ii]) +
                           n0[q] * (mu_n_times_E * z2_grad[ii]) +
                           n_einstein_diffusion_coefficient_times_Wn *
                             z2_grad[ii] -
                           (dr_n[q] * n[jj] + dr_p[q] * p[jj]) * z2[ii]) *
                          JxW;
                      }
                    if constexpr (prm::is_p_enabled)
                      {
                        scratch.p_mobility.mu_operator_on_cell(q,
                                                               E[jj],
                                                               mu_p_times_E);

                        this->template apply_einstein_diffusion_coefficient<
                          Component::p,
                          false>(scratch,
                                 q,
                                 Wp[jj],
                                 p_einstein_diffusion_coefficient_times_Wp);

                        scratch.cc_matrix(i, j) +=
                          (-p[jj] * q3_div[ii] + Wp[jj] * q3[ii] +
                           p[jj] * (mu_p_times_previous_E * z3_grad[ii]) -
                           p0[q] * (mu_p_times_E * z3_grad[ii]) +
                           p_einstein_diffusion_coefficient_times_Wp *
                             z3_grad[ii] -
                           (dr_n[q] * n[jj] + dr_p[q] * p[jj]) * z3[ii]) *
                          JxW;
                      }
                  }
              }
          }
      }
  }



  template <int dim, typename ProblemType>
  template <typename prm>
  void
  NPSolver<dim, ProblemType>::add_cell_products_to_cc_rhs(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &scratch)
  {
    constexpr bool something_enabled =
      prm::is_V_enabled || prm::is_n_enabled || prm::is_p_enabled;

    if constexpr (something_enabled)
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

        // Prevent warnings about unused variables
        if constexpr (!prm::is_V_enabled)
          {
            (void)electric_potential;
            (void)electric_field;
          }
        if constexpr (!prm::is_n_enabled)
          {
            (void)electron_density;
            (void)electron_displacement;
          }
        if constexpr (!prm::is_p_enabled)
          {
            (void)hole_density;
            (void)hole_displacement;
          }

        const auto &V0  = scratch.previous_c_cell.at(Component::V);
        const auto &E0  = scratch.previous_d_cell.at(Component::V);
        const auto &n0  = scratch.previous_c_cell.at(Component::n);
        const auto &Wn0 = scratch.previous_d_cell.at(Component::n);
        const auto &p0  = scratch.previous_c_cell.at(Component::p);
        const auto &Wp0 = scratch.previous_d_cell.at(Component::p);

        const auto &c0 = scratch.doping_cell;

        const auto &phi_n_vector = scratch.phi.at(Component::n);
        const auto &phi_p_vector = scratch.phi.at(Component::p);

        if constexpr (!prm::phi_linearize)
          {
            (void)phi_n_vector;
            (void)phi_p_vector;
          }

        dealii::Tensor<1, dim> epsilon_times_E;

        dealii::Tensor<1, dim> n_einstein_diffusion_coefficient_times_Wn0;
        dealii::Tensor<1, dim> p_einstein_diffusion_coefficient_times_Wp0;

        dealii::Tensor<1, dim> mu_n_times_previous_E;
        dealii::Tensor<1, dim> mu_p_times_previous_E;

        dealii::Tensor<1, dim> Jn;
        dealii::Tensor<1, dim> Jp;

        const double nc = this->problem->band_density.at(Component::n);
        const double nv = this->problem->band_density.at(Component::p);
        const double ec =
          this->problem->band_edge_energy.at(Component::n) * Constants::EV;
        const double ev =
          this->problem->band_edge_energy.at(Component::p) * Constants::EV;

        if constexpr (!prm::thermodyn_eq)
          {
            (void)nc;
            (void)nv;
          }

        double phi_rhs = 0.;

        const double V_rescale =
          this->adimensionalizer
            ->template get_component_rescaling_factor<Component::V>();
        const double n_rescale =
          this->adimensionalizer
            ->template get_component_rescaling_factor<Component::n>();

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const double JxW = scratch.fe_values_cell.JxW(q);

            if constexpr (prm::is_V_enabled)
              scratch.permittivity.epsilon_operator_on_cell(q,
                                                            E0[q],
                                                            epsilon_times_E);

            if constexpr (prm::is_n_enabled)
              {
                scratch.n_mobility.mu_operator_on_cell(q,
                                                       E0[q],
                                                       mu_n_times_previous_E);

                this
                  ->template apply_einstein_diffusion_coefficient<Component::n,
                                                                  false>(
                    scratch,
                    q,
                    Wn0[q],
                    n_einstein_diffusion_coefficient_times_Wn0);

                Jn = n0[q] * mu_n_times_previous_E -
                     n_einstein_diffusion_coefficient_times_Wn0;
              }
            if constexpr (prm::is_p_enabled)
              {
                scratch.p_mobility.mu_operator_on_cell(q,
                                                       E0[q],
                                                       mu_p_times_previous_E);

                this
                  ->template apply_einstein_diffusion_coefficient<Component::p,
                                                                  false>(
                    scratch,
                    q,
                    Wp0[q],
                    p_einstein_diffusion_coefficient_times_Wp0);

                Jp = -p0[q] * mu_p_times_previous_E -
                     p_einstein_diffusion_coefficient_times_Wp0;
              }

            if constexpr (prm::phi_linearize || prm::thermodyn_eq)
              {
                const double KbT_over_q = Constants::KB * scratch.T_cell[q] /
                                          (Constants::Q * V_rescale);
                const double ev_rescaled = ev / (Constants::Q * V_rescale);
                const double ec_rescaled = ec / (Constants::Q * V_rescale);

                const double phi_n =
                  (prm::thermodyn_eq) ? 0 : phi_n_vector[q] / V_rescale;
                const double phi_p =
                  (prm::thermodyn_eq) ? 0 : phi_p_vector[q] / V_rescale;

                phi_rhs =
                  1 / n_rescale *
                  (nv * exp((ev_rescaled - V0[q] + phi_p) / KbT_over_q) -
                   nc * exp((V0[q] - phi_n - ec_rescaled) / KbT_over_q));
              }

            for (unsigned int i = 0; i < dofs_per_component; ++i)
              {
                const unsigned int ii = scratch.enabled_component_indices[i];
                if constexpr (prm::is_V_enabled)
                  {
                    const dealii::Tensor<1, dim> q1 =
                      scratch.fe_values_cell[electric_field].value(ii, q);
                    const double q1_div =
                      scratch.fe_values_cell[electric_field].divergence(ii, q);
                    const double z1 =
                      scratch.fe_values_cell[electric_potential].value(ii, q);
                    const dealii::Tensor<1, dim> z1_grad =
                      scratch.fe_values_cell[electric_potential].gradient(ii,
                                                                          q);

                    scratch.cc_rhs[i] +=
                      (V0[q] * q1_div - E0[q] * q1 + epsilon_times_E * z1_grad +
                       c0[q] * z1) *
                      JxW;

                    if (prm::phi_linearize || prm::thermodyn_eq)
                      scratch.cc_rhs[i] += phi_rhs * z1 * JxW;
                    else
                      scratch.cc_rhs[i] += (-n0[q] + p0[q]) * z1 * JxW;
                  }

                if constexpr (prm::is_n_enabled)
                  {
                    const dealii::Tensor<1, dim> q2 =
                      scratch.fe_values_cell[electron_displacement].value(ii,
                                                                          q);
                    const double q2_div =
                      scratch.fe_values_cell[electron_displacement].divergence(
                        ii, q);
                    const double z2 =
                      scratch.fe_values_cell[electron_density].value(ii, q);
                    const dealii::Tensor<1, dim> z2_grad =
                      scratch.fe_values_cell[electron_density].gradient(ii, q);

                    scratch.cc_rhs[i] +=
                      (n0[q] * q2_div - Wn0[q] * q2 + scratch.r_cell[q] * z2 +
                       Jn * z2_grad) *
                      JxW;
                  }

                if constexpr (prm::is_p_enabled)
                  {
                    const dealii::Tensor<1, dim> q3 =
                      scratch.fe_values_cell[hole_displacement].value(ii, q);
                    const double q3_div =
                      scratch.fe_values_cell[hole_displacement].divergence(ii,
                                                                           q);
                    const double z3 =
                      scratch.fe_values_cell[hole_density].value(ii, q);
                    const dealii::Tensor<1, dim> z3_grad =
                      scratch.fe_values_cell[hole_density].gradient(ii, q);

                    scratch.cc_rhs[i] +=
                      (p0[q] * q3_div - Wp0[q] * q3 + +scratch.r_cell[q] * z3 +
                       Jp * z3_grad) *
                      JxW;
                  }
              }
          }
      }
  }



  template <int dim, typename ProblemType>
  template <typename TauComputerClass>
  void
  NPSolver<dim, ProblemType>::prepare_data_on_face_quadrature_points(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &      scratch,
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const unsigned int                                    face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    TauComputerClass *tau_computer =
      static_cast<TauComputerClass *>(scratch.tau_computer.get());

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      scratch.face_quadrature_points[q] =
        scratch.fe_face_values_trace_restricted.quadrature_point(q);

    for (const auto c : Ddhdg::all_primary_components())
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

    scratch.permittivity.initialize_on_face(scratch.face_quadrature_points);

    if (this->is_enabled(Component::n))
      scratch.n_mobility.initialize_on_face(scratch.face_quadrature_points);
    if (this->is_enabled(Component::p))
      scratch.p_mobility.initialize_on_face(scratch.face_quadrature_points);

    this->problem->temperature->value_list(scratch.face_quadrature_points,
                                           scratch.T_face);

    for (const auto c : this->enabled_components)
      tau_computer->template compute_tau<dim>(
        scratch.face_quadrature_points, cell, face, c, scratch.tau.at(c));

    const double thermal_voltage_rescaling_factor =
      this->adimensionalizer->get_thermal_voltage_rescaling_factor();
    for (unsigned int q = 0; q < n_face_q_points; ++q)
      scratch.U_T_face[q] = (Constants::KB * scratch.T_face[q] / Constants::Q) /
                            thermal_voltage_rescaling_factor;
  }



  template <int dim, typename ProblemType>
  inline void
  NPSolver<dim, ProblemType>::copy_fe_values_on_scratch(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &scratch,
    const unsigned int                              face,
    const unsigned int                              q)
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



  template <int dim, typename ProblemType>
  inline void
  NPSolver<dim, ProblemType>::copy_fe_values_for_trace(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &scratch,
    const unsigned int                              face,
    const unsigned int                              q)
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



  template <int dim, typename ProblemType>
  template <typename prm, DDFluxType dd_flux_type>
  inline void
  NPSolver<dim, ProblemType>::assemble_ct_matrix(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &scratch,
    const unsigned int                              face)
  {
    constexpr bool something_enabled =
      prm::is_V_enabled || prm::is_n_enabled || prm::is_p_enabled;

    if constexpr (!something_enabled)
      (void)face;

    if constexpr (something_enabled)
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

        const auto &E0 = scratch.previous_d_face.at(Component::V);

        dealii::Tensor<1, dim> mu_n_times_previous_E;
        dealii::Tensor<1, dim> mu_p_times_previous_E;

        const std::vector<double> &V_tau = scratch.tau.at(Component::V);
        const std::vector<double> &n_tau = scratch.tau.at(Component::n);
        const std::vector<double> &p_tau = scratch.tau.at(Component::p);

        double E0_times_n = 0.;

        if constexpr (!prm::is_V_enabled)
          (void)V_tau;
        if constexpr (!prm::is_n_enabled)
          (void)n_tau;
        if constexpr (!prm::is_p_enabled)
          (void)p_tau;

        for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            copy_fe_values_on_scratch(scratch, face, q);
            copy_fe_values_for_trace(scratch, face, q);

            const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
            const Tensor<1, dim> normal =
              scratch.fe_face_values_trace_restricted.normal_vector(q);

            if constexpr (dd_flux_type == DDFluxType::qiu_shi_stabilization &&
                          (prm::is_n_enabled || prm::is_p_enabled))
              E0_times_n = E0[q] * normal;

            if constexpr (prm::is_n_enabled &&
                          dd_flux_type != DDFluxType::use_cell)
              scratch.n_mobility.mu_operator_on_face(q,
                                                     E0[q],
                                                     mu_n_times_previous_E);
            if constexpr (prm::is_p_enabled &&
                          dd_flux_type != DDFluxType::use_cell)
              scratch.p_mobility.mu_operator_on_face(q,
                                                     E0[q],
                                                     mu_p_times_previous_E);

            for (unsigned int i = 0; i < cell_dofs_on_face; ++i)
              {
                const unsigned int ii =
                  scratch.fe_cell_support_on_face[face][i];
                for (unsigned int j = 0; j < trace_dofs_on_face; ++j)
                  {
                    const unsigned int jj =
                      scratch.fe_trace_support_on_face[face][j];

                    // Integrals of trace functions using as test function
                    // the restriction of cell test function on the border
                    // i is the index of the test function
                    if constexpr (prm::is_V_enabled)
                      {
                        scratch.ct_matrix(ii, jj) +=
                          (tr_V[j] * (q1[i] * normal) -
                           V_tau[q] * tr_V[j] * z1[i]) *
                          JxW;
                      }
                    if constexpr (prm::is_n_enabled)
                      {
                        if constexpr (dd_flux_type == DDFluxType::use_trace)
                          scratch.ct_matrix(ii, jj) += mu_n_times_previous_E *
                                                       normal * tr_n[j] *
                                                       z2[i] * JxW;
                        if constexpr (dd_flux_type ==
                                      DDFluxType::qiu_shi_stabilization)
                          if (E0_times_n <= 0)
                            scratch.ct_matrix(ii, jj) += mu_n_times_previous_E *
                                                         normal * tr_n[j] *
                                                         z2[i] * JxW;
                        scratch.ct_matrix(ii, jj) +=
                          (tr_n[j] * (q2[i] * normal) +
                           n_tau[q] * (tr_n[j] * z2[i])) *
                          JxW;
                      }
                    if constexpr (prm::is_p_enabled)
                      {
                        if constexpr (dd_flux_type == DDFluxType::use_trace)
                          scratch.ct_matrix(ii, jj) += -mu_p_times_previous_E *
                                                       normal * tr_p[j] *
                                                       z3[i] * JxW;
                        if constexpr (dd_flux_type ==
                                      DDFluxType::qiu_shi_stabilization)
                          if (E0_times_n <= 0)
                            scratch.ct_matrix(ii, jj) +=
                              -mu_p_times_previous_E * normal * tr_p[j] *
                              z3[i] * JxW;
                        scratch.ct_matrix(ii, jj) +=
                          (tr_p[j] * (q3[i] * normal) +
                           p_tau[q] * (tr_p[j] * z3[i])) *
                          JxW;
                      }
                  }
              }
          }
      }
  }



  template <int dim, typename ProblemType>
  template <typename prm>
  inline void
  NPSolver<dim, ProblemType>::add_ct_matrix_terms_to_cc_rhs(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &scratch,
    const unsigned int                              face)
  {
    constexpr bool something_enabled =
      prm::is_V_enabled || prm::is_n_enabled || prm::is_p_enabled;

    if constexpr (!something_enabled)
      (void)face;

    if constexpr (something_enabled)
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

        if constexpr (!prm::is_V_enabled)
          {
            (void)electric_field;
            (void)electric_potential;
          }
        if constexpr (!prm::is_n_enabled)
          {
            (void)electron_displacement;
            (void)electron_density;
          }
        if constexpr (!prm::is_p_enabled)
          {
            (void)hole_displacement;
            (void)hole_density;
          }

        const auto &tr_V0 = scratch.previous_tr_c_face.at(Component::V);
        const auto &tr_n0 = scratch.previous_tr_c_face.at(Component::n);
        const auto &tr_p0 = scratch.previous_tr_c_face.at(Component::p);

        const std::vector<double> &V_tau = scratch.tau.at(Component::V);
        const std::vector<double> &n_tau = scratch.tau.at(Component::n);
        const std::vector<double> &p_tau = scratch.tau.at(Component::p);

        if constexpr (!prm::is_V_enabled)
          (void)V_tau;
        if constexpr (!prm::is_n_enabled)
          (void)n_tau;
        if constexpr (!prm::is_p_enabled)
          (void)p_tau;

        for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
            const Tensor<1, dim> normal =
              scratch.fe_face_values_trace_restricted.normal_vector(q);

            for (unsigned int i = 0;
                 i < scratch.fe_cell_support_on_face[face].size();
                 ++i)
              {
                const unsigned int ii =
                  scratch.fe_cell_support_on_face[face][i];
                const unsigned int iii = scratch.enabled_component_indices[ii];

                if constexpr (prm::is_V_enabled)
                  {
                    const dealii::Tensor<1, dim> q1 =
                      scratch.fe_face_values_cell[electric_field].value(iii, q);
                    const double z1 =
                      scratch.fe_face_values_cell[electric_potential].value(iii,
                                                                            q);

                    scratch.cc_rhs(ii) +=
                      (-tr_V0[q] * (q1 * normal) + V_tau[q] * tr_V0[q] * z1) *
                      JxW;
                  }

                if constexpr (prm::is_n_enabled)
                  {
                    const dealii::Tensor<1, dim> q2 =
                      scratch.fe_face_values_cell[electron_displacement].value(
                        iii, q);
                    const double z2 =
                      scratch.fe_face_values_cell[electron_density].value(iii,
                                                                          q);

                    scratch.cc_rhs(ii) +=
                      (-tr_n0[q] * (q2 * normal) - n_tau[q] * tr_n0[q] * z2) *
                      JxW;
                  }

                if constexpr (prm::is_p_enabled)
                  {
                    const dealii::Tensor<1, dim> q3 =
                      scratch.fe_face_values_cell[hole_displacement].value(iii,
                                                                           q);
                    const double z3 =
                      scratch.fe_face_values_cell[hole_density].value(iii, q);

                    scratch.cc_rhs(ii) +=
                      (-tr_p0[q] * (q3 * normal) - p_tau[q] * tr_p0[q] * z3) *
                      JxW;
                  }
              }
          }
      }
  }



  template <int dim, typename ProblemType>
  template <typename prm, Component c, DDFluxType dd_flux_type>
  inline void
  NPSolver<dim, ProblemType>::assemble_tc_matrix(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &scratch,
    const unsigned int                              face)
  {
    if constexpr (c != Component::V && c != Component::n && c != Component::p)
      AssertThrow(false, InvalidComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    auto &c_ = scratch.c.at(c);
    auto &f  = scratch.d.at(c);
    auto &xi = scratch.tr_c.at(c);

    const auto &E0 = scratch.previous_d_face.at(Component::V);

    dealii::Tensor<1, dim> epsilon_times_E;

    dealii::Tensor<1, dim> mu_n_times_E;
    dealii::Tensor<1, dim> mu_p_times_E;

    dealii::Tensor<1, dim> mu_n_times_previous_E;
    dealii::Tensor<1, dim> mu_p_times_previous_E;

    dealii::Tensor<1, dim> n_einstein_diffusion_coefficient_times_f;
    dealii::Tensor<1, dim> p_einstein_diffusion_coefficient_times_f;

    const std::vector<double> &n0 = scratch.previous_c_face.at(Component::n);
    const std::vector<double> &p0 = scratch.previous_c_face.at(Component::p);

    const std::vector<double> &tr_n0 =
      scratch.previous_tr_c_face.at(Component::n);
    const std::vector<double> &tr_p0 =
      scratch.previous_tr_c_face.at(Component::p);

    const std::vector<double> &tau = scratch.tau.at(c);

    double E0_times_n = 0.;

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);
        copy_fe_values_for_trace(scratch, face, q);

        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        if constexpr (dd_flux_type == DDFluxType::qiu_shi_stabilization &&
                      (c == Component::n || c == Component::p))
          E0_times_n = E0[q] * normal;

        if constexpr (c == Component::n)
          scratch.n_mobility.mu_operator_on_face(q,
                                                 E0[q],
                                                 mu_n_times_previous_E);
        if constexpr (c == Component::p)
          scratch.p_mobility.mu_operator_on_face(q,
                                                 E0[q],
                                                 mu_p_times_previous_E);

        const unsigned int trace_dofs_per_face =
          scratch.fe_trace_support_on_face[face].size();
        const unsigned int cell_dofs_per_face =
          scratch.fe_cell_support_on_face[face].size();

        double cE_flux;
        if constexpr (c == Component::V)
          (void)cE_flux; // Prevent unused-but-set-variable under GCC

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
                if constexpr (c == Component::V)
                  {
                    scratch.permittivity.epsilon_operator_on_face(
                      q, f[j], epsilon_times_E);
                    scratch.tc_matrix(ii, jj) -=
                      (epsilon_times_E * normal + tau[q] * c_[j]) * xi[i] * JxW;
                  }
                if constexpr (c == Component::n)
                  {
                    scratch.n_mobility.mu_operator_on_face(
                      q, scratch.d[Component::V][j], mu_n_times_E);

                    this->template apply_einstein_diffusion_coefficient<
                      Component::n>(scratch,
                                    q,
                                    f[j],
                                    n_einstein_diffusion_coefficient_times_f);

                    if constexpr (dd_flux_type == DDFluxType::use_cell)
                      cE_flux =
                        (c_[j] * mu_n_times_previous_E + n0[q] * mu_n_times_E) *
                        normal;
                    else if constexpr (dd_flux_type == DDFluxType::use_trace)
                      cE_flux = (tr_n0[q] * mu_n_times_E) * normal;
                    else if constexpr (dd_flux_type ==
                                       DDFluxType::qiu_shi_stabilization)
                      cE_flux = (E0_times_n > 0) ?
                                  (c_[j] * mu_n_times_previous_E +
                                   n0[q] * mu_n_times_E) *
                                    normal :
                                  (tr_n0[q] * mu_n_times_E) * normal;
                    else
                      Assert(false, InvalidDDFluxType());

                    scratch.tc_matrix(ii, jj) -=
                      (cE_flux -
                       n_einstein_diffusion_coefficient_times_f * normal -
                       tau[q] * c_[j]) *
                      xi[i] * JxW;
                  }
                if constexpr (c == Component::p)
                  {
                    scratch.p_mobility.mu_operator_on_face(
                      q, scratch.d[Component::V][j], mu_p_times_E);

                    this->template apply_einstein_diffusion_coefficient<
                      Component::p>(scratch,
                                    q,
                                    f[j],
                                    p_einstein_diffusion_coefficient_times_f);

                    if constexpr (dd_flux_type == DDFluxType::use_cell)
                      cE_flux = (-c_[j] * mu_p_times_previous_E -
                                 p0[q] * mu_p_times_E) *
                                normal;
                    else if constexpr (dd_flux_type == DDFluxType::use_trace)
                      cE_flux = (-tr_p0[q] * mu_p_times_E) * normal;
                    else if constexpr (dd_flux_type ==
                                       DDFluxType::qiu_shi_stabilization)
                      cE_flux = (E0_times_n > 0) ?
                                  (-c_[j] * mu_p_times_previous_E -
                                   p0[q] * mu_p_times_E) *
                                    normal :
                                  (-tr_p0[q] * mu_p_times_E) * normal;
                    else
                      Assert(false, InvalidDDFluxType());

                    scratch.tc_matrix(ii, jj) -=
                      (cE_flux -
                       p_einstein_diffusion_coefficient_times_f * normal -
                       tau[q] * c_[j]) *
                      xi[i] * JxW;
                  }
              }
          }
      }
  }



  template <int dim, typename ProblemType>
  template <typename prm, Component c, DDFluxType dd_flux_type>
  inline void
  NPSolver<dim, ProblemType>::add_tc_matrix_terms_to_tt_rhs(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &scratch,
    Ddhdg::NPSolver<dim, ProblemType>::PerTaskData &task_data,
    const unsigned int                              face)
  {
    if constexpr (c != Component::V && c != Component::n && c != Component::p)
      AssertThrow(false, InvalidComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    const FEValuesExtractors::Scalar tr_c_extractor =
      this->get_trace_component_extractor(c, true);

    auto &c0    = scratch.previous_c_face.at(c);
    auto &f0    = scratch.previous_d_face.at(c);
    auto &tr_c0 = scratch.previous_tr_c_face.at(c);

    const auto &E0 = scratch.previous_d_face[Component::V];

    dealii::Tensor<1, dim> epsilon_times_E;
    dealii::Tensor<1, dim> mu_n_times_previous_E;
    dealii::Tensor<1, dim> mu_p_times_previous_E;
    double                 epsilon_times_previous_E_times_normal = 0.;
    double                 mu_n_times_previous_E_times_normal    = 0.;
    double                 mu_p_times_previous_E_times_normal    = 0.;

    // Avoid warnings with GCC
    if constexpr (c != Component::V)
      (void)epsilon_times_previous_E_times_normal;
    if constexpr (c != Component::n)
      (void)mu_n_times_previous_E_times_normal;
    if constexpr (c != Component::p)
      (void)mu_p_times_previous_E_times_normal;

    dealii::Tensor<1, dim> n_einstein_diffusion_coefficient_times_f0;
    dealii::Tensor<1, dim> p_einstein_diffusion_coefficient_times_f0;

    const std::vector<double> &tau = scratch.tau.at(c);

    double n0;
    double p0;

    double E0_times_n = 0.;

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        if constexpr (dd_flux_type == DDFluxType::qiu_shi_stabilization &&
                      (c == Component::n || c == Component::p))
          E0_times_n = E0[q] * normal;

        if constexpr (c == Component::V)
          {
            scratch.permittivity.epsilon_operator_on_face(q,
                                                          E0[q],
                                                          epsilon_times_E);
            epsilon_times_previous_E_times_normal = epsilon_times_E * normal;
          }
        if constexpr (c == Component::n)
          {
            scratch.n_mobility.mu_operator_on_face(q,
                                                   E0[q],
                                                   mu_n_times_previous_E);
            mu_n_times_previous_E_times_normal = mu_n_times_previous_E * normal;
            this->template apply_einstein_diffusion_coefficient<Component::n>(
              scratch, q, f0[q], n_einstein_diffusion_coefficient_times_f0);
          }
        if constexpr (c == Component::p)
          {
            scratch.p_mobility.mu_operator_on_face(q,
                                                   E0[q],
                                                   mu_p_times_previous_E);
            mu_p_times_previous_E_times_normal = mu_p_times_previous_E * normal;
            this->template apply_einstein_diffusion_coefficient<Component::p>(
              scratch, q, f0[q], p_einstein_diffusion_coefficient_times_f0);
          }

        for (unsigned int i = 0;
             i < scratch.fe_trace_support_on_face[face].size();
             ++i)
          {
            const unsigned int ii = scratch.fe_trace_support_on_face[face][i];
            const double       xi =
              scratch.fe_face_values_trace_restricted[tr_c_extractor].value(ii,
                                                                            q);
            if constexpr (c == Component::V)
              {
                task_data.tt_vector[ii] +=
                  (-epsilon_times_previous_E_times_normal - tau[q] * c0[q]) *
                  xi * JxW;
              }
            if constexpr (c == Component::n)
              {
                if constexpr (dd_flux_type == DDFluxType::use_cell)
                  n0 = c0[q];
                else if constexpr (dd_flux_type == DDFluxType::use_trace)
                  n0 = tr_c0[q];
                else if constexpr (dd_flux_type ==
                                   DDFluxType::qiu_shi_stabilization)
                  n0 = (E0_times_n > 0) ? c0[q] : tr_c0[q];
                else
                  Assert(false, InvalidDDFluxType());

                task_data.tt_vector[ii] +=
                  (-n0 * mu_n_times_previous_E_times_normal +
                   n_einstein_diffusion_coefficient_times_f0 * normal +
                   tau[q] * c0[q]) *
                  xi * JxW;
              }
            if constexpr (c == Component::p)
              {
                if constexpr (dd_flux_type == DDFluxType::use_cell)
                  p0 = c0[q];
                else if constexpr (dd_flux_type == DDFluxType::use_trace)
                  p0 = tr_c0[q];
                else if constexpr (dd_flux_type ==
                                   DDFluxType::qiu_shi_stabilization)
                  p0 = (E0_times_n > 0) ? c0[q] : tr_c0[q];
                else
                  Assert(false, InvalidDDFluxType());

                task_data.tt_vector[ii] +=
                  (p0 * mu_p_times_previous_E_times_normal +
                   p_einstein_diffusion_coefficient_times_f0 * normal +
                   tau[q] * c0[q]) *
                  xi * JxW;
              }
          }
      }
  }



  template <int dim, typename ProblemType>
  template <typename prm, Component c, DDFluxType dd_flux_type>
  inline void
  NPSolver<dim, ProblemType>::assemble_tt_matrix(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &scratch,
    Ddhdg::NPSolver<dim, ProblemType>::PerTaskData &task_data,
    const unsigned int                              face)
  {
    if constexpr (c != Component::V && c != Component::n && c != Component::p)
      AssertThrow(false, InvalidComponent());

    auto &tr_c = scratch.tr_c.at(c);
    auto &xi   = scratch.tr_c.at(c);

    const std::vector<double> &tau  = scratch.tau.at(c);
    const double               sign = (c == Component::V) ? -1 : 1;

    const auto &E0 = scratch.previous_d_face.at(Component::V);

    dealii::Tensor<1, dim> mu_n_times_previous_E;
    dealii::Tensor<1, dim> mu_p_times_previous_E;

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();
    const unsigned int trace_dofs_per_face =
      scratch.fe_trace_support_on_face[face].size();

    double E0_times_n = 0;

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        copy_fe_values_on_scratch(scratch, face, q);
        copy_fe_values_for_trace(scratch, face, q);

        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Tensor<1, dim> normal =
          scratch.fe_face_values_trace_restricted.normal_vector(q);

        if constexpr (c == Component::V ||
                      (dd_flux_type != DDFluxType::use_trace &&
                       dd_flux_type != DDFluxType::qiu_shi_stabilization))
          (void)normal;

        if constexpr (dd_flux_type == DDFluxType::qiu_shi_stabilization &&
                      (c == Component::n || c == Component::p))
          E0_times_n = E0[q] * normal;

        if constexpr (c == Component::n && dd_flux_type != DDFluxType::use_cell)
          scratch.n_mobility.mu_operator_on_face(q,
                                                 E0[q],
                                                 mu_n_times_previous_E);
        if constexpr (c == Component::p && dd_flux_type != DDFluxType::use_cell)
          scratch.p_mobility.mu_operator_on_face(q,
                                                 E0[q],
                                                 mu_p_times_previous_E);

        // Integrals of trace functions (both test and trial)
        for (unsigned int i = 0; i < trace_dofs_per_face; ++i)
          {
            const unsigned int ii = scratch.fe_trace_support_on_face[face][i];
            for (unsigned int j = 0; j < trace_dofs_per_face; ++j)
              {
                const unsigned int jj =
                  scratch.fe_trace_support_on_face[face][j];

                if constexpr (c == Component::n)
                  {
                    if constexpr (dd_flux_type == DDFluxType::use_trace)
                      task_data.tt_matrix(ii, jj) +=
                        mu_n_times_previous_E * normal * tr_c[j] * xi[i] * JxW;
                    if constexpr (dd_flux_type ==
                                  DDFluxType::qiu_shi_stabilization)
                      if (E0_times_n <= 0)
                        task_data.tt_matrix(ii, jj) += mu_n_times_previous_E *
                                                       normal * tr_c[j] *
                                                       xi[i] * JxW;
                  }

                if constexpr (c == Component::p)
                  {
                    if constexpr (dd_flux_type == DDFluxType::use_trace)
                      task_data.tt_matrix(ii, jj) +=
                        -mu_p_times_previous_E * normal * tr_c[j] * xi[i] * JxW;
                    if constexpr (dd_flux_type ==
                                  DDFluxType::qiu_shi_stabilization)
                      if (E0_times_n <= 0)
                        task_data.tt_matrix(ii, jj) += -mu_p_times_previous_E *
                                                       normal * tr_c[j] *
                                                       xi[i] * JxW;
                  }

                task_data.tt_matrix(ii, jj) +=
                  sign * tau[q] * tr_c[j] * xi[i] * JxW;
              }
          }
      }
  }



  template <int dim, typename ProblemType>
  template <typename prm, Component c>
  inline void
  NPSolver<dim, ProblemType>::add_tt_matrix_terms_to_tt_rhs(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &scratch,
    Ddhdg::NPSolver<dim, ProblemType>::PerTaskData &task_data,
    const unsigned int                              face)
  {
    if constexpr (c != Component::V && c != Component::n && c != Component::p)
      AssertThrow(false, InvalidComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();
    const unsigned int trace_dofs_per_face =
      scratch.fe_trace_support_on_face[face].size();

    const std::vector<double> &tau  = scratch.tau.at(c);
    const double               sign = (c == Component::V) ? -1 : 1;

    const FEValuesExtractors::Scalar tr_c_extractor =
      this->get_trace_component_extractor(c, true);

    const auto &tr_c0 = scratch.previous_tr_c_face.at(c);

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);

        for (unsigned int i = 0; i < trace_dofs_per_face; ++i)
          {
            const unsigned int ii = scratch.fe_trace_support_on_face[face][i];
            const double       xi =
              scratch.fe_face_values_trace_restricted[tr_c_extractor].value(ii,
                                                                            q);
            task_data.tt_vector[ii] += -sign * tau[q] * tr_c0[q] * xi * JxW;
          }
      }
  }



  template <int dim, typename ProblemType>
  template <typename prm, Ddhdg::Component c>
  inline void
  NPSolver<dim, ProblemType>::apply_dbc_on_face(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &scratch,
    Ddhdg::NPSolver<dim, ProblemType>::PerTaskData &task_data,
    const Ddhdg::DirichletBoundaryCondition<dim> &  dbc,
    unsigned int                                    face)
  {
    if constexpr (c != Component::V && c != Component::n && c != Component::p)
      Assert(false, InvalidComponent());

    // This used to be needed to store the Dirichlet boundary conditions; now we
    // do not have that problem anymore because we have the local_condenser
    (void)task_data;

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

    auto &tr_c0 = scratch.previous_tr_c_face.at(c);

    const double rescaling_factor =
      this->adimensionalizer->template get_component_rescaling_factor<c>();

    // We want now to compute the L2 projection of the Dirichlet boundary
    // condition on the current face
    const unsigned int c_index = scratch.local_condenser.get_component_index(c);
    const auto c_extractor     = this->get_trace_component_extractor(c, true);

    const auto &dofs_on_this_face = scratch.fe_trace_support_on_face[face];

    auto &proj_matrix  = scratch.local_condenser.proj_matrix[c_index];
    auto &proj_rhs     = scratch.local_condenser.proj_rhs[c_index];
    auto &current_dofs = scratch.local_condenser.current_dofs[c_index];
    auto &bf_values    = scratch.local_condenser.bf_values[c_index];

    // Now I clean the memory for the projection matrix and for the rhs
    proj_matrix = 0;
    proj_rhs    = 0;

    // Copy the index of the dofs that are constrained on the current face
    // and count their number
    unsigned int constrained_dofs = 0;
    for (unsigned int i = 0; i < dofs_on_this_face.size(); ++i)
      {
        const unsigned int dof_index = dofs_on_this_face[i];
        if (this->fe_trace_restricted->system_to_block_index(i).first ==
            c_index)
          {
            current_dofs[constrained_dofs] = dof_index;
            ++constrained_dofs;
          }
      }

    // Let us assemble the projection matrix and its rhs
    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        // First of all we prepare a vector with the values of the base
        // functions
        for (unsigned int i = 0; i < constrained_dofs; i++)
          {
            const unsigned int dof_index = current_dofs[i];
            const double       bf_value =
              scratch.fe_face_values_trace_restricted[c_extractor].value(
                dof_index, q);
            bf_values[i] = bf_value;
          }

        const double     JxW = scratch.fe_face_values_trace_restricted.JxW(q);
        const Point<dim> quadrature_point =
          scratch.fe_face_values_trace_restricted.quadrature_point(q);
        double dbc_value =
          (prm::thermodyn_eq) ?
            0 :
            dbc.evaluate(quadrature_point) / rescaling_factor - tr_c0[q];

        for (unsigned int i = 0; i < constrained_dofs; i++)
          {
            for (unsigned int j = 0; j < constrained_dofs; j++)
              proj_matrix(i, j) += bf_values[i] * bf_values[j] * JxW;
            proj_rhs[i] += bf_values[i] * dbc_value * JxW;
          }
      }

    // Solve the linear system. The solution will be stored in proj_rhs
    proj_matrix.set_property(dealii::LAPACKSupport::Property::symmetric);
    proj_matrix.compute_cholesky_factorization();
    proj_matrix.solve(proj_rhs);

    // Store the solution in the local_condenser
    const unsigned int n_cell_constrained_dofs =
      scratch.local_condenser.n_cell_constrained_dofs;
    for (unsigned int i = 0; i < constrained_dofs; ++i)
      {
        const unsigned int j = n_cell_constrained_dofs + i;
        scratch.local_condenser.cell_constrained_dofs[j]   = current_dofs[i];
        scratch.local_condenser.constrained_dofs_values[j] = proj_rhs[i];
      }
    scratch.local_condenser.n_cell_constrained_dofs += constrained_dofs;
  }



  template <int dim, typename ProblemType>
  template <Ddhdg::Component c>
  void
  NPSolver<dim, ProblemType>::apply_nbc_on_face(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &scratch,
    Ddhdg::NPSolver<dim, ProblemType>::PerTaskData &task_data,
    const Ddhdg::NeumannBoundaryCondition<dim> &    nbc,
    unsigned int                                    face)
  {
    if constexpr (c != Component::V && c != Component::n && c != Component::p)
      Assert(false, InvalidComponent());

    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();
    const unsigned int trace_dofs_per_face =
      scratch.fe_trace_support_on_face[face].size();

    auto &tr_c = scratch.tr_c.at(c);

    double rescaling_factor =
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


  template <int dim, typename ProblemType>
  template <typename prm, DDFluxType dd_flux_type>
  inline void
  NPSolver<dim, ProblemType>::add_border_products_to_cc_matrix(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &scratch,
    const unsigned int                              face)
  {
    constexpr bool something_enabled =
      prm::is_V_enabled || prm::is_n_enabled || prm::is_p_enabled;

    if constexpr (!something_enabled)
      (void)face;

    if constexpr (something_enabled)
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

        const auto  n0    = scratch.previous_c_face.at(Component::n);
        const auto  p0    = scratch.previous_c_face.at(Component::p);
        const auto &E0    = scratch.previous_d_face.at(Component::V);
        const auto  tr_n0 = scratch.previous_tr_c_face.at(Component::n);
        const auto  tr_p0 = scratch.previous_tr_c_face.at(Component::p);

        const std::vector<double> &V_tau = scratch.tau.at(Component::V);
        const std::vector<double> &n_tau = scratch.tau.at(Component::n);
        const std::vector<double> &p_tau = scratch.tau.at(Component::p);

        if constexpr (!prm::is_V_enabled)
          (void)V_tau;
        if constexpr (!prm::is_n_enabled)
          (void)n_tau;
        if constexpr (!prm::is_p_enabled)
          (void)p_tau;

        dealii::Tensor<1, dim> epsilon_times_E;

        dealii::Tensor<1, dim> mu_n_times_previous_E;
        dealii::Tensor<1, dim> mu_p_times_previous_E;

        dealii::Tensor<1, dim> mu_n_times_E;
        dealii::Tensor<1, dim> mu_p_times_E;

        dealii::Tensor<1, dim> n_einstein_diffusion_coefficient_times_Wn;
        dealii::Tensor<1, dim> p_einstein_diffusion_coefficient_times_Wp;

        double cE_flux;
        if constexpr (!prm::is_n_enabled && !prm::is_p_enabled)
          (void)cE_flux; // Prevent unused-but-set-variable under GCC

        double E0_times_n = 0;

        for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            copy_fe_values_on_scratch(scratch, face, q);

            // Integrals on the border of the cell of the cell functions
            const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
            const Tensor<1, dim> normal =
              scratch.fe_face_values_trace_restricted.normal_vector(q);

            if constexpr (prm::is_n_enabled)
              scratch.n_mobility.mu_operator_on_face(q,
                                                     E0[q],
                                                     mu_n_times_previous_E);

            if constexpr (prm::is_p_enabled)
              scratch.p_mobility.mu_operator_on_face(q,
                                                     E0[q],
                                                     mu_p_times_previous_E);

            if constexpr (dd_flux_type == DDFluxType::qiu_shi_stabilization &&
                          (prm::is_n_enabled || prm::is_p_enabled))
              E0_times_n = E0[q] * normal;

            for (unsigned int i = 0; i < cell_dofs_per_face; ++i)
              {
                const unsigned int ii =
                  scratch.fe_cell_support_on_face[face][i];
                for (unsigned int j = 0; j < cell_dofs_per_face; ++j)
                  {
                    const unsigned int jj =
                      scratch.fe_cell_support_on_face[face][j];

                    if constexpr (prm::is_V_enabled)
                      {
                        scratch.permittivity.epsilon_operator_on_face(
                          q, E[j], epsilon_times_E);
                        scratch.cc_matrix(ii, jj) +=
                          (epsilon_times_E * normal * z1[i] +
                           V_tau[q] * V[j] * z1[i]) *
                          JxW;
                      }
                    if constexpr (prm::is_n_enabled)
                      {
                        scratch.n_mobility.mu_operator_on_face(q,
                                                               E[j],
                                                               mu_n_times_E);

                        this->template apply_einstein_diffusion_coefficient<
                          Component::n>(
                          scratch,
                          q,
                          Wn[j],
                          n_einstein_diffusion_coefficient_times_Wn);

                        if constexpr (dd_flux_type == DDFluxType::use_cell)
                          cE_flux = (n[j] * mu_n_times_previous_E +
                                     mu_n_times_E * n0[q]) *
                                    normal;
                        else if constexpr (dd_flux_type ==
                                           DDFluxType::use_trace)
                          cE_flux = mu_n_times_E * normal * tr_n0[q];
                        else if constexpr (dd_flux_type ==
                                           DDFluxType::qiu_shi_stabilization)
                          if (E0_times_n > 0)
                            cE_flux = (n[j] * mu_n_times_previous_E +
                                       mu_n_times_E * n0[q]) *
                                      normal;
                          else
                            cE_flux = mu_n_times_E * normal * tr_n0[q];
                        else
                          Assert(false, InvalidDDFluxType());

                        scratch.cc_matrix(ii, jj) +=
                          (cE_flux * z2[i] -
                           n_einstein_diffusion_coefficient_times_Wn * normal *
                             z2[i] -
                           n_tau[q] * n[j] * z2[i]) *
                          JxW;
                      }
                    if constexpr (prm::is_p_enabled)
                      {
                        scratch.p_mobility.mu_operator_on_face(q,
                                                               E[j],
                                                               mu_p_times_E);

                        this->template apply_einstein_diffusion_coefficient<
                          Component::p>(
                          scratch,
                          q,
                          Wp[j],
                          p_einstein_diffusion_coefficient_times_Wp);

                        if constexpr (dd_flux_type == DDFluxType::use_cell)
                          cE_flux = -(p[j] * mu_p_times_previous_E +
                                      mu_p_times_E * p0[q]) *
                                    normal;
                        else if constexpr (dd_flux_type ==
                                           DDFluxType::use_trace)
                          cE_flux = -mu_p_times_E * normal * tr_p0[q];
                        else if constexpr (dd_flux_type ==
                                           DDFluxType::qiu_shi_stabilization)
                          if (E0_times_n > 0)
                            cE_flux = -(p[j] * mu_p_times_previous_E +
                                        mu_p_times_E * p0[q]) *
                                      normal;
                          else
                            cE_flux = -mu_p_times_E * normal * tr_p0[q];
                        else
                          Assert(false, InvalidDDFluxType());

                        scratch.cc_matrix(ii, jj) +=
                          (cE_flux * z3[i] -
                           p_einstein_diffusion_coefficient_times_Wp * normal *
                             z3[i] -
                           p_tau[q] * p[j] * z3[i]) *
                          JxW;
                      }
                  }
              }
          }
      }
  }



  template <int dim, typename ProblemType>
  template <typename prm, DDFluxType dd_flux_type>
  inline void
  NPSolver<dim, ProblemType>::add_border_products_to_cc_rhs(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &scratch,
    const unsigned int                              face)
  {
    constexpr bool something_enabled =
      prm::is_V_enabled || prm::is_n_enabled || prm::is_p_enabled;

    if constexpr (!something_enabled)
      (void)face;

    if constexpr (something_enabled)
      {
        const unsigned int n_face_q_points =
          scratch.fe_face_values_cell.get_quadrature().size();

        const FEValuesExtractors::Scalar electric_potential =
          this->get_component_extractor(Component::V);
        const FEValuesExtractors::Scalar electron_density =
          this->get_component_extractor(Component::n);
        const FEValuesExtractors::Scalar hole_density =
          this->get_component_extractor(Component::p);

        if constexpr (!prm::is_V_enabled)
          (void)electric_potential;
        if constexpr (!prm::is_n_enabled)
          (void)electron_density;
        if constexpr (!prm::is_p_enabled)
          (void)hole_density;

        auto &V0  = scratch.previous_c_face.at(Component::V);
        auto &E0  = scratch.previous_d_face.at(Component::V);
        auto &n0  = scratch.previous_c_face.at(Component::n);
        auto &Wn0 = scratch.previous_d_face.at(Component::n);
        auto &p0  = scratch.previous_c_face.at(Component::p);
        auto &Wp0 = scratch.previous_d_face.at(Component::p);

        auto &tr_n0 = scratch.previous_tr_c_face.at(Component::n);
        auto &tr_p0 = scratch.previous_tr_c_face.at(Component::p);

        const std::vector<double> &V_tau = scratch.tau.at(Component::V);
        const std::vector<double> &n_tau = scratch.tau.at(Component::n);
        const std::vector<double> &p_tau = scratch.tau.at(Component::p);

        dealii::Tensor<1, dim> mu_n_times_previous_E;
        dealii::Tensor<1, dim> mu_p_times_previous_E;

        if constexpr (!prm::is_V_enabled)
          (void)V_tau;
        if constexpr (!prm::is_n_enabled)
          (void)n_tau;
        if constexpr (!prm::is_p_enabled)
          (void)p_tau;


        dealii::Tensor<1, dim> epsilon_times_E;
        double                 epsilon_times_E0_times_normal = 0.;

        dealii::Tensor<1, dim> n_einstein_diffusion_coefficient_times_Wn0;
        dealii::Tensor<1, dim> p_einstein_diffusion_coefficient_times_Wp0;

        double E0_times_n = 0;

        double Jn_flux = 0.;
        double Jp_flux = 0.;

        for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            const double JxW = scratch.fe_face_values_trace_restricted.JxW(q);
            const Tensor<1, dim> normal =
              scratch.fe_face_values_trace_restricted.normal_vector(q);

            if constexpr (dd_flux_type == DDFluxType::qiu_shi_stabilization &&
                          (prm::is_n_enabled || prm::is_p_enabled))
              E0_times_n = E0[q] * normal;

            if constexpr (prm::is_V_enabled)
              {
                scratch.permittivity.epsilon_operator_on_face(q,
                                                              E0[q],
                                                              epsilon_times_E);
                epsilon_times_E0_times_normal = epsilon_times_E * normal;
              }

            if constexpr (prm::is_n_enabled)
              {
                scratch.n_mobility.mu_operator_on_face(q,
                                                       E0[q],
                                                       mu_n_times_previous_E);

                this
                  ->template apply_einstein_diffusion_coefficient<Component::n>(
                    scratch,
                    q,
                    Wn0[q],
                    n_einstein_diffusion_coefficient_times_Wn0);

                if constexpr (dd_flux_type == DDFluxType::use_cell)
                  Jn_flux = (n0[q] * mu_n_times_previous_E -
                             n_einstein_diffusion_coefficient_times_Wn0) *
                            normal;
                else if constexpr (dd_flux_type == DDFluxType::use_trace)
                  Jn_flux = (tr_n0[q] * mu_n_times_previous_E -
                             n_einstein_diffusion_coefficient_times_Wn0) *
                            normal;
                else if constexpr (dd_flux_type ==
                                   DDFluxType::qiu_shi_stabilization)
                  if (E0_times_n > 0)
                    Jn_flux = (n0[q] * mu_n_times_previous_E -
                               n_einstein_diffusion_coefficient_times_Wn0) *
                              normal;
                  else
                    Jn_flux = (tr_n0[q] * mu_n_times_previous_E -
                               n_einstein_diffusion_coefficient_times_Wn0) *
                              normal;
                else
                  Assert(false, InvalidDDFluxType());
              }

            if constexpr (prm::is_p_enabled)
              {
                scratch.p_mobility.mu_operator_on_face(q,
                                                       E0[q],
                                                       mu_p_times_previous_E);

                this
                  ->template apply_einstein_diffusion_coefficient<Component::p>(
                    scratch,
                    q,
                    Wp0[q],
                    p_einstein_diffusion_coefficient_times_Wp0);

                if constexpr (dd_flux_type == DDFluxType::use_cell)
                  Jp_flux = (-p0[q] * mu_p_times_previous_E -
                             p_einstein_diffusion_coefficient_times_Wp0) *
                            normal;
                else if constexpr (dd_flux_type == DDFluxType::use_trace)
                  Jp_flux = (-tr_p0[q] * mu_p_times_previous_E -
                             p_einstein_diffusion_coefficient_times_Wp0) *
                            normal;
                else if constexpr (dd_flux_type ==
                                   DDFluxType::qiu_shi_stabilization)
                  if (E0_times_n > 0)
                    Jp_flux = (-p0[q] * mu_p_times_previous_E -
                               p_einstein_diffusion_coefficient_times_Wp0) *
                              normal;
                  else
                    Jp_flux = (-tr_p0[q] * mu_p_times_previous_E -
                               p_einstein_diffusion_coefficient_times_Wp0) *
                              normal;
                else
                  Assert(false, InvalidDDFluxType());
              }

            for (unsigned int i = 0;
                 i < scratch.fe_cell_support_on_face[face].size();
                 ++i)
              {
                const unsigned int ii =
                  scratch.fe_cell_support_on_face[face][i];
                const unsigned int iii = scratch.enabled_component_indices[ii];

                if constexpr (prm::is_V_enabled)
                  {
                    const double z1 =
                      scratch.fe_face_values_cell[electric_potential].value(iii,
                                                                            q);
                    scratch.cc_rhs[ii] += (-epsilon_times_E0_times_normal * z1 -
                                           V_tau[q] * V0[q] * z1) *
                                          JxW;
                  }
                if constexpr (prm::is_n_enabled)
                  {
                    const double z2 =
                      scratch.fe_face_values_cell[electron_density].value(iii,
                                                                          q);
                    scratch.cc_rhs[ii] +=
                      (-Jn_flux * z2 + n_tau[q] * n0[q] * z2) * JxW;
                  }
                if constexpr (prm::is_p_enabled)
                  {
                    const double z3 =
                      scratch.fe_face_values_cell[hole_density].value(iii, q);
                    scratch.cc_rhs[ii] +=
                      (-Jp_flux * z3 + p_tau[q] * p0[q] * z3) * JxW;
                  }
              }
          }
      }
  }



  template <int dim, typename ProblemType>
  template <typename prm>
  inline void
  NPSolver<dim, ProblemType>::add_trace_terms_to_cc_rhs(
    Ddhdg::NPSolver<dim, ProblemType>::ScratchData &scratch,
    const unsigned int                              face)
  {
    const unsigned int n_face_q_points =
      scratch.fe_face_values_cell.get_quadrature().size();

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

            const std::vector<double> &tau  = scratch.tau.at(c);
            const double               sign = (c == Component::V) ? 1 : -1;

            for (unsigned int i = 0;
                 i < scratch.fe_cell_support_on_face[face].size();
                 ++i)
              {
                const unsigned int ii =
                  scratch.fe_cell_support_on_face[face][i];
                scratch.cc_rhs(ii) += (-f[i] * normal + sign * tau[q] * c_[i]) *
                                      tr_c_solution_values[q] * JxW;
              }
          }
      }
  }



  template <int dim, typename ProblemType>
  template <typename prm, Component c>
  void
  NPSolver<dim, ProblemType>::assemble_flux_conditions(
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
        switch (this->parameters->dd_flux_type)
          {
            case DDFluxType::use_cell:
              this->assemble_tc_matrix<prm, c, DDFluxType::use_cell>(scratch,
                                                                     face);
              this->add_tc_matrix_terms_to_tt_rhs<prm, c, DDFluxType::use_cell>(
                scratch, task_data, face);
              this->assemble_tt_matrix<prm, c, DDFluxType::use_cell>(scratch,
                                                                     task_data,
                                                                     face);
              break;
            case DDFluxType::use_trace:
              this->assemble_tc_matrix<prm, c, DDFluxType::use_trace>(scratch,
                                                                      face);
              this
                ->add_tc_matrix_terms_to_tt_rhs<prm, c, DDFluxType::use_trace>(
                  scratch, task_data, face);
              this->assemble_tt_matrix<prm, c, DDFluxType::use_trace>(scratch,
                                                                      task_data,
                                                                      face);
              break;
            case DDFluxType::qiu_shi_stabilization:
              this
                ->assemble_tc_matrix<prm, c, DDFluxType::qiu_shi_stabilization>(
                  scratch, face);
              this->add_tc_matrix_terms_to_tt_rhs<
                prm,
                c,
                DDFluxType::qiu_shi_stabilization>(scratch, task_data, face);
              this
                ->assemble_tt_matrix<prm, c, DDFluxType::qiu_shi_stabilization>(
                  scratch, task_data, face);
              break;
            default:
              Assert(false, InvalidDDFluxType());
              break;
          }
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



  template <int dim, typename ProblemType>
  template <typename prm>
  void
  NPSolver<dim, ProblemType>::assemble_flux_conditions_wrapper(
    const Component          c,
    ScratchData &            scratch,
    PerTaskData &            task_data,
    const bool               has_dirichlet_conditions,
    const bool               has_neumann_conditions,
    const types::boundary_id face_boundary_id,
    const unsigned int       face)
  {
    switch (c)
      {
        case Component::V:
          assemble_flux_conditions<prm, Component::V>(scratch,
                                                      task_data,
                                                      has_dirichlet_conditions,
                                                      has_neumann_conditions,
                                                      face_boundary_id,
                                                      face);
          break;
        case Component::n:
          assemble_flux_conditions<prm, Component::n>(scratch,
                                                      task_data,
                                                      has_dirichlet_conditions,
                                                      has_neumann_conditions,
                                                      face_boundary_id,
                                                      face);
          break;
        case Component::p:
          assemble_flux_conditions<prm, Component::p>(scratch,
                                                      task_data,
                                                      has_dirichlet_conditions,
                                                      has_neumann_conditions,
                                                      face_boundary_id,
                                                      face);
          break;
        default:
          Assert(false, InvalidComponent());
          break;
      }
  }



  template <int dim, typename ProblemType>
  template <typename prm,
            bool has_boundary_conditions,
            typename TauComputerClass>
  void
  NPSolver<dim, ProblemType>::assemble_system_one_cell_internal(
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

        task_data.n_dirichlet_constrained_dofs = 0;

        // Fix also the values of the dof_indices vector
        cell->get_dof_indices(task_data.dof_indices);
      }

    // Set also that there are no constrained dofs (by the Dirichlet BC) on this
    // cell (we will increase that number later, if needed)
    scratch.local_condenser.n_cell_constrained_dofs = 0;

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

    // Now we must perform the L2 products on the boundary, i.e. for each face
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
            for (const auto c : Ddhdg::all_primary_components())
              {
                const FEValuesExtractors::Scalar extractor =
                  this->get_trace_component_extractor(c);
                scratch.fe_face_values_trace[extractor].get_function_values(
                  this->update_trace, scratch.tr_c_solution_values[c]);
              }
          }

        // Before assembling the other parts of the matrix, we need the values
        // of epsilon, mu and D_n on the quadrature points of the current
        // face. Moreover, we want to populate the scratch object with the
        // values of the solution at the previous step. All this jobs are
        // accomplished by the next function
        this->prepare_data_on_face_quadrature_points<TauComputerClass>(scratch,
                                                                       cell,
                                                                       face);

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
            switch (this->parameters->dd_flux_type)
              {
                case DDFluxType::use_cell:
                  this->assemble_ct_matrix<prm, DDFluxType::use_cell>(scratch,
                                                                      face);
                  break;
                case DDFluxType::use_trace:
                  this->assemble_ct_matrix<prm, DDFluxType::use_trace>(scratch,
                                                                       face);
                  break;
                case DDFluxType::qiu_shi_stabilization:
                  this->assemble_ct_matrix<prm,
                                           DDFluxType::qiu_shi_stabilization>(
                    scratch, face);
                  break;
                default:
                  Assert(false, InvalidComponent());
                  break;
              }
            if constexpr (!has_boundary_conditions)
              {
                if constexpr (prm::is_V_enabled)
                  {
                    switch (this->parameters->dd_flux_type)
                      {
                        case DDFluxType::use_cell:
                          this->assemble_tc_matrix<prm,
                                                   Component::V,
                                                   DDFluxType::use_cell>(
                            scratch, face);
                          this->add_tc_matrix_terms_to_tt_rhs<
                            prm,
                            Component::V,
                            DDFluxType::use_cell>(scratch, task_data, face);
                          this->assemble_tt_matrix<prm,
                                                   Component::V,
                                                   DDFluxType::use_cell>(
                            scratch, task_data, face);
                          break;
                        case DDFluxType::use_trace:
                          this->assemble_tc_matrix<prm,
                                                   Component::V,
                                                   DDFluxType::use_trace>(
                            scratch, face);
                          this->add_tc_matrix_terms_to_tt_rhs<
                            prm,
                            Component::V,
                            DDFluxType::use_trace>(scratch, task_data, face);
                          this->assemble_tt_matrix<prm,
                                                   Component::V,
                                                   DDFluxType::use_cell>(
                            scratch, task_data, face);
                          break;
                        case DDFluxType::qiu_shi_stabilization:
                          this->assemble_tc_matrix<
                            prm,
                            Component::V,
                            DDFluxType::qiu_shi_stabilization>(scratch, face);
                          this->add_tc_matrix_terms_to_tt_rhs<
                            prm,
                            Component::V,
                            DDFluxType::qiu_shi_stabilization>(scratch,
                                                               task_data,
                                                               face);
                          this->assemble_tt_matrix<
                            prm,
                            Component::V,
                            DDFluxType::qiu_shi_stabilization>(scratch,
                                                               task_data,
                                                               face);
                          break;
                        default:
                          Assert(false, InvalidComponent());
                          break;
                      }
                    this->add_tt_matrix_terms_to_tt_rhs<prm, Component::V>(
                      scratch, task_data, face);
                  }
                if constexpr (prm::is_n_enabled)
                  {
                    switch (this->parameters->dd_flux_type)
                      {
                        case DDFluxType::use_cell:
                          this->assemble_tc_matrix<prm,
                                                   Component::n,
                                                   DDFluxType::use_cell>(
                            scratch, face);
                          this->add_tc_matrix_terms_to_tt_rhs<
                            prm,
                            Component::n,
                            DDFluxType::use_cell>(scratch, task_data, face);
                          this->assemble_tt_matrix<prm,
                                                   Component::n,
                                                   DDFluxType::use_cell>(
                            scratch, task_data, face);
                          break;
                        case DDFluxType::use_trace:
                          this->assemble_tc_matrix<prm,
                                                   Component::n,
                                                   DDFluxType::use_trace>(
                            scratch, face);
                          this->add_tc_matrix_terms_to_tt_rhs<
                            prm,
                            Component::n,
                            DDFluxType::use_trace>(scratch, task_data, face);
                          this->assemble_tt_matrix<prm,
                                                   Component::n,
                                                   DDFluxType::use_trace>(
                            scratch, task_data, face);
                          break;
                        case DDFluxType::qiu_shi_stabilization:
                          this->assemble_tc_matrix<
                            prm,
                            Component::n,
                            DDFluxType::qiu_shi_stabilization>(scratch, face);
                          this->add_tc_matrix_terms_to_tt_rhs<
                            prm,
                            Component::n,
                            DDFluxType::qiu_shi_stabilization>(scratch,
                                                               task_data,
                                                               face);
                          this->assemble_tt_matrix<
                            prm,
                            Component::n,
                            DDFluxType::qiu_shi_stabilization>(scratch,
                                                               task_data,
                                                               face);
                          break;
                        default:
                          Assert(false, InvalidComponent());
                          break;
                      }
                    this->add_tt_matrix_terms_to_tt_rhs<prm, Component::n>(
                      scratch, task_data, face);
                  }
                if constexpr (prm::is_p_enabled)
                  {
                    switch (this->parameters->dd_flux_type)
                      {
                        case DDFluxType::use_cell:
                          this->assemble_tc_matrix<prm,
                                                   Component::p,
                                                   DDFluxType::use_cell>(
                            scratch, face);
                          this->add_tc_matrix_terms_to_tt_rhs<
                            prm,
                            Component::p,
                            DDFluxType::use_cell>(scratch, task_data, face);
                          this->assemble_tt_matrix<prm,
                                                   Component::p,
                                                   DDFluxType::use_cell>(
                            scratch, task_data, face);
                          break;
                        case DDFluxType::use_trace:
                          this->assemble_tc_matrix<prm,
                                                   Component::p,
                                                   DDFluxType::use_trace>(
                            scratch, face);
                          this->add_tc_matrix_terms_to_tt_rhs<
                            prm,
                            Component::p,
                            DDFluxType::use_trace>(scratch, task_data, face);
                          this->assemble_tt_matrix<prm,
                                                   Component::p,
                                                   DDFluxType::use_trace>(
                            scratch, task_data, face);
                          break;
                        case DDFluxType::qiu_shi_stabilization:
                          this->assemble_tc_matrix<
                            prm,
                            Component::p,
                            DDFluxType::qiu_shi_stabilization>(scratch, face);
                          this->add_tc_matrix_terms_to_tt_rhs<
                            prm,
                            Component::p,
                            DDFluxType::qiu_shi_stabilization>(scratch,
                                                               task_data,
                                                               face);
                          this->assemble_tt_matrix<
                            prm,
                            Component::p,
                            DDFluxType::qiu_shi_stabilization>(scratch,
                                                               task_data,
                                                               face);
                          break;
                        default:
                          Assert(false, InvalidComponent());
                          break;
                      }
                    this->add_tt_matrix_terms_to_tt_rhs<prm, Component::p>(
                      scratch, task_data, face);
                  }
              }
            else
              {
                const types::boundary_id face_boundary_id =
                  cell->face(face)->boundary_id();
                for (const Component c : this->enabled_components)
                  {
                    const bool has_dirichlet_bc =
                      this->problem->boundary_handler
                        ->has_dirichlet_boundary_conditions(face_boundary_id,
                                                            c);
                    const bool has_neumann_bc =
                      this->problem->boundary_handler
                        ->has_neumann_boundary_conditions(face_boundary_id, c);

                    this->assemble_flux_conditions_wrapper<prm>(
                      c,
                      scratch,
                      task_data,
                      has_dirichlet_bc,
                      has_neumann_bc,
                      face_boundary_id,
                      face);
                  }
              }
          }

        // These are the last terms of the ll matrix, the ones that are
        // generated by L2 products only on the boundary of the cell
        switch (this->parameters->dd_flux_type)
          {
            case DDFluxType::use_cell:
              this->add_border_products_to_cc_matrix<prm, DDFluxType::use_cell>(
                scratch, face);
              this->add_border_products_to_cc_rhs<prm, DDFluxType::use_cell>(
                scratch, face);
              break;
            case DDFluxType::use_trace:
              this
                ->add_border_products_to_cc_matrix<prm, DDFluxType::use_trace>(
                  scratch, face);
              this->add_border_products_to_cc_rhs<prm, DDFluxType::use_trace>(
                scratch, face);
              break;
            case DDFluxType::qiu_shi_stabilization:
              this->add_border_products_to_cc_matrix<
                prm,
                DDFluxType::qiu_shi_stabilization>(scratch, face);
              this->add_border_products_to_cc_rhs<
                prm,
                DDFluxType::qiu_shi_stabilization>(scratch, face);
              break;
            default:
              Assert(false, InvalidComponent());
              break;
          }

        if (task_data.trace_reconstruct)
          this->add_trace_terms_to_cc_rhs<prm>(scratch, face);
      }

    if constexpr (has_boundary_conditions)
      if (!task_data.trace_reconstruct)
        scratch.local_condenser.condense_ct_matrix(scratch.ct_matrix,
                                                   scratch.cc_rhs);

#ifdef WITH_MUTEX
    inversion_mutex.lock();
#endif

    scratch.cc_matrix.gauss_jordan();

#ifdef WITH_MUTEX
    inversion_mutex.unlock();
#endif

    if (!task_data.trace_reconstruct)
      {
        scratch.tc_matrix.mmult(scratch.tmp_matrix, scratch.cc_matrix);
        scratch.tmp_matrix.vmult_add(task_data.tt_vector, scratch.cc_rhs);
        scratch.tmp_matrix.mmult(task_data.tt_matrix, scratch.ct_matrix, true);

        // We need to copy the Dirichlet boundary conditions in the task object
        if constexpr (has_boundary_conditions)
          {
            const unsigned int n_constrained_dofs =
              scratch.local_condenser.n_cell_constrained_dofs;
            for (unsigned int i = 0; i < n_constrained_dofs; ++i)
              {
                const unsigned int dof_index =
                  scratch.local_condenser.cell_constrained_dofs[i];
                const dealii::types::global_dof_index global_dof_index =
                  task_data.dof_indices[dof_index];
                task_data.dirichlet_trace_dof_indices[i] = global_dof_index;
                task_data.dirichlet_trace_dof_values[i] =
                  scratch.local_condenser.constrained_dofs_values[i];
                task_data.n_dirichlet_constrained_dofs = n_constrained_dofs;
              }
          }
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


  template <int dim, typename ProblemType>
  template <typename prm, typename TauComputerClass>
  void
  NPSolver<dim, ProblemType>::assemble_system_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData &                                         scratch,
    PerTaskData &                                         task_data)
  {
    // First of all, we want to check if any boundary condition is set for this
    // cell
    bool has_boundary_conditions = false;
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
         ++face)
      {
        const types::boundary_id face_boundary_id =
          cell->face(face)->boundary_id();

        for (const auto c : this->enabled_components)
          {
            if (this->problem->boundary_handler
                  ->has_dirichlet_boundary_conditions(face_boundary_id, c))
              has_boundary_conditions = true;
            if (this->problem->boundary_handler
                  ->has_neumann_boundary_conditions(face_boundary_id, c))
              has_boundary_conditions = true;
            if (has_boundary_conditions)
              break;
          }
        if (has_boundary_conditions)
          break;
      }
    // Now we call the appropriate function. In this way, from now on when
    // we are far from the boundary, we do not have to care about boundary
    // conditions
    if (has_boundary_conditions)
      this->assemble_system_one_cell_internal<prm, true, TauComputerClass>(
        cell, scratch, task_data);
    else
      this->assemble_system_one_cell_internal<prm, false, TauComputerClass>(
        cell, scratch, task_data);
  }

} // namespace Ddhdg
