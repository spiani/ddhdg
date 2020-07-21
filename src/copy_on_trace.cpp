#include "deal.II/base/exceptions.h"
#include <deal.II/base/work_stream.h>

#include <deal.II/grid/tria_accessor.h>

#include "np_solver.h"

namespace Ddhdg
{
  namespace CopyTraceInternalTools
  {
    template <int dim>
    struct CTCopyData
    {
      explicit CTCopyData(unsigned int n_of_dofs);

      CTCopyData(const CTCopyData &other) = default;

      std::vector<bool>                      examined_faces;
      std::vector<std::vector<unsigned int>> dof_indices;
      std::vector<std::vector<double>>       dof_values;
    };

    template <int dim>
    CTCopyData<dim>::CTCopyData(unsigned int n_of_dofs)
      : examined_faces(dealii::GeometryInfo<dim>::faces_per_cell, false)
      , dof_indices(dealii::GeometryInfo<dim>::faces_per_cell,
                    std::vector<unsigned int>(n_of_dofs))
      , dof_values(dealii::GeometryInfo<dim>::faces_per_cell,
                   std::vector<double>(n_of_dofs))
    {}

    template <int dim, class Permittivity>
    struct CTScratchData
    {
      static std::map<Component, std::vector<std::vector<unsigned int>>>
      check_dofs_on_faces(
        const FiniteElement<dim> & fe,
        const std::set<Component> &components = all_components());

      static std::map<Component, unsigned int>
      count_dofs_per_component(
        const std::map<Component, std::vector<std::vector<unsigned int>>>
          &fe_support_on_face);

      template <class data_type>
      static std::map<Component, data_type>
      initialize_map_on_components(
        unsigned int               n,
        const std::set<Component> &components = all_components());

      template <class data_type>
      static std::map<Component, data_type>
      initialize_map_on_components(
        const std::map<Component, unsigned int> &dofs_per_component);

      CTScratchData(
        const dealii::FiniteElement<dim> &fe_cell,
        const dealii::FiniteElement<dim> &fe_trace,
        const std::set<Component> &       active_components,
        const std::map<Component, dealii::FEValuesExtractors::Scalar>
          &trace_extractors,
        const std::map<Component,
                       std::pair<dealii::FEValuesExtractors::Scalar,
                                 dealii::FEValuesExtractors::Vector>>
          &                            cell_extractors,
        const dealii::QGauss<dim - 1> &face_quadrature_formula,
        dealii::UpdateFlags            cell_flags,
        dealii::UpdateFlags            trace_flags,
        const Permittivity &           permittivity);

      CTScratchData(const CTScratchData<dim, Permittivity> &ct_scratch_data);

      unsigned int
      total_dofs_per_face();

      inline void
      clean_matrix();

      inline void
      clean_rhs();

      inline void
      assemble_matrix(unsigned int face_number);

      template <class CellIteratorType>
      inline void
      copy_data_for_cell_regular(
        const CellIteratorType &          cell,
        unsigned int                      face,
        const Problem<dim, Permittivity> &problem,
        const Adimensionalizer &          adimensionalizer,
        const dealii::Vector<double> &    current_solution);

      template <class CellIteratorType>
      inline void
      copy_data_for_cell_local_ref(
        const CellIteratorType &          cell,
        unsigned int                      face,
        unsigned int                      subface,
        const Problem<dim, Permittivity> &problem,
        const Adimensionalizer &          adimensionalizer,
        const dealii::Vector<double> &    current_solution);

      template <class CellIteratorType>
      inline void
      copy_data_for_cell(const CellIteratorType &          cell,
                         unsigned int                      face,
                         unsigned int                      subface,
                         const Problem<dim, Permittivity> &problem,
                         const Adimensionalizer &          adimensionalizer,
                         const dealii::Vector<double> &    current_solution);

      template <TraceProjectionStrategy strategy, bool regular_face>
      inline void
      assemble_rhs(unsigned int face_number);

      inline void
      assemble_rhs(unsigned int            face_number,
                   TraceProjectionStrategy strategy,
                   bool                    regular_face);

      inline void
      multiply_rhs(double k);

      inline void
      solve_system();

      inline void
      copy_solution(unsigned int face_number, CTCopyData<dim> &copy_data);

      dealii::FEFaceValues<dim>    fe_face_values_cell;
      dealii::FEFaceValues<dim>    fe_face_values_trace;
      dealii::FESubfaceValues<dim> fe_subface_values_cell;

      const std::set<Component> active_components;

      const std::map<Component, dealii::FEValuesExtractors::Scalar>
        trace_extractors;
      const std::map<Component,
                     std::pair<dealii::FEValuesExtractors::Scalar,
                               dealii::FEValuesExtractors::Vector>>
        cell_extractors;

      const std::map<Component, std::vector<std::vector<unsigned int>>>
        fe_trace_support_on_face;

      const std::map<Component, unsigned int> dofs_per_component_on_face;

      Permittivity                                     permittivity;
      std::vector<unsigned int>                        trace_global_dofs;
      std::vector<Point<dim>>                          quadrature_points;
      std::map<Component, std::vector<double>>         stabilized_tau;
      std::vector<Tensor<2, dim>>                      mu_n;
      std::vector<Tensor<2, dim>>                      mu_p;
      std::vector<Tensor<2, dim>>                      D_n;
      std::vector<Tensor<2, dim>>                      D_p;
      std::vector<double>                              T;
      std::vector<double>                              U_T;
      std::map<Component, std::vector<double>>         c;
      std::map<Component, std::vector<Tensor<1, dim>>> d;
      std::map<Component, std::vector<double>>         tr_c;

      std::map<Component, LAPACKFullMatrix<double>> matrix;
      std::map<Component, dealii::Vector<double>>   rhs;

      // This are just aliases to the previous attributes and are useful only to
      // make the functions written for the NPSolver::ScratchData work also for
      // this class
      std::vector<double> &        U_T_face;
      std::vector<double> &        U_T_cell;
      std::vector<Tensor<2, dim>> &mu_n_face;
      std::vector<Tensor<2, dim>> &mu_n_cell;
      std::vector<Tensor<2, dim>> &mu_p_face;
      std::vector<Tensor<2, dim>> &mu_p_cell;
    };

    template <int dim, class Permittivity>
    std::map<Component, std::vector<std::vector<unsigned int>>>
    CTScratchData<dim, Permittivity>::check_dofs_on_faces(
      const FiniteElement<dim> & fe,
      const std::set<Component> &components)
    {
      const unsigned int faces_per_cell = GeometryInfo<dim>::faces_per_cell;
      const unsigned int dofs_per_cell  = fe.dofs_per_cell;

      std::map<Component, std::vector<std::vector<unsigned int>>>
        fe_support_on_face;

      for (const auto c : components)
        {
          const unsigned int c_index = get_component_index(c);
          fe_support_on_face[c] =
            std::vector<std::vector<unsigned int>>(faces_per_cell);
          for (unsigned int face = 0; face < faces_per_cell; ++face)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int current_index =
                  fe.system_to_block_index(i).first;
                if (current_index == c_index && fe.has_support_on_face(i, face))
                  fe_support_on_face[c][face].push_back(i);
              }
        }
      return fe_support_on_face;
    }

    template <int dim, class Permittivity>
    std::map<Component, unsigned int>
    CTScratchData<dim, Permittivity>::count_dofs_per_component(
      const std::map<Component, std::vector<std::vector<unsigned int>>>
        &fe_support_on_face)
    {
      std::map<Component, unsigned int> dofs_per_component_on_face;
      for (const auto &[c, dofs_vector] : fe_support_on_face)
        dofs_per_component_on_face[c] = dofs_vector[0].size();
      return dofs_per_component_on_face;
    }

    template <int dim, class Permittivity>
    template <class data_type>
    std::map<Component, data_type>
    CTScratchData<dim, Permittivity>::initialize_map_on_components(
      unsigned int               n,
      const std::set<Component> &components)
    {
      std::map<Component, data_type> map_on_components;
      for (const auto c : components)
        {
          map_on_components[c] = data_type(n);
        }
      return map_on_components;
    }

    template <int dim, class Permittivity>
    template <class data_type>
    std::map<Component, data_type>
    CTScratchData<dim, Permittivity>::initialize_map_on_components(
      const std::map<Component, unsigned int> &dofs_per_component)
    {
      std::map<Component, data_type> map_on_components;
      for (const auto &[c, n_of_dofs] : dofs_per_component)
        map_on_components[c] = data_type(n_of_dofs);
      return map_on_components;
    }

    template <int dim, class Permittivity>
    CTScratchData<dim, Permittivity>::CTScratchData(
      const dealii::FiniteElement<dim> &fe_cell,
      const dealii::FiniteElement<dim> &fe_trace,
      const std::set<Component> &       active_components,
      const std::map<Component, dealii::FEValuesExtractors::Scalar>
        &trace_extractors,
      const std::map<Component,
                     std::pair<dealii::FEValuesExtractors::Scalar,
                               dealii::FEValuesExtractors::Vector>>
        &                            cell_extractors,
      const dealii::QGauss<dim - 1> &face_quadrature_formula,
      dealii::UpdateFlags            cell_flags,
      dealii::UpdateFlags            trace_flags,
      const Permittivity &           permittivity)
      : fe_face_values_cell(fe_cell, face_quadrature_formula, cell_flags)
      , fe_face_values_trace(fe_trace, face_quadrature_formula, trace_flags)
      , fe_subface_values_cell(fe_cell, face_quadrature_formula, cell_flags)
      , active_components(active_components)
      , trace_extractors(trace_extractors)
      , cell_extractors(cell_extractors)
      , fe_trace_support_on_face(
          check_dofs_on_faces(fe_trace, active_components))
      , dofs_per_component_on_face(
          count_dofs_per_component(fe_trace_support_on_face))
      , permittivity(permittivity)
      , trace_global_dofs(fe_trace.dofs_per_cell)
      , quadrature_points(face_quadrature_formula.size())
      , stabilized_tau(initialize_map_on_components<std::vector<double>>(
          face_quadrature_formula.size()))
      , mu_n(face_quadrature_formula.size())
      , mu_p(face_quadrature_formula.size())
      , D_n(face_quadrature_formula.size())
      , D_p(face_quadrature_formula.size())
      , T(face_quadrature_formula.size())
      , U_T(face_quadrature_formula.size())
      , c(initialize_map_on_components<std::vector<double>>(
          face_quadrature_formula.size()))
      , d(initialize_map_on_components<std::vector<Tensor<1, dim>>>(
          face_quadrature_formula.size()))
      , tr_c(initialize_map_on_components<std::vector<double>>(
          dofs_per_component_on_face))
      , matrix(initialize_map_on_components<LAPACKFullMatrix<double>>(
          dofs_per_component_on_face))
      , rhs(initialize_map_on_components<dealii::Vector<double>>(
          dofs_per_component_on_face))
      , U_T_face(U_T)
      , U_T_cell(U_T)
      , mu_n_face(mu_n)
      , mu_n_cell(mu_n)
      , mu_p_face(mu_p)
      , mu_p_cell(mu_p)
    {}

    template <int dim, class Permittivity>
    CTScratchData<dim, Permittivity>::CTScratchData(
      const CTScratchData<dim, Permittivity> &ct_scratch_data)
      : fe_face_values_cell(
          ct_scratch_data.fe_face_values_cell.get_fe(),
          ct_scratch_data.fe_face_values_cell.get_quadrature(),
          ct_scratch_data.fe_face_values_cell.get_update_flags())
      , fe_face_values_trace(
          ct_scratch_data.fe_face_values_trace.get_fe(),
          ct_scratch_data.fe_face_values_trace.get_quadrature(),
          ct_scratch_data.fe_face_values_trace.get_update_flags())
      , fe_subface_values_cell(
          ct_scratch_data.fe_subface_values_cell.get_fe(),
          ct_scratch_data.fe_subface_values_cell.get_quadrature(),
          ct_scratch_data.fe_subface_values_cell.get_update_flags())
      , active_components(ct_scratch_data.active_components)
      , trace_extractors(ct_scratch_data.trace_extractors)
      , cell_extractors(ct_scratch_data.cell_extractors)
      , fe_trace_support_on_face(ct_scratch_data.fe_trace_support_on_face)
      , dofs_per_component_on_face(ct_scratch_data.dofs_per_component_on_face)
      , permittivity(ct_scratch_data.permittivity)
      , trace_global_dofs(ct_scratch_data.trace_global_dofs)
      , quadrature_points(ct_scratch_data.quadrature_points)
      , stabilized_tau(ct_scratch_data.stabilized_tau)
      , mu_n(ct_scratch_data.mu_n)
      , mu_p(ct_scratch_data.mu_p)
      , D_n(ct_scratch_data.D_n)
      , D_p(ct_scratch_data.D_p)
      , T(ct_scratch_data.T)
      , U_T(ct_scratch_data.U_T)
      , c(ct_scratch_data.c)
      , d(ct_scratch_data.d)
      , tr_c(ct_scratch_data.tr_c)
      , matrix(ct_scratch_data.matrix)
      , rhs(ct_scratch_data.rhs)
      , U_T_face(U_T)
      , U_T_cell(U_T)
      , mu_n_face(mu_n)
      , mu_n_cell(mu_n)
      , mu_p_face(mu_p)
      , mu_p_cell(mu_p)
    {}

    template <int dim, class Permittivity>
    unsigned int
    CTScratchData<dim, Permittivity>::total_dofs_per_face()
    {
      unsigned int dofs_per_face = 0;
      for (const auto cmp : this->active_components)
        dofs_per_face += this->dofs_per_component_on_face.at(cmp);
      return dofs_per_face;
    }

    template <int dim, class Permittivity>
    void
    CTScratchData<dim, Permittivity>::clean_matrix()
    {
      for (const auto &element : this->matrix)
        {
          const auto component = element.first;
          auto &     c_matrix  = this->matrix.at(component);
          c_matrix             = 0.;
        }
    }

    template <int dim, class Permittivity>
    void
    CTScratchData<dim, Permittivity>::clean_rhs()
    {
      for (const auto &element : this->rhs)
        {
          const auto component = element.first;
          auto &     c_rhs     = this->rhs.at(component);
          c_rhs                = 0.;
        }
    }

    template <int dim, class Permittivity>
    void
    CTScratchData<dim, Permittivity>::assemble_matrix(
      const unsigned int face_number)
    {
      this->clean_matrix();

      const unsigned int q_points =
        this->fe_face_values_trace.get_quadrature().size();
      unsigned int dofs_per_face = 0;

      for (const auto cmp : this->active_components)
        {
          dofs_per_face = this->dofs_per_component_on_face.at(cmp);
          const auto &dofs_per_face_indices =
            this->fe_trace_support_on_face.at(cmp)[face_number];
          const auto c_extractor = this->trace_extractors.at(cmp);
          auto &     tr_cmp      = this->tr_c.at(cmp);
          auto &     c_matrix    = this->matrix.at(cmp);
          for (unsigned int q = 0; q < q_points; q++)
            {
              const double JxW = this->fe_face_values_trace.JxW(q);
              for (unsigned int i = 0; i < dofs_per_face; i++)
                {
                  const unsigned int ii = dofs_per_face_indices[i];
                  tr_cmp[i] =
                    this->fe_face_values_trace[c_extractor].value(ii, q);
                }
              for (unsigned int i = 0; i < dofs_per_face; i++)
                for (unsigned int j = 0; j < dofs_per_face; j++)
                  c_matrix(i, j) += tr_cmp[i] * tr_cmp[j] * JxW;
            }
        }
    }

    template <int dim, class Permittivity>
    template <class CellIteratorType>
    void
    CTScratchData<dim, Permittivity>::copy_data_for_cell_regular(
      const CellIteratorType &          cell,
      unsigned int                      face,
      const Problem<dim, Permittivity> &problem,
      const Adimensionalizer &          adimensionalizer,
      const dealii::Vector<double> &    current_solution)
    {
      this->fe_face_values_cell.reinit(cell, face);

      const unsigned int q_points =
        this->fe_face_values_cell.get_quadrature().size();

      // Read the position of the quadrature points
      for (unsigned int q = 0; q < q_points; ++q)
        this->quadrature_points[q] =
          this->fe_face_values_cell.quadrature_point(q);

      // Compute the permittivity
      this->permittivity.initialize_on_face(
        this->quadrature_points,
        adimensionalizer.get_permittivity_rescaling_factor());

      // Prepare the data of V and E on the cells
      const auto V_extractor = this->cell_extractors.at(Component::V).first;
      const auto E_extractor = this->cell_extractors.at(Component::V).second;
      this->fe_face_values_cell[V_extractor].get_function_values(
        current_solution, this->c.at(Component::V));
      this->fe_face_values_cell[E_extractor].get_function_values(
        current_solution, this->d.at(Component::V));

      if (this->active_components.find(Component::n) !=
          this->active_components.end())
        {
          // Prepare the values of mu_n
          problem.n_electron_mobility->compute_electron_mobility(
            this->quadrature_points, this->mu_n);
          adimensionalizer.template adimensionalize_electron_mobility<dim>(
            this->mu_n);

          // Prepare the values of n and Wn
          const auto n_extractor = this->cell_extractors.at(Component::n).first;
          const auto Wn_extractor =
            this->cell_extractors.at(Component::n).second;
          this->fe_face_values_cell[n_extractor].get_function_values(
            current_solution, this->c.at(Component::n));
          this->fe_face_values_cell[Wn_extractor].get_function_values(
            current_solution, this->d.at(Component::n));
        }

      if (this->active_components.find(Component::p) !=
          this->active_components.end())
        {
          problem.p_electron_mobility->compute_electron_mobility(
            this->quadrature_points, this->mu_p);
          adimensionalizer.template adimensionalize_electron_mobility<dim>(
            this->mu_p);

          // Prepare the values of p and Wp
          const auto p_extractor = this->cell_extractors.at(Component::p).first;
          const auto Wp_extractor =
            this->cell_extractors.at(Component::p).second;
          this->fe_face_values_cell[p_extractor].get_function_values(
            current_solution, this->c.at(Component::p));
          this->fe_face_values_cell[Wp_extractor].get_function_values(
            current_solution, this->d.at(Component::p));
        }

      problem.temperature->value_list(this->quadrature_points, this->T);

      const double thermal_voltage_rescaling_factor =
        adimensionalizer.get_thermal_voltage_rescaling_factor();
      for (unsigned int q = 0; q < q_points; ++q)
        this->U_T[q] = Constants::KB * this->T[q] /
                       (Constants::Q * thermal_voltage_rescaling_factor);
    }

    template <int dim, class Permittivity>
    template <class CellIteratorType>
    void
    CTScratchData<dim, Permittivity>::copy_data_for_cell_local_ref(
      const CellIteratorType &          cell,
      unsigned int                      face,
      unsigned int                      subface,
      const Problem<dim, Permittivity> &problem,
      const Adimensionalizer &          adimensionalizer,
      const dealii::Vector<double> &    current_solution)
    {
      this->fe_subface_values_cell.reinit(cell, face, subface);

      const unsigned int q_points =
        this->fe_subface_values_cell.get_quadrature().size();

      // Read the position of the quadrature points
      for (unsigned int q = 0; q < q_points; ++q)
        this->quadrature_points[q] =
          this->fe_subface_values_cell.quadrature_point(q);

      // Compute the permittivity
      this->permittivity.initialize_on_face(
        this->quadrature_points,
        adimensionalizer.get_permittivity_rescaling_factor());

      // Prepare the data of V and E on the cells
      const auto V_extractor = this->cell_extractors.at(Component::V).first;
      const auto E_extractor = this->cell_extractors.at(Component::V).second;
      this->fe_subface_values_cell[V_extractor].get_function_values(
        current_solution, this->c.at(Component::V));
      this->fe_subface_values_cell[E_extractor].get_function_values(
        current_solution, this->d.at(Component::V));

      if (this->active_components.find(Component::n) !=
          this->active_components.end())
        {
          // Prepare the values of mu_n
          problem.n_electron_mobility->compute_electron_mobility(
            this->quadrature_points, this->mu_n);
          adimensionalizer.template adimensionalize_electron_mobility<dim>(
            this->mu_n);

          // Prepare the values of n and Wn
          const auto n_extractor = this->cell_extractors.at(Component::n).first;
          const auto Wn_extractor =
            this->cell_extractors.at(Component::n).second;
          this->fe_subface_values_cell[n_extractor].get_function_values(
            current_solution, this->c.at(Component::n));
          this->fe_subface_values_cell[Wn_extractor].get_function_values(
            current_solution, this->d.at(Component::n));
        }

      if (this->active_components.find(Component::p) !=
          this->active_components.end())
        {
          problem.p_electron_mobility->compute_electron_mobility(
            this->quadrature_points, this->mu_p);
          adimensionalizer.template adimensionalize_electron_mobility<dim>(
            this->mu_p);

          // Prepare the values of p and Wp
          const auto p_extractor = this->cell_extractors.at(Component::p).first;
          const auto Wp_extractor =
            this->cell_extractors.at(Component::p).second;
          this->fe_subface_values_cell[p_extractor].get_function_values(
            current_solution, this->c.at(Component::p));
          this->fe_subface_values_cell[Wp_extractor].get_function_values(
            current_solution, this->d.at(Component::p));
        }

      problem.temperature->value_list(this->quadrature_points, this->T);

      const double thermal_voltage_rescaling_factor =
        adimensionalizer.get_thermal_voltage_rescaling_factor();
      for (unsigned int q = 0; q < q_points; ++q)
        this->U_T[q] = Constants::KB * this->T[q] /
                       (Constants::Q * thermal_voltage_rescaling_factor);
    }

    template <int dim, class Permittivity>
    template <class CellIteratorType>
    void
    CTScratchData<dim, Permittivity>::copy_data_for_cell(
      const CellIteratorType &          cell,
      unsigned int                      face,
      unsigned int                      subface,
      const Problem<dim, Permittivity> &problem,
      const Adimensionalizer &          adimensionalizer,
      const dealii::Vector<double> &    current_solution)
    {
      if (subface == dealii::numbers::invalid_unsigned_int)
        return this->copy_data_for_cell_regular(
          cell, face, problem, adimensionalizer, current_solution);
      return this->copy_data_for_cell_local_ref(
        cell, face, subface, problem, adimensionalizer, current_solution);
    }

    template <int dim, class Permittivity>
    template <TraceProjectionStrategy strategy, bool regular_face>
    void
    CTScratchData<dim, Permittivity>::assemble_rhs(
      const unsigned int face_number)
    {
      if (strategy != TraceProjectionStrategy::l2_average &&
          strategy != TraceProjectionStrategy::reconstruct_problem_solution)
        Assert(false, InvalidStrategy());

      const unsigned int q_points =
        this->fe_face_values_cell.get_quadrature().size();

      const double c_strategy = (strategy == l2_average) ? 0. : 1.;

      double         JxW;
      Tensor<1, dim> normal;

      // Rhs for the V equation
      if (this->active_components.find(Component::V) !=
          this->active_components.end())
        {
          const auto &V     = this->c.at(Component::V);
          const auto &E     = this->d.at(Component::V);
          auto &      tr_V  = this->tr_c.at(Component::V);
          const auto &tau   = this->stabilized_tau.at(Component::V);
          auto &      V_rhs = this->rhs.at(Component::V);

          dealii::Tensor<1, dim> epsilon_times_E;
          double                 epsilon_times_E_times_normal;

          double V_rhs_term;

          unsigned int dofs_per_face =
            this->dofs_per_component_on_face.at(Component::V);
          const auto &dofs_per_face_indices =
            this->fe_trace_support_on_face.at(Component::V)[face_number];
          const auto V_extractor = this->trace_extractors.at(Component::V);

          for (unsigned int q = 0; q < q_points; ++q)
            {
              if (regular_face)
                {
                  JxW    = this->fe_face_values_cell.JxW(q);
                  normal = this->fe_face_values_cell.normal_vector(q);
                }
              else
                {
                  JxW    = this->fe_subface_values_cell.JxW(q);
                  normal = this->fe_subface_values_cell.normal_vector(q);
                }

              this->permittivity.epsilon_operator_on_face(q,
                                                          E[q],
                                                          epsilon_times_E);
              epsilon_times_E_times_normal = epsilon_times_E * normal;

              V_rhs_term =
                0.5 *
                (V[q] + c_strategy * epsilon_times_E_times_normal / tau[q]) *
                JxW;

              // Copy the values of the test functions on the quadrature point
              for (unsigned int i = 0; i < dofs_per_face; i++)
                {
                  const unsigned int ii = dofs_per_face_indices[i];
                  tr_V[i] =
                    this->fe_face_values_trace[V_extractor].value(ii, q);
                }

              for (unsigned int i = 0; i < dofs_per_face; i++)
                V_rhs[i] += tr_V[i] * V_rhs_term;
            }
        }

      // Rhs for the n equation
      if (this->active_components.find(Component::n) !=
          this->active_components.end())
        {
          const auto &E     = this->d.at(Component::V);
          const auto &n     = this->c.at(Component::n);
          const auto &Wn    = this->d.at(Component::n);
          auto &      tr_n  = this->tr_c.at(Component::n);
          const auto &tau   = this->stabilized_tau.at(Component::n);
          auto &      n_rhs = this->rhs.at(Component::n);

          double n_rhs_term = 0.;

          double mu_n_times_E_times_normal  = 0.;
          double D_n_times_W_n_times_normal = 0.;

          unsigned int dofs_per_face =
            this->dofs_per_component_on_face.at(Component::n);
          const auto &dofs_per_face_indices =
            this->fe_trace_support_on_face.at(Component::n)[face_number];
          const auto n_extractor = this->trace_extractors.at(Component::n);

          for (unsigned int q = 0; q < q_points; ++q)
            {
              if (regular_face)
                {
                  JxW    = this->fe_face_values_cell.JxW(q);
                  normal = this->fe_face_values_cell.normal_vector(q);
                }
              else
                {
                  JxW    = this->fe_subface_values_cell.JxW(q);
                  normal = this->fe_subface_values_cell.normal_vector(q);
                }

              mu_n_times_E_times_normal  = this->mu_n[q] * E[q] * normal;
              D_n_times_W_n_times_normal = this->D_n[q] * Wn[q] * normal;

              n_rhs_term = (0.5 * c_strategy / tau[q] *
                              (-mu_n_times_E_times_normal * n[q] +
                               D_n_times_W_n_times_normal) +
                            0.5 * n[q]) *
                           JxW;

              // Copy the values of the test functions on the quadrature point
              for (unsigned int i = 0; i < dofs_per_face; i++)
                {
                  const unsigned int ii = dofs_per_face_indices[i];
                  tr_n[i] =
                    this->fe_face_values_trace[n_extractor].value(ii, q);
                }

              for (unsigned int i = 0; i < dofs_per_face; i++)
                n_rhs[i] += tr_n[i] * n_rhs_term;
            }
        }

      // Rhs for the p equation
      if (this->active_components.find(Component::p) !=
          this->active_components.end())
        {
          const auto &E     = this->d.at(Component::V);
          const auto &p     = this->c.at(Component::p);
          const auto &Wp    = this->d.at(Component::p);
          auto &      tr_p  = this->tr_c.at(Component::p);
          const auto &tau   = this->stabilized_tau.at(Component::p);
          auto &      p_rhs = this->rhs.at(Component::p);

          double p_rhs_term = 0.;

          double mu_p_times_E_times_normal  = 0.;
          double D_p_times_W_p_times_normal = 0.;

          unsigned int dofs_per_face =
            this->dofs_per_component_on_face.at(Component::p);
          const auto &dofs_per_face_indices =
            this->fe_trace_support_on_face.at(Component::p)[face_number];
          const auto p_extractor = this->trace_extractors.at(Component::p);

          for (unsigned int q = 0; q < q_points; ++q)
            {
              if (regular_face)
                {
                  JxW    = this->fe_face_values_cell.JxW(q);
                  normal = this->fe_face_values_cell.normal_vector(q);
                }
              else
                {
                  JxW    = this->fe_subface_values_cell.JxW(q);
                  normal = this->fe_subface_values_cell.normal_vector(q);
                }

              mu_p_times_E_times_normal  = this->mu_p[q] * E[q] * normal;
              D_p_times_W_p_times_normal = this->D_p[q] * Wp[q] * normal;

              p_rhs_term = (0.5 / tau[q] * c_strategy *
                              (-mu_p_times_E_times_normal * p[q] +
                               D_p_times_W_p_times_normal) +
                            0.5 * p[q]) *
                           JxW;

              // Copy the values of the test functions on the quadrature point
              for (unsigned int i = 0; i < dofs_per_face; i++)
                {
                  const unsigned int ii = dofs_per_face_indices[i];
                  tr_p[i] =
                    this->fe_face_values_trace[p_extractor].value(ii, q);
                }

              for (unsigned int i = 0; i < dofs_per_face; i++)
                p_rhs[i] += tr_p[i] * p_rhs_term;
            }
        }
    }

    template <int dim, class Permittivity>
    void
    CTScratchData<dim, Permittivity>::assemble_rhs(
      const unsigned int            face_number,
      const TraceProjectionStrategy strategy,
      const bool                    regular_face)
    {
      switch (strategy)
        {
          case l2_average:
            if (regular_face)
              this->template assemble_rhs<l2_average, true>(face_number);
            else
              this->template assemble_rhs<l2_average, false>(face_number);
            break;
          case reconstruct_problem_solution:
            if (regular_face)
              this->template assemble_rhs<reconstruct_problem_solution, true>(
                face_number);
            else
              this->template assemble_rhs<reconstruct_problem_solution, false>(
                face_number);
            break;
          default:
            Assert(false, InvalidStrategy());
        }
    }

    template <int dim, class Permittivity>
    void
    CTScratchData<dim, Permittivity>::multiply_rhs(const double k)
    {
      for (const auto cmp : this->active_components)
        {
          auto &c_rhs = this->rhs.at(cmp);
          for (unsigned int i = 0; i < c_rhs.size(); i++)
            c_rhs[i] *= k;
        }
    }

    template <int dim, class Permittivity>
    void
    CTScratchData<dim, Permittivity>::solve_system()
    {
      for (const auto cmp : this->active_components)
        {
          auto &c_matrix = this->matrix.at(cmp);
          auto &c_rhs    = this->rhs.at(cmp);

          c_matrix.compute_lu_factorization();
          c_matrix.solve(c_rhs);
        }
    }

    template <int dim, class Permittivity>
    void
    CTScratchData<dim, Permittivity>::copy_solution(
      const unsigned int face_number,
      CTCopyData<dim> &  copy_data)
    {
      Assert(copy_data.examined_faces[face_number] == false,
             ExcInternalError());
      copy_data.examined_faces[face_number] = true;
      unsigned int i                        = 0;
      for (const Component cmp : this->active_components)
        {
          const auto &dofs_indices =
            this->fe_trace_support_on_face.at(cmp)[face_number];
          const auto &       c_rhs     = this->rhs.at(cmp);
          const unsigned int n_of_dofs = dofs_indices.size();
          for (unsigned int j = 0; j < n_of_dofs; j++)
            {
              copy_data.dof_indices[face_number][i + j] =
                this->trace_global_dofs[dofs_indices[j]];
              copy_data.dof_values[face_number][i + j] = c_rhs[j];
            }
          i += n_of_dofs;
        }
    }

  } // namespace CopyTraceInternalTools



  template <int dim, class Permittivity>
  template <typename CTScratchData>
  void
  NPSolver<dim, Permittivity>::copy_trace_compute_tau(
    CTScratchData &scratch) const
  {
    for (const auto c : scratch.active_components)
      {
        const double tau_rescaled = this->adimensionalizer->adimensionalize_tau(
          this->parameters->tau.at(c), c);
        auto &stb_tau = scratch.stabilized_tau.at(c);

        const unsigned int n_q_points =
          scratch.fe_face_values_cell.get_quadrature().size();

        for (unsigned int q = 0; q < n_q_points; q++)
          stb_tau[q] = this->template compute_stabilized_tau(
            scratch,
            tau_rescaled,
            scratch.fe_face_values_cell.normal_vector(q),
            q,
            c);
      }
  }



  template <int dim, class Permittivity>
  template <typename CTScratchData>
  void
  NPSolver<dim, Permittivity>::copy_trace_compute_D_n_and_D_p(
    CTScratchData &scratch) const
  {
    for (unsigned int q = 0;
         q < scratch.fe_face_values_cell.get_quadrature().size();
         q++)
      {
        scratch.D_n[q] =
          this->template compute_einstein_diffusion_coefficient<Component::n>(
            scratch, q);
        scratch.D_p[q] =
          this->template compute_einstein_diffusion_coefficient<Component::p>(
            scratch, q);
      }
  }



  template <int dim, class Permittivity>
  template <typename IteratorType1,
            typename IteratorType2,
            typename CTScratchData,
            typename CTCopyData>
  void
  NPSolver<dim, Permittivity>::copy_trace_face_worker(
    const IteratorType1 &         cell1,
    const unsigned int            face1,
    const IteratorType2 &         cell2,
    const unsigned int            face2,
    CTScratchData &               scratch,
    CTCopyData &                  copy_data,
    const TraceProjectionStrategy strategy) const
  {
    // Initialize the fe_face_values for the trace; it will be
    // initialized just once per face, so it is enough to find one
    // direction for which the face is not a subface
    typename DoFHandler<dim>::active_cell_iterator trace_cell(
      &(*(this->triangulation)),
      cell1->level(),
      cell1->index(),
      &(this->dof_handler_trace));
    scratch.fe_face_values_trace.reinit(trace_cell, face1);
    trace_cell->get_dof_indices(scratch.trace_global_dofs);

    // Now we assemble the matrix
    scratch.assemble_matrix(face1);

    // Let us set to zero the rhs
    scratch.clean_rhs();

    // Now we load the scratch data with the data from cell 1
    scratch.template copy_data_for_cell<IteratorType1>(
      cell1,
      face1,
      dealii::numbers::invalid_unsigned_int,
      *(this->problem),
      *(this->adimensionalizer),
      this->get_solution_vector());

    this->copy_trace_compute_tau(scratch);
    this->copy_trace_compute_D_n_and_D_p(scratch);

    scratch.assemble_rhs(face1, strategy, true);

    // The same, but for face 2
    scratch.template copy_data_for_cell<IteratorType2>(
      cell2,
      face2,
      dealii::numbers::invalid_unsigned_int,
      *(this->problem),
      *(this->adimensionalizer),
      this->get_solution_vector());

    this->copy_trace_compute_tau(scratch);
    this->copy_trace_compute_D_n_and_D_p(scratch);

    // Here we have "face1" and not "face2" because, in this function,
    // the face is related with the FeValues for the trace, that has
    // been initialized on face1
    scratch.assemble_rhs(face1, strategy, true);

    scratch.solve_system();

    scratch.copy_solution(face1, copy_data);
  }



  template <int dim, class Permittivity>
  template <typename IteratorType, typename CTScratchData, typename CTCopyData>
  void
  NPSolver<dim, Permittivity>::copy_trace_boundary_worker(
    const IteratorType &          cell,
    const unsigned int            face,
    CTScratchData &               scratch,
    CTCopyData &                  copy_data,
    const TraceProjectionStrategy strategy) const
  {
    typename DoFHandler<dim>::active_cell_iterator trace_cell(
      &(*(this->triangulation)),
      cell->level(),
      cell->index(),
      &(this->dof_handler_trace));
    scratch.fe_face_values_trace.reinit(trace_cell, face);
    trace_cell->get_dof_indices(scratch.trace_global_dofs);

    // Now we assemble the matrix
    scratch.assemble_matrix(face);

    // Let us set to zero the rhs
    scratch.clean_rhs();

    scratch.copy_data_for_cell(cell,
                               face,
                               dealii::numbers::invalid_unsigned_int,
                               *(this->problem),
                               *(this->adimensionalizer),
                               this->get_solution_vector());

    this->copy_trace_compute_tau(scratch);
    this->copy_trace_compute_D_n_and_D_p(scratch);

    const unsigned int q_points =
      scratch.fe_face_values_trace.get_quadrature().size();

    switch (strategy)
      {
          case l2_average: {
            // This case is easy! Just assemble the residual like it was a
            // internal cell and multiply it by 2 (because it is like
            // there was another cell with the very same values attached
            // on the other side of this cell
            scratch.template assemble_rhs<l2_average, true>(face);
            scratch.multiply_rhs(2.);
            break;
          }
          case reconstruct_problem_solution: {
            // This, instead, is the not so easy case. Indeed, here we
            // have three cases: no boundary conditions (then we are in
            // the same case we were before), Dirichlet BC or Neumann BC
            // In any case, we build the standard residual. This is a
            // waste of times for the component that have Dirichlet
            // boundary conditions, but otherwise I would have to write
            // an ad-hoc function again
            scratch.template assemble_rhs<reconstruct_problem_solution, true>(
              face);
            scratch.multiply_rhs(2.);

            const types::boundary_id face_boundary_id =
              cell->face(face)->boundary_id();

            for (const Component cmp : scratch.active_components)
              {
                auto & c_rhs = scratch.rhs.at(cmp);
                double bc_value;

                unsigned int dofs_per_face =
                  scratch.dofs_per_component_on_face.at(cmp);
                const auto &dofs_per_face_indices =
                  scratch.fe_trace_support_on_face.at(cmp)[face];
                const auto c_extractor = scratch.trace_extractors.at(cmp);

                const bool has_dirichlet_bc =
                  this->problem->boundary_handler
                    ->has_dirichlet_boundary_conditions(face_boundary_id, cmp);
                if (has_dirichlet_bc)
                  {
                    const double rescaling_factor =
                      this->adimensionalizer->get_component_rescaling_factor(
                        cmp);

                    // Reset the previous values
                    c_rhs = 0.;

                    const auto dbc =
                      this->problem->boundary_handler
                        ->get_dirichlet_conditions_for_id(face_boundary_id,
                                                          cmp);

                    for (unsigned int q = 0; q < q_points; ++q)
                      {
                        const double JxW = scratch.fe_face_values_cell.JxW(q);

                        bc_value = dbc.evaluate(scratch.quadrature_points[q]) /
                                   rescaling_factor;

                        for (unsigned int i = 0; i < dofs_per_face; i++)
                          {
                            const unsigned int ii = dofs_per_face_indices[i];
                            c_rhs[i] +=
                              scratch.fe_face_values_trace[c_extractor].value(
                                ii, q) *
                              bc_value * JxW;
                          }
                      }
                    continue;
                  }

                const bool has_neumann_bc =
                  this->problem->boundary_handler
                    ->has_neumann_boundary_conditions(face_boundary_id, cmp);
                if (has_neumann_bc)
                  {
                    const double rescaling_factor =
                      this->adimensionalizer
                        ->get_neumann_boundary_condition_rescaling_factor(cmp);

                    const auto nbc =
                      this->problem->boundary_handler
                        ->get_neumann_conditions_for_id(face_boundary_id, cmp);
                    for (unsigned int q = 0; q < q_points; ++q)
                      {
                        const double JxW = scratch.fe_face_values_cell.JxW(q);

                        bc_value = nbc.evaluate(scratch.quadrature_points[q]) /
                                   rescaling_factor;

                        for (unsigned int i = 0; i < dofs_per_face; i++)
                          {
                            const unsigned int ii = dofs_per_face_indices[i];
                            c_rhs[i] +=
                              scratch.fe_face_values_trace[c_extractor].value(
                                ii, q) *
                              bc_value * JxW;
                          }
                      }
                    continue;
                  }
              }
            break;
          }
        default:
          Assert(false, InvalidStrategy());
      }

    scratch.solve_system();

    scratch.copy_solution(face, copy_data);
  }



  template <int dim, class Permittivity>
  template <typename CTCopyData>
  void
  NPSolver<dim, Permittivity>::copy_trace_copier(const CTCopyData &copy_data)
  {
    const unsigned int faces_per_cell =
      dealii::GeometryInfo<dim>::faces_per_cell;
    const unsigned int dofs_per_face = copy_data.dof_indices[0].size();
    for (unsigned int face = 0; face < faces_per_cell; face++)
      if (copy_data.examined_faces[face])
        for (unsigned int i = 0; i < dofs_per_face; i++)
          {
            this->current_solution_trace[copy_data.dof_indices[face][i]] =
              copy_data.dof_values[face][i];
          }
  }



  template <int dim, class Permittivity>
  template <typename IteratorType,
            typename CTScratchData,
            typename CTCopyData,
            TraceProjectionStrategy strategy>
  void
  NPSolver<dim, Permittivity>::copy_trace_cell_worker(const IteratorType &cell,
                                                      CTScratchData &scratch,
                                                      CTCopyData &   copy_data)
  {
    const unsigned int faces_per_cell =
      dealii::GeometryInfo<dim>::faces_per_cell;

    // Reset the copy_data
    for (unsigned int face = 0; face < faces_per_cell; face++)
      copy_data.examined_faces[face] = false;

    // We examine each face one by one
    for (unsigned int face = 0; face < faces_per_cell; face++)
      {
        // First of all, let us check if the face is on the boundary
        if (cell->face(face)->at_boundary())
          {
            // If this is the case, we are done! It is enough to call the
            // boundary worker
            this->copy_trace_boundary_worker(
              cell, face, scratch, copy_data, strategy);
            continue;
          }

        auto neighbor              = cell->neighbor(face);
        using NeighborIteratorType = decltype(neighbor);

        // If we are on a small cell, we continue; this face will be
        // handled when we will be on the coarser cell
        if (cell->neighbor_is_coarser(face))
          continue;

        const unsigned int neighbor_face = cell->neighbor_of_neighbor(face);

        // Now we are in the regular case (I do not care about the
        // anisotropic refinements). If dim == 1, we are always in the
        // regular case, because faces are points which can not be divided
        // in subfaces
        if (dim == 1 || !neighbor->has_children())
          {
            // This ensures that we perform this computation just once for
            // each face
            if (face < neighbor_face)
              this->copy_trace_face_worker<
                IteratorType,
                NeighborIteratorType,
                CopyTraceInternalTools::CTScratchData<dim, Permittivity>,
                CopyTraceInternalTools::CTCopyData<dim>>(cell,
                                                         face,
                                                         neighbor,
                                                         neighbor_face,
                                                         scratch,
                                                         copy_data,
                                                         strategy);
            continue;
          }

        this->copy_trace_face_worker<
          IteratorType,
          NeighborIteratorType,
          CopyTraceInternalTools::CTScratchData<dim, Permittivity>,
          CopyTraceInternalTools::CTCopyData<dim>>(
          cell, face, neighbor, neighbor_face, scratch, copy_data, strategy);
      }
  }



  template <int dim, class Permittivity>
  void
  NPSolver<dim, Permittivity>::project_cell_function_on_trace(
    const std::set<Component> &components,
    TraceProjectionStrategy    strategy)
  {
    using ActiveCellIteratorType =
      decltype(this->dof_handler_cell.begin_active());

    const QGauss<dim - 1> face_quadrature_formula(
      this->get_number_of_quadrature_points());
    const UpdateFlags flags_cell(update_values | update_quadrature_points |
                                 update_normal_vectors | update_JxW_values);
    const UpdateFlags flags_trace(update_values | update_JxW_values);

    // Associate to each component its extractor (for the trace and for the
    // cells)
    std::map<Component, dealii::FEValuesExtractors::Scalar> trace_extractors;
    std::map<Component,
             std::pair<dealii::FEValuesExtractors::Scalar,
                       dealii::FEValuesExtractors::Vector>>
      cell_extractors;
    for (const auto c : all_components())
      {
        const auto d        = component2displacement(c);
        trace_extractors[c] = this->get_trace_component_extractor(c);
        cell_extractors[c]  = {this->get_component_extractor(c),
                              this->get_displacement_extractor(d)};
      }

    CopyTraceInternalTools::CTScratchData<dim, Permittivity> scratch(
      *(this->fe_cell),
      *(this->fe_trace),
      components,
      trace_extractors,
      cell_extractors,
      face_quadrature_formula,
      flags_cell,
      flags_trace,
      *(this->problem->permittivity));

    CopyTraceInternalTools::CTCopyData<dim> copy_data(
      scratch.total_dofs_per_face());

    typedef void (NPSolver<dim, Permittivity>::*copy_trace_worker_pointer_type)(
      const ActiveCellIteratorType &                            cell,
      CopyTraceInternalTools::CTScratchData<dim, Permittivity> &scratch,
      CopyTraceInternalTools::CTCopyData<dim> &                 task_data);

    copy_trace_worker_pointer_type copy_trace_worker_pointer;

    // Get the right pointer to the copy_trace_cell_worker function
    switch (strategy)
      {
        case TraceProjectionStrategy::l2_average:
          copy_trace_worker_pointer =
            &NPSolver<dim, Permittivity>::template copy_trace_cell_worker<
              ActiveCellIteratorType,
              CopyTraceInternalTools::template CTScratchData<dim, Permittivity>,
              CopyTraceInternalTools::template CTCopyData<dim>,
              TraceProjectionStrategy::l2_average>;
          break;
        case TraceProjectionStrategy::reconstruct_problem_solution:
          copy_trace_worker_pointer =
            &NPSolver<dim, Permittivity>::template copy_trace_cell_worker<
              ActiveCellIteratorType,
              CopyTraceInternalTools::template CTScratchData<dim, Permittivity>,
              CopyTraceInternalTools::template CTCopyData<dim>,
              TraceProjectionStrategy::reconstruct_problem_solution>;
          break;
        default:
          Assert(false, InvalidStrategy());
      }

    WorkStream::run(this->dof_handler_cell.begin_active(),
                    this->dof_handler_cell.end(),
                    *this,
                    copy_trace_worker_pointer,
                    &NPSolver<dim, Permittivity>::template copy_trace_copier<
                      CopyTraceInternalTools::CTCopyData<dim>>,
                    scratch,
                    copy_data);

    this->global_constraints.distribute(this->current_solution_trace);
  }

  template class NPSolver<1, HomogeneousPermittivity<1>>;
  template class NPSolver<2, HomogeneousPermittivity<2>>;
  template class NPSolver<3, HomogeneousPermittivity<3>>;
} // namespace Ddhdg