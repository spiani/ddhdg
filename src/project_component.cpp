#include "deal.II/base/exceptions.h"

#include "np_solver.h"

namespace Ddhdg
{
  namespace ProjectComponentInternalTools
  {
    struct PCCopyData
    {
      explicit PCCopyData(unsigned int dofs_per_cell);

      PCCopyData(const PCCopyData &other) = default;

      std::vector<unsigned int> dof_indices;
      std::vector<double>       dof_values;
    };

    PCCopyData::PCCopyData(const unsigned int dofs_per_cell)
      : dof_indices(dofs_per_cell)
      , dof_values(dofs_per_cell)
    {}

    template <int dim>
    struct PCScratchData
    {
      template <bool for_displacement = false>
      static std::vector<unsigned int>
      check_dofs_on_current_component(const FiniteElement<dim> &   fe,
                                      unsigned int                 c_index,
                                      const dealii::ComponentMask &c_mask);

      static std::vector<std::vector<unsigned int>>
      check_displacement_dofs_on_face(
        const FiniteElement<dim> &       fe,
        const std::vector<unsigned int> &displacement_dofs_indices);

      PCScratchData(std::shared_ptr<const dealii::Function<dim>> c_function,
                    const dealii::FiniteElement<dim> &           fe_cell,
                    unsigned int                                 c_index,
                    const dealii::ComponentMask &                c_mask,
                    const FEValuesExtractors::Scalar &           c_extractor,
                    const FEValuesExtractors::Vector &           d_extractor,
                    const dealii::QGauss<dim> &    quadrature_formula,
                    const dealii::QGauss<dim - 1> &face_quadrature_formula,
                    dealii::UpdateFlags            fe_values_flags,
                    dealii::UpdateFlags            fe_face_values_flags);

      PCScratchData(const PCScratchData<dim> &pc_scratch_data);

      inline void
      save_data_for_quasi_fermi_potentials(
        double,
        double,
        const dealii::FEValuesExtractors::Scalar &)
      {}

      const std::shared_ptr<const dealii::Function<dim>> c_function;

      dealii::FEValues<dim>     fe_values;
      dealii::FEFaceValues<dim> fe_face_values;

      const std::vector<unsigned int> on_current_component;
      const std::vector<unsigned int> on_current_displacement;

      const unsigned int dofs_per_component;
      const unsigned int dofs_per_displacement;

      const std::vector<std::vector<unsigned int>> on_current_face;

      const FEValuesExtractors::Scalar c_extractor;
      const FEValuesExtractors::Vector d_extractor;

      dealii::LAPACKFullMatrix<double> component_matrix;
      dealii::Vector<double>           component_residual;
      dealii::LAPACKFullMatrix<double> displacement_matrix;
      dealii::Vector<double>           displacement_residual;

      std::vector<dealii::types::global_dof_index> dof_indices;

      // Temporary buffer for the values of the local base function on a
      // quadrature point
      std::vector<double>         c_bf;
      std::vector<Tensor<1, dim>> d_bf;
      std::vector<double>         d_div_bf;

      std::vector<Point<dim>> cell_quadrature_points;
      std::vector<Point<dim>> face_quadrature_points;

      std::vector<double> evaluated_c;
      std::vector<double> evaluated_c_face;
    };

    template <int dim>
    template <bool for_displacement>
    std::vector<unsigned int>
    PCScratchData<dim>::check_dofs_on_current_component(
      const FiniteElement<dim> &   fe,
      const unsigned int           c_index,
      const dealii::ComponentMask &c_mask)
    {
      std::vector<unsigned int> current_component_dofs;
      for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
        {
          const auto &       local_fe         = fe.get_sub_fe(c_mask);
          const unsigned int l_c_index        = (for_displacement) ? 0 : 1;
          const auto         local_block_info = fe.system_to_block_index(i);
          const unsigned int local_block      = local_block_info.first;
          const unsigned int local_index      = local_block_info.second;
          if (local_block == c_index)
            if (local_fe.system_to_block_index(local_index).first == l_c_index)
              current_component_dofs.push_back(i);
        }
      return current_component_dofs;
    }

    template <int dim>
    std::vector<std::vector<unsigned int>>
    PCScratchData<dim>::check_displacement_dofs_on_face(
      const FiniteElement<dim> &       fe,
      const std::vector<unsigned int> &displacement_dofs_indices)
    {
      std::vector<std::vector<unsigned int>> component_support_on_face(
        GeometryInfo<dim>::faces_per_cell);

      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        for (unsigned int i = 0; i < displacement_dofs_indices.size(); ++i)
          {
            const unsigned int dof_index = displacement_dofs_indices[i];
            if (fe.has_support_on_face(dof_index, face))
              component_support_on_face[face].push_back(i);
          }
      return component_support_on_face;
    }

    template <int dim>
    PCScratchData<dim>::PCScratchData(
      const std::shared_ptr<const dealii::Function<dim>> c_function,
      const dealii::FiniteElement<dim> &                 fe_cell,
      const unsigned int                                 c_index,
      const dealii::ComponentMask &                      c_mask,
      const FEValuesExtractors::Scalar &                 c_extractor,
      const FEValuesExtractors::Vector &                 d_extractor,
      const dealii::QGauss<dim> &                        quadrature_formula,
      const dealii::QGauss<dim - 1> &face_quadrature_formula,
      dealii::UpdateFlags            fe_values_flags,
      dealii::UpdateFlags            fe_face_values_flags)
      : c_function(c_function)
      , fe_values(fe_cell, quadrature_formula, fe_values_flags)
      , fe_face_values(fe_cell, face_quadrature_formula, fe_face_values_flags)
      , on_current_component(
          check_dofs_on_current_component<false>(fe_cell, c_index, c_mask))
      , on_current_displacement(
          check_dofs_on_current_component<true>(fe_cell, c_index, c_mask))
      , dofs_per_component(on_current_component.size())
      , dofs_per_displacement(on_current_displacement.size())
      , on_current_face(
          check_displacement_dofs_on_face(fe_cell, on_current_displacement))
      , c_extractor(c_extractor)
      , d_extractor(d_extractor)
      , component_matrix(dofs_per_component, dofs_per_component)
      , component_residual(dofs_per_component)
      , displacement_matrix(dofs_per_displacement, dofs_per_displacement)
      , displacement_residual(dofs_per_displacement)
      , dof_indices(fe_cell.dofs_per_cell)
      , c_bf(dofs_per_component)
      , d_bf(dofs_per_displacement)
      , d_div_bf(dofs_per_displacement)
      , cell_quadrature_points(quadrature_formula.size())
      , face_quadrature_points(face_quadrature_formula.size())
      , evaluated_c(quadrature_formula.size())
      , evaluated_c_face(face_quadrature_formula.size())
    {}

    template <int dim>
    PCScratchData<dim>::PCScratchData(const PCScratchData<dim> &pc_scratch_data)
      : c_function(pc_scratch_data.c_function)
      , fe_values(pc_scratch_data.fe_values.get_fe(),
                  pc_scratch_data.fe_values.get_quadrature(),
                  pc_scratch_data.fe_values.get_update_flags())
      , fe_face_values(pc_scratch_data.fe_face_values.get_fe(),
                       pc_scratch_data.fe_face_values.get_quadrature(),
                       pc_scratch_data.fe_face_values.get_update_flags())
      , on_current_component(pc_scratch_data.on_current_component)
      , on_current_displacement(pc_scratch_data.on_current_displacement)
      , dofs_per_component(pc_scratch_data.dofs_per_component)
      , dofs_per_displacement(pc_scratch_data.dofs_per_displacement)
      , on_current_face(pc_scratch_data.on_current_face)
      , c_extractor(pc_scratch_data.c_extractor)
      , d_extractor(pc_scratch_data.d_extractor)
      , component_matrix(dofs_per_component, dofs_per_component)
      , component_residual(dofs_per_component)
      , displacement_matrix(dofs_per_displacement, dofs_per_displacement)
      , displacement_residual(dofs_per_displacement)
      , dof_indices(pc_scratch_data.dof_indices.size())
      , c_bf(pc_scratch_data.c_bf.size())
      , d_bf(pc_scratch_data.d_bf.size())
      , d_div_bf(pc_scratch_data.d_div_bf.size())
      , cell_quadrature_points(pc_scratch_data.cell_quadrature_points.size())
      , face_quadrature_points(pc_scratch_data.face_quadrature_points.size())
      , evaluated_c(pc_scratch_data.evaluated_c.size())
      , evaluated_c_face(pc_scratch_data.evaluated_c_face.size())
    {}

    template <int dim>
    struct PCQuasiFermiPotentialScratchData : public PCScratchData<dim>
    {
      PCQuasiFermiPotentialScratchData(
        std::shared_ptr<const dealii::Function<dim>> c_function,
        const dealii::FiniteElement<dim> &           fe_cell,
        unsigned int                                 c_index,
        const dealii::ComponentMask &                c_mask,
        const FEValuesExtractors::Scalar &           c_extractor,
        const FEValuesExtractors::Vector &           d_extractor,
        const dealii::QGauss<dim> &                  quadrature_formula,
        const dealii::QGauss<dim - 1> &              face_quadrature_formula,
        dealii::UpdateFlags                          fe_values_flags,
        dealii::UpdateFlags                          fe_face_values_flags);

      // PCQuasiFermiPotentialScratchData(
      //  const PCQuasiFermiPotentialScratchData<dim> &pc_qfp_scratch_data);

      inline void
      save_data_for_quasi_fermi_potentials(
        double                                    V_rescale_,
        double                                    c_rescale_,
        const dealii::FEValuesExtractors::Scalar &V_extractor_);

      std::vector<double> V_values_cell;
      std::vector<double> temperature_cell;
      std::vector<double> V_values_face;
      std::vector<double> temperature_face;

      double V_rescale = 0.;
      double c_rescale = 0.;

      dealii::FEValuesExtractors::Scalar V_extractor;
    };



    template <int dim>
    PCQuasiFermiPotentialScratchData<dim>::PCQuasiFermiPotentialScratchData(
      const std::shared_ptr<const dealii::Function<dim>> c_function,
      const dealii::FiniteElement<dim> &                 fe_cell,
      const unsigned int                                 c_index,
      const dealii::ComponentMask &                      c_mask,
      const FEValuesExtractors::Scalar &                 c_extractor,
      const FEValuesExtractors::Vector &                 d_extractor,
      const dealii::QGauss<dim> &                        quadrature_formula,
      const dealii::QGauss<dim - 1> &face_quadrature_formula,
      dealii::UpdateFlags            fe_values_flags,
      dealii::UpdateFlags            fe_face_values_flags)
      : PCScratchData<dim>(c_function,
                           fe_cell,
                           c_index,
                           c_mask,
                           c_extractor,
                           d_extractor,
                           quadrature_formula,
                           face_quadrature_formula,
                           fe_values_flags,
                           fe_face_values_flags)
      , V_values_cell(quadrature_formula.size())
      , temperature_cell(quadrature_formula.size())
      , V_values_face(face_quadrature_formula.size())
      , temperature_face(face_quadrature_formula.size())
    {}



    template <int dim>
    void
    PCQuasiFermiPotentialScratchData<dim>::save_data_for_quasi_fermi_potentials(
      double                                    V_rescale_,
      double                                    c_rescale_,
      const dealii::FEValuesExtractors::Scalar &V_extractor_)
    {
      this->V_rescale   = V_rescale_;
      this->c_rescale   = c_rescale_;
      this->V_extractor = V_extractor_;
    }

  } // namespace ProjectComponentInternalTools



  template <int dim, class Permittivity>
  template <typename PCScratchData, typename PCCopyData, Component c>
  void
  NPSolver<dim, Permittivity>::project_component_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    PCScratchData &                                       scratch,
    PCCopyData &                                          copy_data) const
  {
    constexpr bool c_is_primal =
      c == Component::V || c == Component::n || c == Component::p;

    constexpr Component primal_c =
      (c_is_primal) ? c :
                      ((c == Component::phi_n) ? Component::n : Component::p);
    (void)primal_c;

    const unsigned int dofs_per_component    = scratch.dofs_per_component;
    const unsigned int dofs_per_displacement = scratch.dofs_per_displacement;
    const unsigned int dofs_per_face = scratch.on_current_face[0].size();
    const unsigned int faces_per_cell =
      dealii::GeometryInfo<dim>::faces_per_cell;

    const unsigned int n_q_points = scratch.fe_values.get_quadrature().size();
    const unsigned int n_face_q_points =
      scratch.fe_face_values.get_quadrature().size();

    scratch.component_matrix      = 0.;
    scratch.displacement_matrix   = 0.;
    scratch.component_residual    = 0.;
    scratch.displacement_residual = 0.;

    scratch.fe_values.reinit(cell);
    cell->get_dof_indices(scratch.dof_indices);

    // Get the position of the quadrature points
    for (unsigned int q = 0; q < n_q_points; ++q)
      scratch.cell_quadrature_points[q] = scratch.fe_values.quadrature_point(q);

    if constexpr (!c_is_primal)
      {
        // If we are computing n (or p) knowing only phi_n or phi_p, we need
        // this data to perform the conversion
        this->problem->temperature->value_list(scratch.cell_quadrature_points,
                                               scratch.temperature_cell);
        scratch.fe_values[scratch.V_extractor].get_function_values(
          this->current_solution_cell, scratch.V_values_cell);
        for (unsigned int q = 0; q < n_q_points; q++)
          scratch.V_values_cell[q] *= scratch.V_rescale;
      }

    // Compute the values of the analytic functions at the quadrature points
    scratch.c_function->value_list(scratch.cell_quadrature_points,
                                   scratch.evaluated_c);

    if constexpr (!c_is_primal)
      {
        for (unsigned int q = 0; q < n_q_points; q++)
          scratch.evaluated_c[q] = this->template compute_density<primal_c>(
                                     scratch.evaluated_c[q],
                                     scratch.V_values_cell[q],
                                     scratch.temperature_cell[q]) /
                                   scratch.c_rescale;
      }

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        // Copy data of the shape function
        for (unsigned int k = 0; k < dofs_per_component; ++k)
          {
            const unsigned int i = scratch.on_current_component[k];
            scratch.c_bf[k] =
              scratch.fe_values[scratch.c_extractor].value(i, q);
          }
        for (unsigned int k = 0; k < dofs_per_displacement; ++k)
          {
            const unsigned int i = scratch.on_current_displacement[k];
            scratch.d_bf[k] =
              scratch.fe_values[scratch.d_extractor].value(i, q);
            scratch.d_div_bf[k] =
              scratch.fe_values[scratch.d_extractor].divergence(i, q);
          }

        const double JxW = scratch.fe_values.JxW(q);

        for (unsigned int i = 0; i < dofs_per_component; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_component; ++j)
              {
                scratch.component_matrix(i, j) +=
                  scratch.c_bf[j] * scratch.c_bf[i] * JxW;
              }
            scratch.component_residual[i] +=
              scratch.evaluated_c[q] * scratch.c_bf[i] * JxW;
          }

        for (unsigned int i = 0; i < dofs_per_displacement; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_displacement; ++j)
              {
                scratch.displacement_matrix(i, j) +=
                  scratch.d_bf[i] * scratch.d_bf[j] * JxW;
              }
            scratch.displacement_residual[i] +=
              scratch.evaluated_c[q] * scratch.d_div_bf[i] * JxW;
          }
      }

    // The matrices are OK, and so is the residual for the component!
    // The residual for the displacement, instead, needs some terms
    // computed on the faces
    for (unsigned int face = 0; face < faces_per_cell; ++face)
      {
        scratch.fe_face_values.reinit(cell, face);

        for (unsigned int q = 0; q < n_face_q_points; ++q)
          scratch.face_quadrature_points[q] =
            scratch.fe_face_values.quadrature_point(q);

        if constexpr (!c_is_primal)
          {
            this->problem->temperature->value_list(
              scratch.face_quadrature_points, scratch.temperature_face);
            scratch.fe_face_values[scratch.V_extractor].get_function_values(
              this->current_solution_cell, scratch.V_values_face);
            for (unsigned int q = 0; q < n_q_points; q++)
              scratch.V_values_face[q] *= scratch.V_rescale;
          }

        scratch.c_function->value_list(scratch.face_quadrature_points,
                                       scratch.evaluated_c_face);

        if constexpr (!c_is_primal)
          {
            for (unsigned int q = 0; q < n_face_q_points; q++)
              scratch.evaluated_c_face[q] =
                this->template compute_density<primal_c>(
                  scratch.evaluated_c[q],
                  scratch.V_values_face[q],
                  scratch.temperature_face[q]) /
                scratch.c_rescale;
          }

        for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            const double JxW    = scratch.fe_face_values.JxW(q);
            const auto   normal = scratch.fe_face_values.normal_vector(q);

            for (unsigned int k = 0; k < dofs_per_face; ++k)
              {
                const auto kk  = scratch.on_current_face[face][k];
                const auto kkk = scratch.on_current_displacement[kk];
                const auto f_bf_face =
                  scratch.fe_face_values[scratch.d_extractor].value(kkk, q);
                scratch.displacement_residual[kk] +=
                  (-scratch.evaluated_c_face[q] * (f_bf_face * normal)) * JxW;
              }
          }
      }

    scratch.component_matrix.compute_lu_factorization();
    scratch.displacement_matrix.compute_lu_factorization();
    scratch.component_matrix.solve(scratch.component_residual);
    scratch.displacement_matrix.solve(scratch.displacement_residual);

    // Now we copy the results inside the copy_data object
    for (unsigned int i = 0; i < dofs_per_component; ++i)
      {
        const auto local_dof_index  = scratch.on_current_component[i];
        const auto global_dof_index = scratch.dof_indices[local_dof_index];

        copy_data.dof_indices[i] = global_dof_index;
        copy_data.dof_values[i]  = scratch.component_residual[i];
      }
    for (unsigned int i = 0; i < dofs_per_displacement; ++i)
      {
        const auto local_dof_index  = scratch.on_current_displacement[i];
        const auto global_dof_index = scratch.dof_indices[local_dof_index];

        copy_data.dof_indices[dofs_per_component + i] = global_dof_index;
        copy_data.dof_values[dofs_per_component + i] =
          scratch.displacement_residual[i];
      }
  }



  template <int dim, class Permittivity>
  template <typename PCCopyData>
  void
  NPSolver<dim, Permittivity>::project_component_copier(PCCopyData &copy_data)
  {
    for (unsigned int i = 0; i < copy_data.dof_indices.size(); ++i)
      this->current_solution_cell[copy_data.dof_indices[i]] =
        copy_data.dof_values[i];
  }



  template <int dim, class Permittivity>
  template <Component c>
  void
  NPSolver<dim, Permittivity>::project_component_private(
    const std::shared_ptr<const dealii::Function<dim>> c_function)
  {
    Assert(c == Component::V || c == Component::n || c == Component::p ||
             c == Component::phi_n || c == Component::phi_p,
           InvalidComponent());

    constexpr bool c_is_primal =
      c == Component::V || c == Component::n || c == Component::p;

    typedef ProjectComponentInternalTools::PCScratchData<dim> PrimalScratchData;
    typedef ProjectComponentInternalTools::PCQuasiFermiPotentialScratchData<dim>
      NotPrimalScratchData;

    typedef typename std::conditional<c_is_primal,
                                      PrimalScratchData,
                                      NotPrimalScratchData>::type ScratchData;

    typedef ProjectComponentInternalTools::PCCopyData CopyData;

    if (!this->initialized)
      this->setup_overall_system();

    const unsigned int n_q_per_side = this->get_number_of_quadrature_points();

    const QGauss<dim>     quadrature_formula(n_q_per_side);
    const QGauss<dim - 1> face_quadrature_formula(n_q_per_side);

    const UpdateFlags fe_values_flags(update_values | update_gradients |
                                      update_JxW_values |
                                      update_quadrature_points);
    const UpdateFlags fe_face_values_flags(
      update_values | update_normal_vectors | update_quadrature_points |
      update_JxW_values);

    constexpr Component primal_c =
      (c_is_primal) ? c :
                      ((c == Component::phi_n) ? Component::n : Component::p);


    const unsigned int c_index = get_component_index(primal_c);

    constexpr Displacement      primal_d   = component2displacement(primal_c);
    const dealii::ComponentMask c_mask     = this->get_component_mask(primal_c);
    const dealii::ComponentMask d_mask     = this->get_component_mask(primal_d);
    const dealii::ComponentMask total_mask = c_mask | d_mask;

    const FEValuesExtractors::Vector d_extractor =
      this->get_displacement_extractor(primal_d);
    const FEValuesExtractors::Scalar c_extractor =
      this->get_component_extractor(primal_c);

    std::shared_ptr<const dealii::Function<dim>> c_function_rescaled;
    if (c_is_primal)
      c_function_rescaled =
        this->adimensionalizer
          ->template adimensionalize_component_function<dim>(c_function, c);
    else
      c_function_rescaled = c_function;

    ScratchData scratch((c_is_primal) ? c_function_rescaled : c_function,
                        *(this->fe_cell),
                        c_index,
                        total_mask,
                        c_extractor,
                        d_extractor,
                        quadrature_formula,
                        face_quadrature_formula,
                        fe_values_flags,
                        fe_face_values_flags);

    if (!c_is_primal)
      scratch.save_data_for_quasi_fermi_potentials(
        this->adimensionalizer
          ->template get_component_rescaling_factor<Component::V>(),
        this->adimensionalizer
          ->template get_component_rescaling_factor<primal_c>(),
        this->get_component_extractor(Component::V));

    CopyData copy_data(scratch.dofs_per_component +
                       scratch.dofs_per_displacement);

    for (const auto &cell : this->dof_handler_cell.active_cell_iterators())
      {
        this->project_component_one_cell<ScratchData, CopyData, c>(cell,
                                                                   scratch,
                                                                   copy_data);
        this->project_component_copier(copy_data);
      }

    // Now we need to copy the trace from the values on the cells
    std::set<Component> current_component_set{primal_c};
    this->project_cell_function_on_trace(current_component_set,
                                         TraceProjectionStrategy::l2_average);
  }



  template <int dim, class Permittivity>
  void
  NPSolver<dim, Permittivity>::project_component(
    const Component                                    c,
    const std::shared_ptr<const dealii::Function<dim>> c_function)
  {
    switch (c)
      {
        case Component::V:
          this->project_component_private<Component::V>(c_function);
          break;
        case Component::n:
          this->project_component_private<Component::n>(c_function);
          break;
        case Component::p:
          this->project_component_private<Component::p>(c_function);
          break;
        case Component::phi_n:
          this->project_component_private<Component::phi_n>(c_function);
          break;
        case Component::phi_p:
          this->project_component_private<Component::phi_p>(c_function);
          break;
        default:
          Assert(false, InvalidComponent());
      }
  }



  template class NPSolver<1, HomogeneousPermittivity<1>>;
  template class NPSolver<2, HomogeneousPermittivity<2>>;
  template class NPSolver<3, HomogeneousPermittivity<3>>;
} // namespace Ddhdg
