#include "deal.II/base/exceptions.h"

#include "np_solver.h"

namespace Ddhdg
{
  namespace DerivativeQuantitiesInternalTools
  {
    struct DQ_phi_n
    {
      static constexpr bool is_a_vector   = false;
      static constexpr bool needs_V       = true;
      static constexpr bool needs_E       = false;
      static constexpr bool needs_n       = true;
      static constexpr bool needs_Wn      = false;
      static constexpr bool needs_p       = false;
      static constexpr bool needs_Wp      = false;
      static constexpr bool needs_epsilon = false;
      static constexpr bool needs_mu_n    = false;
      static constexpr bool needs_mu_p    = false;
      static constexpr bool needs_D_n     = false;
      static constexpr bool needs_D_p     = false;
      static constexpr bool needs_T       = true;

      static constexpr bool redimensionalize = true;

      template <int dim>
      static constexpr void
      generate_epsilon_input(dealii::Tensor<1, dim> &epsilon_input)
      {
        (void)epsilon_input;
      }

      template <int dim>
      static constexpr void
      generate_mu_n_input(const dealii::Tensor<1, dim> &E,
                          dealii::Tensor<1, dim>       &mu_n_input)
      {
        (void)E;
        (void)mu_n_input;
      }

      template <int dim>
      static constexpr void
      generate_mu_p_input(const dealii::Tensor<1, dim> &E,
                          dealii::Tensor<1, dim>       &mu_p_input)
      {
        (void)E;
        (void)mu_p_input;
      }

      template <int dim>
      static constexpr void
      generate_D_n_input(const dealii::Tensor<1, dim> &E,
                         const dealii::Tensor<1, dim> &Wn,
                         const dealii::Tensor<1, dim> &Wp,
                         dealii::Tensor<1, dim>       &D_n_input)
      {
        (void)E;
        (void)Wn;
        (void)Wp;
        (void)D_n_input;
      }

      template <int dim>
      static constexpr void
      generate_D_p_input(const dealii::Tensor<1, dim> &E,
                         const dealii::Tensor<1, dim> &Wn,
                         const dealii::Tensor<1, dim> &Wp,
                         dealii::Tensor<1, dim>       &D_p_input)
      {
        (void)E;
        (void)Wn;
        (void)Wp;
        (void)D_p_input;
      }

      template <int dim, typename ProblemType>
      static constexpr double
      compute(double                          V,
              dealii::Tensor<1, dim>          E,
              double                          n,
              dealii::Tensor<1, dim>          Wn,
              double                          p,
              dealii::Tensor<1, dim>          Wp,
              dealii::Tensor<1, dim>          epsilon_output,
              dealii::Tensor<1, dim>          mu_n_output,
              dealii::Tensor<1, dim>          mu_p_output,
              dealii::Tensor<1, dim>          D_n_output,
              dealii::Tensor<1, dim>          D_p_output,
              double                          T,
              const dealii::Point<dim>       &q,
              const Solver<dim, ProblemType> &solver,
              unsigned int                    dimension = 0)
      {
        (void)E;
        (void)Wn;
        (void)p;
        (void)Wp;
        (void)epsilon_output;
        (void)mu_n_output;
        (void)mu_p_output;
        (void)D_n_output;
        (void)D_p_output;
        (void)q;
        (void)dimension;
        const double rescaling_factor =
          solver.adimensionalizer
            ->template get_component_rescaling_factor<Component::V>();
        const double phi_n =
          solver.template compute_quasi_fermi_potential<Component::n>(n, V, T);

        // I divide for the rescaling factor; this ensures that the number that
        // I get is reasonable; after having solved the linear system, I will
        // multiply the solution by the rescaling factor to restore the right
        // number
        return phi_n / rescaling_factor;
      }
    };

    struct DQ_phi_p
    {
      static constexpr bool is_a_vector   = false;
      static constexpr bool needs_V       = true;
      static constexpr bool needs_E       = false;
      static constexpr bool needs_n       = false;
      static constexpr bool needs_Wn      = false;
      static constexpr bool needs_p       = true;
      static constexpr bool needs_Wp      = false;
      static constexpr bool needs_epsilon = false;
      static constexpr bool needs_mu_n    = false;
      static constexpr bool needs_mu_p    = false;
      static constexpr bool needs_D_n     = false;
      static constexpr bool needs_D_p     = false;
      static constexpr bool needs_T       = true;

      static constexpr bool redimensionalize = true;

      template <int dim>
      static constexpr void
      generate_epsilon_input(dealii::Tensor<1, dim> &epsilon_input)
      {
        (void)epsilon_input;
      }

      template <int dim>
      static constexpr void
      generate_mu_n_input(const dealii::Tensor<1, dim> &E,
                          dealii::Tensor<1, dim>       &mu_n_input)
      {
        (void)E;
        (void)mu_n_input;
      }

      template <int dim>
      static constexpr void
      generate_mu_p_input(const dealii::Tensor<1, dim> &E,
                          dealii::Tensor<1, dim>       &mu_p_input)
      {
        (void)E;
        (void)mu_p_input;
      }

      template <int dim>
      static constexpr void
      generate_D_n_input(const dealii::Tensor<1, dim> &E,
                         const dealii::Tensor<1, dim> &Wn,
                         const dealii::Tensor<1, dim> &Wp,
                         dealii::Tensor<1, dim>       &D_n_input)
      {
        (void)E;
        (void)Wn;
        (void)Wp;
        (void)D_n_input;
      }

      template <int dim>
      static constexpr void
      generate_D_p_input(const dealii::Tensor<1, dim> &E,
                         const dealii::Tensor<1, dim> &Wn,
                         const dealii::Tensor<1, dim> &Wp,
                         dealii::Tensor<1, dim>       &D_p_input)
      {
        (void)E;
        (void)Wn;
        (void)Wp;
        (void)D_p_input;
      }

      template <int dim, typename ProblemType>
      static constexpr double
      compute(double                          V,
              dealii::Tensor<1, dim>          E,
              double                          n,
              dealii::Tensor<1, dim>          Wn,
              double                          p,
              dealii::Tensor<1, dim>          Wp,
              dealii::Tensor<1, dim>          epsilon_output,
              dealii::Tensor<1, dim>          mu_n_output,
              dealii::Tensor<1, dim>          mu_p_output,
              dealii::Tensor<1, dim>          D_n_output,
              dealii::Tensor<1, dim>          D_p_output,
              double                          T,
              const dealii::Point<dim>       &q,
              const Solver<dim, ProblemType> &solver,
              unsigned int                    dimension = 0)
      {
        (void)E;
        (void)n;
        (void)Wn;
        (void)Wp;
        (void)epsilon_output;
        (void)mu_n_output;
        (void)mu_p_output;
        (void)D_n_output;
        (void)D_p_output;
        (void)q;
        (void)dimension;
        const double rescaling_factor =
          solver.adimensionalizer
            ->template get_component_rescaling_factor<Component::V>();
        const double phi_p =
          solver.template compute_quasi_fermi_potential<Component::p>(p, V, T);
        return phi_p / rescaling_factor;
      }
    };

    struct DQ_Jn
    {
      static constexpr bool is_a_vector   = true;
      static constexpr bool needs_V       = false;
      static constexpr bool needs_E       = true;
      static constexpr bool needs_n       = true;
      static constexpr bool needs_Wn      = true;
      static constexpr bool needs_p       = false;
      static constexpr bool needs_Wp      = false;
      static constexpr bool needs_epsilon = false;
      static constexpr bool needs_mu_n    = true;
      static constexpr bool needs_mu_p    = false;
      static constexpr bool needs_D_n     = true;
      static constexpr bool needs_D_p     = false;
      static constexpr bool needs_T       = false;

      static constexpr bool redimensionalize = false;

      template <int dim>
      static constexpr void
      generate_epsilon_input(dealii::Tensor<1, dim> &epsilon_input)
      {
        (void)epsilon_input;
      }

      template <int dim>
      static constexpr void
      generate_mu_n_input(const dealii::Tensor<1, dim> &E,
                          dealii::Tensor<1, dim>       &mu_n_input)
      {
        mu_n_input = E;
      }

      template <int dim>
      static constexpr void
      generate_mu_p_input(const dealii::Tensor<1, dim> &E,
                          dealii::Tensor<1, dim>       &mu_p_input)
      {
        mu_p_input = E;
      }

      template <int dim>
      static constexpr void
      generate_D_n_input(const dealii::Tensor<1, dim> &E,
                         const dealii::Tensor<1, dim> &Wn,
                         const dealii::Tensor<1, dim> &Wp,
                         dealii::Tensor<1, dim>       &D_n_input)
      {
        (void)E;
        (void)Wp;
        D_n_input = Wn;
      }

      template <int dim>
      static constexpr void
      generate_D_p_input(const dealii::Tensor<1, dim> &E,
                         const dealii::Tensor<1, dim> &Wn,
                         const dealii::Tensor<1, dim> &Wp,
                         dealii::Tensor<1, dim>       &D_p_input)
      {
        (void)E;
        (void)Wn;
        (void)Wp;
        (void)D_p_input;
      }

      template <int dim, typename ProblemType>
      static constexpr double
      compute(double                          V,
              dealii::Tensor<1, dim>          E,
              double                          n,
              dealii::Tensor<1, dim>          Wn,
              double                          p,
              dealii::Tensor<1, dim>          Wp,
              dealii::Tensor<1, dim>          epsilon_output,
              dealii::Tensor<1, dim>          mu_n_output,
              dealii::Tensor<1, dim>          mu_p_output,
              dealii::Tensor<1, dim>          D_n_output,
              dealii::Tensor<1, dim>          D_p_output,
              double                          T,
              const dealii::Point<dim>       &q,
              const Solver<dim, ProblemType> &solver,
              unsigned int                    dimension = 0)
      {
        (void)V;
        (void)E;
        (void)Wn;
        (void)p;
        (void)Wp;
        (void)epsilon_output;
        (void)mu_p_output;
        (void)D_p_output;
        (void)T;
        (void)q;
        (void)solver;
        return n * mu_n_output[dimension] - D_n_output[dimension];
      }
    };



    struct DQ_Jp
    {
      static constexpr bool is_a_vector   = true;
      static constexpr bool needs_V       = false;
      static constexpr bool needs_E       = true;
      static constexpr bool needs_n       = false;
      static constexpr bool needs_Wn      = false;
      static constexpr bool needs_p       = true;
      static constexpr bool needs_Wp      = true;
      static constexpr bool needs_epsilon = false;
      static constexpr bool needs_mu_n    = false;
      static constexpr bool needs_mu_p    = true;
      static constexpr bool needs_D_n     = false;
      static constexpr bool needs_D_p     = true;
      static constexpr bool needs_T       = false;

      static constexpr bool redimensionalize = false;

      template <int dim>
      static constexpr void
      generate_epsilon_input(dealii::Tensor<1, dim> &epsilon_input)
      {
        (void)epsilon_input;
      }

      template <int dim>
      static constexpr void
      generate_mu_n_input(const dealii::Tensor<1, dim> &E,
                          dealii::Tensor<1, dim>       &mu_n_input)
      {
        mu_n_input = E;
      }

      template <int dim>
      static constexpr void
      generate_mu_p_input(const dealii::Tensor<1, dim> &E,
                          dealii::Tensor<1, dim>       &mu_p_input)
      {
        mu_p_input = E;
      }

      template <int dim>
      static constexpr void
      generate_D_n_input(const dealii::Tensor<1, dim> &E,
                         const dealii::Tensor<1, dim> &Wn,
                         const dealii::Tensor<1, dim> &Wp,
                         dealii::Tensor<1, dim>       &D_n_input)
      {
        (void)E;
        (void)Wn;
        (void)Wp;
        (void)D_n_input;
      }

      template <int dim>
      static constexpr void
      generate_D_p_input(const dealii::Tensor<1, dim> &E,
                         const dealii::Tensor<1, dim> &Wn,
                         const dealii::Tensor<1, dim> &Wp,
                         dealii::Tensor<1, dim>       &D_p_input)
      {
        (void)E;
        (void)Wn;
        D_p_input = Wp;
      }

      template <int dim, typename ProblemType>
      static constexpr double
      compute(double                          V,
              dealii::Tensor<1, dim>          E,
              double                          n,
              dealii::Tensor<1, dim>          Wn,
              double                          p,
              dealii::Tensor<1, dim>          Wp,
              dealii::Tensor<1, dim>          epsilon_output,
              dealii::Tensor<1, dim>          mu_n_output,
              dealii::Tensor<1, dim>          mu_p_output,
              dealii::Tensor<1, dim>          D_n_output,
              dealii::Tensor<1, dim>          D_p_output,
              double                          T,
              const dealii::Point<dim>       &q,
              const Solver<dim, ProblemType> &solver,
              unsigned int                    dimension = 0)
      {
        (void)V;
        (void)E;
        (void)n;
        (void)Wn;
        (void)Wp;
        (void)epsilon_output;
        (void)mu_n_output;
        (void)D_n_output;
        (void)T;
        (void)q;
        (void)solver;
        return p * mu_p_output[dimension] + D_p_output[dimension];
      }
    };



    struct DQ_R
    {
      static constexpr bool is_a_vector   = false;
      static constexpr bool needs_V       = false;
      static constexpr bool needs_E       = false;
      static constexpr bool needs_n       = true;
      static constexpr bool needs_Wn      = false;
      static constexpr bool needs_p       = true;
      static constexpr bool needs_Wp      = false;
      static constexpr bool needs_epsilon = false;
      static constexpr bool needs_mu_n    = false;
      static constexpr bool needs_mu_p    = false;
      static constexpr bool needs_D_n     = false;
      static constexpr bool needs_D_p     = false;
      static constexpr bool needs_T       = false;

      static constexpr bool redimensionalize = false;

      template <int dim>
      static constexpr void
      generate_epsilon_input(dealii::Tensor<1, dim> &epsilon_input)
      {
        (void)epsilon_input;
      }

      template <int dim>
      static constexpr void
      generate_mu_n_input(const dealii::Tensor<1, dim> &E,
                          dealii::Tensor<1, dim>       &mu_n_input)
      {
        (void)E;
        (void)mu_n_input;
      }

      template <int dim>
      static constexpr void
      generate_mu_p_input(const dealii::Tensor<1, dim> &E,
                          dealii::Tensor<1, dim>       &mu_p_input)
      {
        (void)E;
        (void)mu_p_input;
      }

      template <int dim>
      static constexpr void
      generate_D_n_input(const dealii::Tensor<1, dim> &E,
                         const dealii::Tensor<1, dim> &Wn,
                         const dealii::Tensor<1, dim> &Wp,
                         dealii::Tensor<1, dim>       &D_n_input)
      {
        (void)E;
        (void)Wn;
        (void)Wp;
        (void)D_n_input;
      }

      template <int dim>
      static constexpr void
      generate_D_p_input(const dealii::Tensor<1, dim> &E,
                         const dealii::Tensor<1, dim> &Wn,
                         const dealii::Tensor<1, dim> &Wp,
                         dealii::Tensor<1, dim>       &D_p_input)
      {
        (void)E;
        (void)Wn;
        (void)Wp;
        (void)D_p_input;
      }

      template <int dim, typename ProblemType>
      static constexpr double
      compute(double                            V,
              dealii::Tensor<1, dim>            E,
              double                            n,
              dealii::Tensor<1, dim>            Wn,
              double                            p,
              dealii::Tensor<1, dim>            Wp,
              dealii::Tensor<1, dim>            epsilon_output,
              dealii::Tensor<1, dim>            mu_n_output,
              dealii::Tensor<1, dim>            mu_p_output,
              dealii::Tensor<1, dim>            D_n_output,
              dealii::Tensor<1, dim>            D_p_output,
              double                            T,
              const dealii::Point<dim>         &q,
              const NPSolver<dim, ProblemType> &solver,
              unsigned int                      dimension = 0)
      {
        (void)V;
        (void)E;
        (void)Wn;
        (void)Wp;
        (void)epsilon_output;
        (void)mu_n_output;
        (void)mu_p_output;
        (void)D_n_output;
        (void)D_p_output;
        (void)T;
        (void)dimension;
        const double rescaling =
          solver.adimensionalizer->get_component_rescaling_factor(Component::n);
        return solver.problem->recombination_term->compute_recombination_term(
          n, p, q, rescaling);
      }
    };



    struct DQCopyData
    {
      explicit DQCopyData(unsigned int dofs_per_cell);

      DQCopyData(const DQCopyData &other) = default;

      std::vector<unsigned int> dof_indices;
      std::vector<double>       dof_values;
    };

    DQCopyData::DQCopyData(const unsigned int dofs_per_cell)
      : dof_indices(dofs_per_cell)
      , dof_values(dofs_per_cell)
    {}

    template <int dim, typename quantity, typename Problem>
    struct DQScratchData
    {
      template <bool vector_quantity>
      static std::vector<std::vector<unsigned int>>
      generate_dof_dimension_map(const dealii::FiniteElement<dim> &fe);

      DQScratchData(
        const dealii::FiniteElement<dim> &fe_quantity,
        const dealii::FiniteElement<dim> &fe_cell,
        const dealii::QGauss<dim>        &quadrature_formula,
        dealii::UpdateFlags               fe_values_flags,
        dealii::UpdateFlags               fe_values_cell_flags,
        const typename Problem::PermittivityClass::PermittivityComputer
                                                                 &permittivity,
        const typename Problem::NMobilityClass::MobilityComputer &n_mobility,
        const typename Problem::PMobilityClass::MobilityComputer &p_mobility);

      DQScratchData(
        const DQScratchData<dim, quantity, Problem> &dq_scratch_data);

      dealii::FEValues<dim> fe_values_quantity;
      dealii::FEValues<dim> fe_values_cell;

      const unsigned int dofs_per_dimension;

      std::vector<Point<dim>> cell_quadrature_points;

      std::vector<dealii::LAPACKFullMatrix<double>> projection_matrix;
      std::vector<dealii::Vector<double>>           rhs;

      std::vector<dealii::types::global_dof_index> global_dof_indices;

      std::vector<double> quantity_values;

      std::vector<double>                 V_values;
      std::vector<dealii::Tensor<1, dim>> E_values;
      std::vector<double>                 n_values;
      std::vector<dealii::Tensor<1, dim>> Wn_values;
      std::vector<double>                 p_values;
      std::vector<dealii::Tensor<1, dim>> Wp_values;

      std::vector<double> T;
      std::vector<double> U_T_cell;

      std::vector<dealii::Tensor<1, dim>> eps_output;

      std::vector<dealii::Tensor<1, dim>> mu_n_output;
      std::vector<dealii::Tensor<1, dim>> mu_p_output;

      std::vector<dealii::Tensor<1, dim>> D_n_output;
      std::vector<dealii::Tensor<1, dim>> D_p_output;

      std::vector<double> base_functions;

      typename Problem::PermittivityClass::PermittivityComputer permittivity;
      typename Problem::NMobilityClass::MobilityComputer        n_mobility;
      typename Problem::PMobilityClass::MobilityComputer        p_mobility;

      const std::vector<std::vector<unsigned int>> dof_dimension_map;
    };

    template <int dim, typename quantity, typename Problem>
    template <bool vector_quantity>
    std::vector<std::vector<unsigned int>>
    DQScratchData<dim, quantity, Problem>::generate_dof_dimension_map(
      const dealii::FiniteElement<dim> &fe)
    {
      constexpr unsigned int n_of_dims = (vector_quantity) ? dim : 1;
      std::vector<std::vector<unsigned int>> dof_dimension_map(n_of_dims);

      if constexpr (n_of_dims == 1)
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          dof_dimension_map[0].push_back(i);
      else
        {
          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            {
              const unsigned int current_component =
                fe.system_to_component_index(i).first;
              dof_dimension_map[current_component].push_back(i);
            }
        }
      return dof_dimension_map;
    }

    template <int dim, typename quantity, typename Problem>
    DQScratchData<dim, quantity, Problem>::DQScratchData(
      const dealii::FiniteElement<dim> &fe_quantity,
      const dealii::FiniteElement<dim> &fe_cell,
      const dealii::QGauss<dim>        &quadrature_formula,
      const dealii::UpdateFlags         fe_values_flags,
      const dealii::UpdateFlags         fe_values_cell_flags,
      const typename Problem::PermittivityClass::PermittivityComputer
                                                               &permittivity,
      const typename Problem::NMobilityClass::MobilityComputer &n_mobility,
      const typename Problem::PMobilityClass::MobilityComputer &p_mobility)
      : fe_values_quantity(fe_quantity, quadrature_formula, fe_values_flags)
      , fe_values_cell(fe_cell, quadrature_formula, fe_values_cell_flags)
      , dofs_per_dimension((quantity::is_a_vector) ?
                             fe_quantity.dofs_per_cell / dim :
                             fe_quantity.dofs_per_cell)
      , cell_quadrature_points(quadrature_formula.size())
      , projection_matrix(std::vector<dealii::LAPACKFullMatrix<double>>(
          (quantity::is_a_vector) ? dim : 1,
          dealii::LAPACKFullMatrix<double>(this->dofs_per_dimension,
                                           this->dofs_per_dimension)))
      , rhs(std::vector<dealii::Vector<double>>(
          (quantity::is_a_vector) ? dim : 1,
          dealii::Vector<double>(this->dofs_per_dimension)))
      , global_dof_indices(fe_quantity.dofs_per_cell)
      , quantity_values(quadrature_formula.size())
      , V_values((quantity::needs_V) ? quadrature_formula.size() : 0)
      , E_values((quantity::needs_E) ? quadrature_formula.size() : 0)
      , n_values((quantity::needs_n) ? quadrature_formula.size() : 0)
      , Wn_values((quantity::needs_Wn) ? quadrature_formula.size() : 0)
      , p_values((quantity::needs_p) ? quadrature_formula.size() : 0)
      , Wp_values((quantity::needs_Wp) ? quadrature_formula.size() : 0)
      , T((quantity::needs_T || quantity::needs_D_n || quantity::needs_D_p) ?
            quadrature_formula.size() :
            0)
      , U_T_cell(
          (quantity::needs_T || quantity::needs_D_n || quantity::needs_D_p) ?
            quadrature_formula.size() :
            0)
      , eps_output((quantity::needs_epsilon) ? quadrature_formula.size() : 0)
      , mu_n_output((quantity::needs_mu_n) ? quadrature_formula.size() : 0)
      , mu_p_output((quantity::needs_mu_p) ? quadrature_formula.size() : 0)
      , D_n_output((quantity::needs_D_n) ? quadrature_formula.size() : 0)
      , D_p_output((quantity::needs_D_p) ? quadrature_formula.size() : 0)
      , base_functions(this->dofs_per_dimension)
      , permittivity(permittivity)
      , n_mobility(n_mobility)
      , p_mobility(p_mobility)
      , dof_dimension_map(
          generate_dof_dimension_map<quantity::is_a_vector>(fe_quantity))
    {}

    template <int dim, typename quantity, typename Problem>
    DQScratchData<dim, quantity, Problem>::DQScratchData(
      const DQScratchData<dim, quantity, Problem> &dq_scratch_data)
      : fe_values_quantity(
          dq_scratch_data.fe_values_quantity.get_fe(),
          dq_scratch_data.fe_values_quantity.get_quadrature(),
          dq_scratch_data.fe_values_quantity.get_update_flags())
      , fe_values_cell(dq_scratch_data.fe_values_cell.get_fe(),
                       dq_scratch_data.fe_values_cell.get_quadrature(),
                       dq_scratch_data.fe_values_cell.get_update_flags())
      , dofs_per_dimension(dq_scratch_data.dofs_per_dimension)
      , cell_quadrature_points(dq_scratch_data.cell_quadrature_points.size())
      , projection_matrix(dq_scratch_data.projection_matrix)
      , rhs(dq_scratch_data.rhs)
      , global_dof_indices(dq_scratch_data.global_dof_indices)
      , quantity_values(dq_scratch_data.quantity_values)
      , V_values(dq_scratch_data.V_values)
      , E_values(dq_scratch_data.E_values)
      , n_values(dq_scratch_data.n_values)
      , Wn_values(dq_scratch_data.Wn_values)
      , p_values(dq_scratch_data.p_values)
      , Wp_values(dq_scratch_data.Wp_values)
      , T(dq_scratch_data.T)
      , U_T_cell(dq_scratch_data.U_T_cell)
      , eps_output(dq_scratch_data.eps_output)
      , mu_n_output(dq_scratch_data.mu_n_output)
      , mu_p_output(dq_scratch_data.mu_p_output)
      , D_n_output(dq_scratch_data.D_n_output)
      , D_p_output(dq_scratch_data.D_p_output)
      , base_functions(dq_scratch_data.base_functions)
      , permittivity(dq_scratch_data.permittivity)
      , n_mobility(dq_scratch_data.n_mobility)
      , p_mobility(dq_scratch_data.p_mobility)
      , dof_dimension_map(dq_scratch_data.dof_dimension_map)
    {}
  } // namespace DerivativeQuantitiesInternalTools



  template <int dim, typename Problem>
  template <typename DQScratchData, typename DQCopyData, typename quantity>
  void
  NPSolver<dim, Problem>::derivative_quantities_project_on_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    DQScratchData                                        &scratch,
    DQCopyData                                           &copy_data) const
  {
    typename DoFHandler<dim>::active_cell_iterator system_cell(
      &(*(this->triangulation)),
      cell->level(),
      cell->index(),
      &(this->dof_handler_cell));

    scratch.fe_values_quantity.reinit(cell);
    scratch.fe_values_cell.reinit(system_cell);

    const unsigned int n_of_systems = (quantity::is_a_vector) ? dim : 1;
    const unsigned int n_q_points =
      scratch.fe_values_quantity.get_quadrature().size();

    cell->get_dof_indices(scratch.global_dof_indices);

    // Clear the current system (for all the components)
    for (unsigned int i = 0; i < n_of_systems; ++i)
      {
        scratch.projection_matrix[i] = 0.;
        scratch.rhs[i]               = 0.;
      }

    // Copy quadrature points
    for (unsigned int q = 0; q < n_q_points; ++q)
      scratch.cell_quadrature_points[q] =
        scratch.fe_values_quantity.quadrature_point(q);

    // Compute everything that we need to compute the derived quantity
    if constexpr (quantity::needs_V)
      {
        const auto extractor = this->get_component_extractor(Component::V);
        scratch.fe_values_cell[extractor].get_function_values(
          this->current_solution_cell, scratch.V_values);
        if (quantity::redimensionalize)
          this->adimensionalizer
            ->template inplace_redimensionalize_component<Component::V>(
              scratch.V_values);
      }
    if constexpr (quantity::needs_E)
      {
        const auto extractor =
          this->get_displacement_extractor(Displacement::E);
        scratch.fe_values_cell[extractor].get_function_values(
          this->current_solution_cell, scratch.E_values);
        if (quantity::redimensionalize)
          this->adimensionalizer->inplace_redimensionalize_displacement(
            scratch.E_values, Displacement::E);
      }
    if constexpr (quantity::needs_n)
      {
        const auto extractor = this->get_component_extractor(Component::n);
        scratch.fe_values_cell[extractor].get_function_values(
          this->current_solution_cell, scratch.n_values);
        if (quantity::redimensionalize)
          this->adimensionalizer
            ->template inplace_redimensionalize_component<Component::n>(
              scratch.n_values);
      }
    if constexpr (quantity::needs_Wn)
      {
        const auto extractor =
          this->get_displacement_extractor(Displacement::Wn);
        scratch.fe_values_cell[extractor].get_function_values(
          this->current_solution_cell, scratch.Wn_values);
        if (quantity::redimensionalize)
          this->adimensionalizer->inplace_redimensionalize_displacement(
            scratch.Wn_values, Displacement::Wn);
      }
    if constexpr (quantity::needs_p)
      {
        const auto extractor = this->get_component_extractor(Component::p);
        scratch.fe_values_cell[extractor].get_function_values(
          this->current_solution_cell, scratch.p_values);
        if (quantity::redimensionalize)
          this->adimensionalizer
            ->template inplace_redimensionalize_component<Component::p>(
              scratch.p_values);
      }
    if constexpr (quantity::needs_Wp)
      {
        const auto extractor =
          this->get_displacement_extractor(Displacement::Wp);
        scratch.fe_values_cell[extractor].get_function_values(
          this->current_solution_cell, scratch.Wp_values);
        if (quantity::redimensionalize)
          this->adimensionalizer->inplace_redimensionalize_displacement(
            scratch.Wp_values, Displacement::Wp);
      }
    if constexpr (quantity::needs_epsilon)
      {
        scratch.permittivity.initialize_on_cell(scratch.cell_quadrature_points);

        dealii::Tensor<1, dim> epsilon_input;
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            quantity::generate_epsilon_input(epsilon_input);
            scratch.permittivity.epsilon_operator_on_cell(
              q, epsilon_input, scratch.epsilon_output[q]);
          }
      }
    if constexpr (quantity::needs_mu_n || quantity::needs_D_n)
      scratch.n_mobility.initialize_on_cell(scratch.cell_quadrature_points);
    if constexpr (quantity::needs_mu_p || quantity::needs_D_p)
      scratch.p_mobility.initialize_on_cell(scratch.cell_quadrature_points);

    if constexpr (quantity::needs_T || quantity::needs_D_n ||
                  quantity::needs_D_p)
      {
        this->problem->temperature->value_list(scratch.cell_quadrature_points,
                                               scratch.T);

        const double thermal_voltage_rescaling_factor =
          (quantity::redimensionalize) ?
            1 :
            this->adimensionalizer->get_thermal_voltage_rescaling_factor();

        for (unsigned int q = 0; q < n_q_points; q++)
          scratch.U_T_cell[q] =
            scratch.T[q] * Constants::KB /
            (Constants::Q * thermal_voltage_rescaling_factor);
      }

    if constexpr (quantity::needs_mu_n)
      {
        dealii::Tensor<1, dim> mu_n_input;
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            quantity::generate_mu_n_input(scratch.E_values[q], mu_n_input);
            scratch.n_mobility.mu_operator_on_cell(q,
                                                   mu_n_input,
                                                   scratch.mu_n_output[q]);
          }
      }
    if constexpr (quantity::needs_mu_p)
      {
        dealii::Tensor<1, dim> mu_p_input;
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            quantity::generate_mu_p_input(scratch.E_values[q], mu_p_input);
            scratch.n_mobility.mu_operator_on_cell(q,
                                                   mu_p_input,
                                                   scratch.mu_p_output[q]);
          }
      }
    if constexpr (quantity::needs_D_n)
      {
        dealii::Tensor<1, dim> D_n_input;
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            quantity::generate_D_n_input(scratch.E_values[q],
                                         scratch.Wn_values[q],
                                         scratch.Wp_values[q],
                                         D_n_input);
            this->apply_einstein_diffusion_coefficient<Component::n, false>(
              scratch, q, D_n_input, scratch.D_n_output[q]);
          }
      }
    if constexpr (quantity::needs_D_p)
      {
        dealii::Tensor<1, dim> D_p_input;
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            quantity::generate_D_p_input(scratch.E_values[q],
                                         scratch.Wn_values[q],
                                         scratch.Wp_values[q],
                                         D_p_input);
            this->apply_einstein_diffusion_coefficient<Component::p, false>(
              scratch, q, D_p_input, scratch.D_p_output[q]);
          }
      }

    // This tensor will be used as a placeholder for the values that must not be
    // computed
    dealii::Tensor<1, dim> empty_tensor;

    // Finally, we can start the real function! Now we project the function
    // component by component (i.e. if the quantity is a vector of three
    // components, we solve three systems, one for each component)
    for (unsigned int d = 0; d < n_of_systems; ++d)
      {
        // First of all, compute the value of the derived quantity on each
        // quadrature point
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const double V_value =
              (quantity::needs_V) ? scratch.V_values[q] : 0.;
            const double n_value =
              (quantity::needs_n) ? scratch.n_values[q] : 0.;
            const double p_value =
              (quantity::needs_p) ? scratch.p_values[q] : 0.;
            const double T_value = (quantity::needs_T) ? scratch.T[q] : 0.;

            const dealii::Tensor<1, dim> &E_value =
              (quantity::needs_E) ? scratch.E_values[q] : empty_tensor;
            const dealii::Tensor<1, dim> &Wn_value =
              (quantity::needs_Wn) ? scratch.Wn_values[q] : empty_tensor;
            const dealii::Tensor<1, dim> &Wp_value =
              (quantity::needs_Wp) ? scratch.Wp_values[q] : empty_tensor;
            const dealii::Tensor<1, dim> &epsilon_value =
              (quantity::needs_epsilon) ? scratch.eps_output[q] : empty_tensor;
            const dealii::Tensor<1, dim> &mu_n_output =
              (quantity::needs_mu_n) ? scratch.mu_n_output[q] : empty_tensor;
            const dealii::Tensor<1, dim> &mu_p_output =
              (quantity::needs_mu_p) ? scratch.mu_p_output[q] : empty_tensor;
            const dealii::Tensor<1, dim> &D_n_output =
              (quantity::needs_D_n) ? scratch.D_n_output[q] : empty_tensor;
            const dealii::Tensor<1, dim> &D_p_output =
              (quantity::needs_D_p) ? scratch.D_p_output[q] : empty_tensor;

            scratch.quantity_values[q] =
              quantity::compute(V_value,
                                E_value,
                                n_value,
                                Wn_value,
                                p_value,
                                Wp_value,
                                epsilon_value,
                                mu_n_output,
                                mu_p_output,
                                D_n_output,
                                D_p_output,
                                T_value,
                                scratch.cell_quadrature_points[q],
                                *this,
                                d);
          }

        // We prepare some pointers for the current matrix
        auto &projection_matrix = scratch.projection_matrix[d];
        auto &rhs               = scratch.rhs[d];

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            // Get the values of the shape functions
            for (unsigned int i = 0; i < scratch.dofs_per_dimension; ++i)
              {
                const unsigned int ii = scratch.dof_dimension_map[d][i];
                scratch.base_functions[i] =
                  scratch.fe_values_quantity.shape_value(ii, q);
              }

            const double JxW = scratch.fe_values_quantity.JxW(q);

            // Assemble the matrix
            for (unsigned int i = 0; i < scratch.dofs_per_dimension; ++i)
              for (unsigned int j = 0; j < scratch.dofs_per_dimension; ++j)
                projection_matrix(i, j) +=
                  scratch.base_functions[i] * scratch.base_functions[j] * JxW;

            // Assemble the Rhs
            for (unsigned int i = 0; i < scratch.dofs_per_dimension; ++i)
              rhs[i] +=
                scratch.quantity_values[q] * scratch.base_functions[i] * JxW;
          }

        // Solve the linear system
        projection_matrix.compute_lu_factorization();
        projection_matrix.solve(rhs);

        // Copy the values inside the copy_data object
        for (unsigned int i = 0; i < scratch.dofs_per_dimension; ++i)
          {
            const unsigned int ii     = scratch.dof_dimension_map[d][i];
            copy_data.dof_indices[ii] = scratch.global_dof_indices[ii];
            copy_data.dof_values[ii]  = rhs[i];
          }
      }
  }



  template <int dim, typename Problem>
  template <typename DQCopyData>
  void
  NPSolver<dim, Problem>::derivative_quantities_copier(
    const DQCopyData       &copy_data,
    dealii::Vector<double> &data) const
  {
    for (unsigned int i = 0; i < copy_data.dof_indices.size(); ++i)
      data[copy_data.dof_indices[i]] = copy_data.dof_values[i];
  }



  template <int dim, typename Problem>
  template <typename quantity>
  void
  NPSolver<dim, Problem>::derivative_quantities_compute_derived_quantity(
    const dealii::DoFHandler<dim> &dof,
    dealii::Vector<double>        &data) const
  {
    typedef DerivativeQuantitiesInternalTools::
      DQScratchData<dim, quantity, Problem>
                                                          ScratchData;
    typedef DerivativeQuantitiesInternalTools::DQCopyData CopyData;

    Assert(
      this->initialized,
      ExcMessage(
        "Can not produce derived quantities when the system is not initialized"));

    data.reinit(dof.n_dofs());

    const unsigned int n_q_per_side = this->get_number_of_quadrature_points();

    const QGauss<dim> quadrature_formula(n_q_per_side);

    const UpdateFlags flags(update_values | update_JxW_values |
                            update_quadrature_points);
    const UpdateFlags flags_cell(update_values | update_JxW_values);

    ScratchData scratch(
      dof.get_fe(),
      *(this->fe_cell),
      quadrature_formula,
      flags,
      flags_cell,
      this->problem->permittivity->get_computer(
        quantity::redimensionalize ?
          1. :
          this->adimensionalizer->get_permittivity_rescaling_factor()),
      this->problem->n_mobility->get_computer(
        quantity::redimensionalize ?
          1. :
          this->adimensionalizer->get_mobility_rescaling_factor()),
      this->problem->p_mobility->get_computer(
        quantity::redimensionalize ?
          1. :
          this->adimensionalizer->get_mobility_rescaling_factor()));

    CopyData copy_data(dof.get_fe().dofs_per_cell);

    for (const auto &cell : dof.active_cell_iterators())
      {
        this->derivative_quantities_project_on_one_cell<ScratchData,
                                                        CopyData,
                                                        quantity>(cell,
                                                                  scratch,
                                                                  copy_data);
        this->derivative_quantities_copier(copy_data, data);
      }
  }



  template <int dim, typename ProblemType>
  template <Component cmp>
  void
  NPSolver<dim, ProblemType>::compute_current(
    const dealii::DoFHandler<dim> &dof,
    dealii::Vector<double>        &data,
    const bool                     redimensionalize) const
  {
    switch (cmp)
      {
        case Component::n:
          this->derivative_quantities_compute_derived_quantity<
            DerivativeQuantitiesInternalTools::DQ_Jn>(dof, data);
          break;
        case Component::p:
          this->derivative_quantities_compute_derived_quantity<
            DerivativeQuantitiesInternalTools::DQ_Jp>(dof, data);
          break;
        default:
          Assert(false, InvalidComponent())
      }

    const double thermal_voltage_rf =
      this->adimensionalizer->get_thermal_voltage_rescaling_factor();
    const double mu_rf =
      this->adimensionalizer->get_mobility_rescaling_factor();
    const double n_rf =
      this->adimensionalizer
        ->template get_displacement_rescaling_factor<Displacement::Wn>();
    const double J_rf = thermal_voltage_rf * mu_rf * n_rf * Constants::Q;

    if (redimensionalize)
      for (unsigned int i = 0; i < data.size(); ++i)
        data[i] *= J_rf;
  }



  template <int dim, typename ProblemType>
  template <Component cmp>
  void
  NPSolver<dim, ProblemType>::compute_qf_potential(
    const dealii::DoFHandler<dim> &dof,
    dealii::Vector<double>        &data) const
  {
    switch (cmp)
      {
        case Component::n:
          this->derivative_quantities_compute_derived_quantity<
            DerivativeQuantitiesInternalTools::DQ_phi_n>(dof, data);
          break;
        case Component::p:
          this->derivative_quantities_compute_derived_quantity<
            DerivativeQuantitiesInternalTools::DQ_phi_p>(dof, data);
          break;
        default:
          Assert(false, InvalidComponent())
      }
    const double rescaling_factor =
      this->adimensionalizer
        ->template get_component_rescaling_factor<Component::V>();

    for (unsigned int i = 0; i < data.size(); ++i)
      data[i] *= rescaling_factor;
  }


  template <int dim, typename ProblemType>
  void
  NPSolver<dim, ProblemType>::compute_recombination_term(
    const dealii::DoFHandler<dim> &dof,
    dealii::Vector<double>        &data,
    const bool                     redimensionalize) const
  {
    this->derivative_quantities_compute_derived_quantity<
      DerivativeQuantitiesInternalTools::DQ_R>(dof, data);

    if (!redimensionalize)
      this->adimensionalizer->adimensionalize_recombination_term(data);
  }



  template void
  NPSolver<1, HomogeneousProblem<1>>::compute_qf_potential<Component::n>(
    const dealii::DoFHandler<1> &dof,
    dealii::Vector<double>      &data) const;
  template void
  NPSolver<2, HomogeneousProblem<2>>::compute_qf_potential<Component::n>(
    const dealii::DoFHandler<2> &dof,
    dealii::Vector<double>      &data) const;
  template void
  NPSolver<3, HomogeneousProblem<3>>::compute_qf_potential<Component::n>(
    const dealii::DoFHandler<3> &dof,
    dealii::Vector<double>      &data) const;
  template void
  NPSolver<1, HomogeneousProblem<1>>::compute_qf_potential<Component::p>(
    const dealii::DoFHandler<1> &dof,
    dealii::Vector<double>      &data) const;
  template void
  NPSolver<2, HomogeneousProblem<2>>::compute_qf_potential<Component::p>(
    const dealii::DoFHandler<2> &dof,
    dealii::Vector<double>      &data) const;
  template void
  NPSolver<3, HomogeneousProblem<3>>::compute_qf_potential<Component::p>(
    const dealii::DoFHandler<3> &dof,
    dealii::Vector<double>      &data) const;

  template void
  NPSolver<1, HomogeneousProblem<1>>::compute_current<Component::n>(
    const dealii::DoFHandler<1> &dof,
    dealii::Vector<double>      &data,
    bool                         redimensionalize) const;
  template void
  NPSolver<2, HomogeneousProblem<2>>::compute_current<Component::n>(
    const dealii::DoFHandler<2> &dof,
    dealii::Vector<double>      &data,
    bool                         redimensionalize) const;
  template void
  NPSolver<3, HomogeneousProblem<3>>::compute_current<Component::n>(
    const dealii::DoFHandler<3> &dof,
    dealii::Vector<double>      &data,
    bool                         redimensionalize) const;
  template void
  NPSolver<1, HomogeneousProblem<1>>::compute_current<Component::p>(
    const dealii::DoFHandler<1> &dof,
    dealii::Vector<double>      &data,
    bool                         redimensionalize) const;
  template void
  NPSolver<2, HomogeneousProblem<2>>::compute_current<Component::p>(
    const dealii::DoFHandler<2> &dof,
    dealii::Vector<double>      &data,
    bool                         redimensionalize) const;
  template void
  NPSolver<3, HomogeneousProblem<3>>::compute_current<Component::p>(
    const dealii::DoFHandler<3> &dof,
    dealii::Vector<double>      &data,
    bool                         redimensionalize) const;

  template void
  NPSolver<1, HomogeneousProblem<1>>::compute_recombination_term(
    const dealii::DoFHandler<1> &dof,
    dealii::Vector<double>      &data,
    bool                         redimensionalize) const;
  template void
  NPSolver<2, HomogeneousProblem<2>>::compute_recombination_term(
    const dealii::DoFHandler<2> &dof,
    dealii::Vector<double>      &data,
    bool                         redimensionalize) const;
  template void
  NPSolver<3, HomogeneousProblem<3>>::compute_recombination_term(
    const dealii::DoFHandler<3> &dof,
    dealii::Vector<double>      &data,
    bool                         redimensionalize) const;
} // namespace Ddhdg
