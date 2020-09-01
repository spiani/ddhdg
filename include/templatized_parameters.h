#pragma once

#include "np_solver.h"

namespace Ddhdg
{
  DeclExceptionMsg(MetaprogrammingError,
                   "Something went wrong with the loop on templates");

  template <int dim, typename ProblemType>
  class TemplatizedParametersInterface
  {
  public:
    [[nodiscard]] virtual unsigned int
    get_parameter_mask() const = 0;

    virtual TemplatizedParametersInterface<dim, ProblemType> *
    get_previous() const = 0;

    virtual
      typename NPSolver<dim, ProblemType>::assemble_system_one_cell_pointer
      get_assemble_system_one_cell_function() const = 0;

    virtual ~TemplatizedParametersInterface() = default;
  };

  template <int dim, typename ProblemType, unsigned int parameter_mask>
  class TemplatizedParameters
    : public TemplatizedParametersInterface<dim, ProblemType>
  {
  public:
    static const unsigned int mask         = parameter_mask;
    static const bool         thermodyn_eq = (parameter_mask / 8) % 2;
    static const bool         is_V_enabled = (parameter_mask / 4) % 2;
    static const bool         is_n_enabled = (parameter_mask / 2) % 2;
    static const bool         is_p_enabled = (parameter_mask / 1) % 2;

    [[nodiscard]] unsigned int
    get_parameter_mask() const override;

    TemplatizedParametersInterface<dim, ProblemType> *
    get_previous() const override;

    typename NPSolver<dim, ProblemType>::assemble_system_one_cell_pointer
    get_assemble_system_one_cell_function() const override;
  };

  template <int dim, typename ProblemType, unsigned int parameter_mask>
  unsigned int
  TemplatizedParameters<dim, ProblemType, parameter_mask>::get_parameter_mask()
    const
  {
    return mask;
  }

  template <int dim, typename ProblemType, unsigned int parameter_mask>
  TemplatizedParametersInterface<dim, ProblemType> *
  TemplatizedParameters<dim, ProblemType, parameter_mask>::get_previous() const
  {
    if (parameter_mask == 0)
      Assert(false, MetaprogrammingError());
    const unsigned int k = (parameter_mask == 0) ? 0 : parameter_mask - 1;
    return new TemplatizedParameters<dim, ProblemType, k>();
  }

  template <int dim, typename ProblemType, unsigned int parameter_mask>
  typename NPSolver<dim, ProblemType>::assemble_system_one_cell_pointer
  TemplatizedParameters<dim, ProblemType, parameter_mask>::
    get_assemble_system_one_cell_function() const
  {
    return &NPSolver<dim, ProblemType>::template assemble_system_one_cell<
      TemplatizedParameters<dim, ProblemType, parameter_mask>>;
  }

} // namespace Ddhdg
