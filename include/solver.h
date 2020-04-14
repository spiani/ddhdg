#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/thread_management.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/chunk_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include <iostream>

#include "convergence_table.h"
#include "nonlinear_iteration_results.h"
#include "problem.h"

namespace Ddhdg
{
  using namespace dealii;

  DeclExceptionMsg(NoTraceIn1D, "The trace can not be saved in 1D");

  template <int dim>
  class Solver
  {
  public:
    virtual void
    refine_grid(unsigned int i) = 0;

    virtual void
    refine_grid_once();

    virtual void
    interpolate_component(
      Component                                    c,
      std::shared_ptr<const dealii::Function<dim>> c_function) = 0;

    virtual void
    project_component(
      Component                                    c,
      std::shared_ptr<const dealii::Function<dim>> c_function) = 0;

    virtual void
    set_component(Component                                    c,
                  std::shared_ptr<const dealii::Function<dim>> c_function,
                  bool                                         use_projection);

    virtual void
    set_current_solution(
      std::shared_ptr<const dealii::Function<dim>> V_function,
      std::shared_ptr<const dealii::Function<dim>> n_function,
      std::shared_ptr<const dealii::Function<dim>> p_function,
      bool                                         use_projection) = 0;

    [[nodiscard]] virtual bool
    is_enabled(Component c) const = 0;

    virtual void
    enable_component(Component c) = 0;

    virtual void
    disable_component(Component c) = 0;

    virtual void
    enable_components(const std::set<Component> &c) = 0;

    virtual void
    disable_components(const std::set<Component> &c) = 0;

    virtual void
    set_enabled_components(bool V_enabled, bool n_enabled, bool p_enabled) = 0;

    virtual NonlinearIterationResults
    run(double absolute_tol,
        double relative_tol,
        int    max_number_of_iterations) = 0;

    virtual NonlinearIterationResults
    run() = 0;

    virtual double
    estimate_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c,
      dealii::VectorTools::NormType                norm) const = 0;

    virtual double
    estimate_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Displacement                                 d,
      dealii::VectorTools::NormType                norm) const = 0;

    virtual double
    estimate_error_on_trace(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c,
      dealii::VectorTools::NormType                norm) const = 0;

    virtual double
    estimate_l2_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c) const = 0;

    virtual double
    estimate_l2_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Displacement                                 d) const = 0;

    virtual double
    estimate_l2_error_on_trace(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c) const = 0;

    virtual double
    estimate_h1_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c) const = 0;

    virtual double
    estimate_h1_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Displacement                                 d) const = 0;

    virtual double
    estimate_linfty_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c) const = 0;

    virtual double
    estimate_linfty_error(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Displacement                                 d) const = 0;

    virtual double
    estimate_linfty_error_on_trace(
      std::shared_ptr<const dealii::Function<dim>> expected_solution,
      Component                                    c) const = 0;

    virtual void
    output_results(const std::string &solution_filename,
                   bool               save_update) const = 0;

    virtual void
    output_results(const std::string &solution_filename) const;

    virtual void
    output_results(const std::string &solution_filename,
                   const std::string &trace_filename,
                   bool               save_update) const = 0;

    virtual void
    output_results(const std::string &solution_filename,
                   const std::string &trace_filename) const;

    virtual void
    print_convergence_table(
      std::shared_ptr<Ddhdg::ConvergenceTable>     error_table,
      std::shared_ptr<const dealii::Function<dim>> expected_V_solution,
      std::shared_ptr<const dealii::Function<dim>> expected_n_solution,
      std::shared_ptr<const dealii::Function<dim>> expected_p_solution,
      unsigned int                                 n_cycles,
      unsigned int                                 initial_refinements) = 0;

    virtual void
    print_convergence_table(
      std::shared_ptr<Ddhdg::ConvergenceTable>     error_table,
      std::shared_ptr<const dealii::Function<dim>> expected_V_solution,
      std::shared_ptr<const dealii::Function<dim>> expected_n_solution,
      std::shared_ptr<const dealii::Function<dim>> expected_p_solution,
      std::shared_ptr<const dealii::Function<dim>> initial_V_function,
      std::shared_ptr<const dealii::Function<dim>> initial_n_function,
      std::shared_ptr<const dealii::Function<dim>> initial_p_function,
      unsigned int                                 n_cycles,
      unsigned int                                 initial_refinements) = 0;

    virtual ~Solver() = default;
  };

} // namespace Ddhdg
