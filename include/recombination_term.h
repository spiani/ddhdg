#pragma once

#include <deal.II/base/function_parser.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include "components.h"
#include "constants.h"

namespace Ddhdg
{
  template <int dim>
  class RecombinationTerm
  {
  public:
    virtual double
    compute_recombination_term(double                    n,
                               double                    p,
                               const dealii::Point<dim> &q) const = 0;

    virtual double
    compute_derivative_of_recombination_term(double                    n,
                                             double                    p,
                                             const dealii::Point<dim> &q,
                                             Component c) const = 0;

    virtual void
    compute_multiple_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<double> &            p,
      const std::vector<dealii::Point<dim>> &P,
      std::vector<double> &                  r) const;

    virtual void
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<double> &            p,
      const std::vector<dealii::Point<dim>> &P,
      Component                              c,
      std::vector<double> &                  r) const;

    virtual ~RecombinationTerm() = default;
  };

  template <int dim>
  class LinearRecombinationTerm : public RecombinationTerm<dim>
  {
  public:
    explicit LinearRecombinationTerm(
      std::shared_ptr<dealii::Function<dim>> constant_term,
      std::shared_ptr<dealii::Function<dim>> n_linear_coefficient,
      std::shared_ptr<dealii::Function<dim>> p_linear_coefficient);

    explicit LinearRecombinationTerm(
      const std::string &constant_term_str,
      const std::string &n_linear_coefficient_str,
      const std::string &p_linear_coefficient_str);

    LinearRecombinationTerm(
      const LinearRecombinationTerm<dim> &linear_recombination_term);

    [[nodiscard]] inline std::shared_ptr<dealii::Function<dim>>
    get_constant_term() const
    {
      return constant_term;
    }

    [[nodiscard]] inline std::shared_ptr<dealii::Function<dim>>
    get_n_linear_coefficient() const
    {
      return n_linear_coefficient;
    }

    [[nodiscard]] inline std::shared_ptr<dealii::Function<dim>>
    get_p_linear_coefficient() const
    {
      return p_linear_coefficient;
    }

    double
    compute_recombination_term(double                    n,
                               double                    p,
                               const dealii::Point<dim> &q) const override;

    double
    compute_derivative_of_recombination_term(double                    n,
                                             double                    p,
                                             const dealii::Point<dim> &q,
                                             Component c) const override;

    void
    compute_multiple_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<double> &            p,
      const std::vector<dealii::Point<dim>> &P,
      std::vector<double> &                  r) const override;

    void
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<double> &            p,
      const std::vector<dealii::Point<dim>> &P,
      Component                              c,
      std::vector<double> &                  r) const override;

    virtual ~LinearRecombinationTerm() = default;

  private:
    std::shared_ptr<dealii::Function<dim>> constant_term;
    std::shared_ptr<dealii::Function<dim>> n_linear_coefficient;
    std::shared_ptr<dealii::Function<dim>> p_linear_coefficient;
  };
} // namespace Ddhdg
