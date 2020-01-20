#pragma once

#include <deal.II/base/function_parser.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include "constants.h"

namespace Ddhdg
{
  template <int dim>
  class RecombinationTerm
  {
  public:
    virtual double
    compute_recombination_term(double n, const dealii::Point<dim> &q) const = 0;

    virtual double
    compute_derivative_of_recombination_term(
      double                    n,
      const dealii::Point<dim> &q) const = 0;

    virtual void
    compute_multiple_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<dealii::Point<dim>> &P,
      std::vector<double> &                  r) const;

    virtual void
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<dealii::Point<dim>> &P,
      std::vector<double> &                  r) const;

    virtual ~RecombinationTerm()
    {}
  };

  template <int dim>
  class LinearRecombinationTerm : public RecombinationTerm<dim>
  {
  public:
    explicit LinearRecombinationTerm(const std::string &constant_term,
                                     const std::string &linear_coefficient);

    explicit LinearRecombinationTerm(
      const LinearRecombinationTerm<dim> &analytic_recombination_term);

    [[nodiscard]] inline std::string
    get_constant_term() const
    {
      return constant_term;
    }

    [[nodiscard]] inline std::string
    get_linear_coefficient() const
    {
      return linear_coefficient;
    }

    virtual double
    compute_recombination_term(double n, const dealii::Point<dim> &q) const;

    virtual double
    compute_derivative_of_recombination_term(double                    n,
                                             const dealii::Point<dim> &q) const;

    virtual void
    compute_multiple_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<dealii::Point<dim>> &P,
      std::vector<double> &                  r) const;

    virtual void
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<dealii::Point<dim>> &P,
      std::vector<double> &                  r) const;

    virtual ~LinearRecombinationTerm()
    {}

  private:
    const std::string constant_term;
    const std::string linear_coefficient;

    dealii::FunctionParser<dim> parsed_constant_term;
    dealii::FunctionParser<dim> parsed_linear_coefficient;
  };
} // namespace Ddhdg
