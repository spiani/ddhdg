#include "recombination_term.h"

#include <utility>


namespace Ddhdg
{
  template <int dim>
  void
  RecombinationTerm<dim>::compute_multiple_recombination_terms(
    const std::vector<double> &            n,
    const std::vector<double> &            p,
    const std::vector<dealii::Point<dim>> &P,
    bool                                   clear_vector,
    std::vector<double> &                  r) const
  {
    const std::size_t n_of_points = P.size();

    Assert(n_of_points == n.size(),
           dealii::ExcDimensionMismatch(n_of_points, n.size()));
    Assert(n_of_points == p.size(),
           dealii::ExcDimensionMismatch(n_of_points, p.size()));
    Assert(n_of_points == r.size(),
           dealii::ExcDimensionMismatch(n_of_points, r.size()));

    if (clear_vector)
      for (std::size_t q = 0; q < n_of_points; q++)
        r[q] = 0.;

    for (std::size_t q = 0; q < n_of_points; q++)
      r[q] += compute_recombination_term(n[q], p[q], P[q]);
  }



  template <int dim>
  void
  RecombinationTerm<dim>::compute_multiple_derivatives_of_recombination_terms(
    const std::vector<double> &            n,
    const std::vector<double> &            p,
    const std::vector<dealii::Point<dim>> &P,
    const Component                        c,
    bool                                   clear_vector,
    std::vector<double> &                  r) const
  {
    const std::size_t n_of_points = P.size();

    Assert(n_of_points == n.size(),
           dealii::ExcDimensionMismatch(n_of_points, n.size()));
    Assert(n_of_points == p.size(),
           dealii::ExcDimensionMismatch(n_of_points, p.size()));
    Assert(n_of_points == r.size(),
           dealii::ExcDimensionMismatch(n_of_points, r.size()));

    if (clear_vector)
      for (std::size_t q = 0; q < n_of_points; q++)
        r[q] = 0.;

    for (std::size_t q = 0; q < n_of_points; q++)
      r[q] += compute_derivative_of_recombination_term(n[q], p[q], P[q], c);
  }



  template <int dim>
  LinearRecombinationTerm<dim>::LinearRecombinationTerm(
    const std::shared_ptr<dealii::Function<dim>> constant_term,
    const std::shared_ptr<dealii::Function<dim>> n_linear_coefficient,
    const std::shared_ptr<dealii::Function<dim>> p_linear_coefficient)
    : constant_term(constant_term)
    , n_linear_coefficient(n_linear_coefficient)
    , p_linear_coefficient(p_linear_coefficient)
  {}



  template <int dim>
  LinearRecombinationTerm<dim>::LinearRecombinationTerm(
    const std::string &constant_term_str,
    const std::string &n_linear_coefficient_str,
    const std::string &p_linear_coefficient_str)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> parsed_constant_term =
      std::make_shared<dealii::FunctionParser<dim>>(1);
    std::shared_ptr<dealii::FunctionParser<dim>> parsed_n_linear_coefficient =
      std::make_shared<dealii::FunctionParser<dim>>(1);
    std::shared_ptr<dealii::FunctionParser<dim>> parsed_p_linear_coefficient =
      std::make_shared<dealii::FunctionParser<dim>>(1);
    parsed_constant_term->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      constant_term_str,
      Constants::constants);
    parsed_n_linear_coefficient->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      n_linear_coefficient_str,
      Constants::constants);
    parsed_p_linear_coefficient->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      p_linear_coefficient_str,
      Constants::constants);
    this->constant_term        = parsed_constant_term;
    this->n_linear_coefficient = parsed_n_linear_coefficient;
    this->p_linear_coefficient = parsed_p_linear_coefficient;
  }



  template <int dim>
  LinearRecombinationTerm<dim>::LinearRecombinationTerm(
    const LinearRecombinationTerm<dim> &linear_recombination_term)
    : constant_term(linear_recombination_term.constant_term)
    , n_linear_coefficient(linear_recombination_term.n_linear_coefficient)
    , p_linear_coefficient(linear_recombination_term.p_linear_coefficient)
  {}



  template <int dim>
  double
  LinearRecombinationTerm<dim>::compute_recombination_term(
    const double              n,
    const double              p,
    const dealii::Point<dim> &q) const
  {
    const double a = this->constant_term->value(q);
    const double b = this->n_linear_coefficient->value(q);
    const double c = this->p_linear_coefficient->value(q);
    return a + n * b + p * c;
  }



  template <int dim>
  double
  LinearRecombinationTerm<dim>::compute_derivative_of_recombination_term(
    const double              n,
    const double              p,
    const dealii::Point<dim> &q,
    Component                 c) const
  {
    (void)n;
    (void)p;
    switch (c)
      {
        case Component::n:
          return this->n_linear_coefficient->value(q);
        case Component::p:
          return this->p_linear_coefficient->value(q);
        default:
          Assert(false, InvalidComponent());
          break;
      }
    return 0.;
  }



  template <int dim>
  void
  LinearRecombinationTerm<dim>::compute_multiple_recombination_terms(
    const std::vector<double> &            n,
    const std::vector<double> &            p,
    const std::vector<dealii::Point<dim>> &P,
    bool                                   clear_vector,
    std::vector<double> &                  r) const
  {
    const std::size_t n_of_points = P.size();

    Assert(n_of_points == n.size(),
           dealii::ExcDimensionMismatch(n_of_points, n.size()));
    Assert(n_of_points == p.size(),
           dealii::ExcDimensionMismatch(n_of_points, p.size()));
    Assert(n_of_points == r.size(),
           dealii::ExcDimensionMismatch(n_of_points, r.size()));

    if (clear_vector)
      for (std::size_t q = 0; q < n_of_points; q++)
        r[q] = 0.;

    std::vector<double> a(n_of_points);
    this->constant_term->value_list(P, a);

    std::vector<double> b(n_of_points);
    this->n_linear_coefficient->value_list(P, b);

    std::vector<double> c(n_of_points);
    this->p_linear_coefficient->value_list(P, c);

    for (std::size_t q = 0; q < n_of_points; q++)
      r[q] += a[q] + b[q] * n[q] + c[q] * p[q];
  }



  template <int dim>
  void
  LinearRecombinationTerm<dim>::
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<double> &            p,
      const std::vector<dealii::Point<dim>> &P,
      const Component                        c,
      bool                                   clear_vector,
      std::vector<double> &                  r) const
  {
    (void)n;
    (void)p;

    const std::size_t n_of_points = P.size();

    Assert(n_of_points == n.size(),
           dealii::ExcDimensionMismatch(n_of_points, n.size()));
    Assert(n_of_points == p.size(),
           dealii::ExcDimensionMismatch(n_of_points, p.size()));
    Assert(n_of_points == r.size(),
           dealii::ExcDimensionMismatch(n_of_points, r.size()));

    std::vector<double> old_values;

    if (!clear_vector)
      {
        old_values.resize(n_of_points);
        old_values = r;
      }

    switch (c)
      {
        case Component::n:
          this->n_linear_coefficient->value_list(P, r);
          break;
        case Component::p:
          this->p_linear_coefficient->value_list(P, r);
          break;
        default:
          Assert(false, InvalidComponent());
          break;
      }

    if (!clear_vector)
      for (std::size_t q = 0; q < n_of_points; q++)
        r[q] += old_values[q];
  }


  template class RecombinationTerm<1>;
  template class RecombinationTerm<2>;
  template class RecombinationTerm<3>;

  template class LinearRecombinationTerm<1>;
  template class LinearRecombinationTerm<2>;
  template class LinearRecombinationTerm<3>;

} // namespace Ddhdg
