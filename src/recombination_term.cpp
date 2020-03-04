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
    std::vector<double> &                  r) const
  {
    const std::size_t n_of_points = P.size();

    Assert(n_of_points == n.size(),
           dealii::ExcDimensionMismatch(n_of_points, n.size()));
    Assert(n_of_points == p.size(),
           dealii::ExcDimensionMismatch(n_of_points, p.size()));
    Assert(n_of_points == r.size(),
           dealii::ExcDimensionMismatch(n_of_points, r.size()));

    for (std::size_t q = 0; q < n_of_points; q++)
      r[q] = compute_recombination_term(n[q], p[q], P[q]);
  }



  template <int dim>
  void
  RecombinationTerm<dim>::compute_multiple_derivatives_of_recombination_terms(
    const std::vector<double> &            n,
    const std::vector<double> &            p,
    const std::vector<dealii::Point<dim>> &P,
    const Component                        c,
    std::vector<double> &                  r) const
  {
    const std::size_t n_of_points = P.size();

    Assert(n_of_points == n.size(),
           dealii::ExcDimensionMismatch(n_of_points, n.size()));
    Assert(n_of_points == p.size(),
           dealii::ExcDimensionMismatch(n_of_points, p.size()));
    Assert(n_of_points == r.size(),
           dealii::ExcDimensionMismatch(n_of_points, r.size()));

    for (std::size_t q = 0; q < n_of_points; q++)
      r[q] = compute_derivative_of_recombination_term(n[q], p[q], P[q], c);
  }



  template <int dim>
  LinearRecombinationTerm<dim>::LinearRecombinationTerm(
    const std::string &constant_term,
    const std::string &n_linear_coefficient,
    const std::string &p_linear_coefficient)
    : constant_term(constant_term)
    , n_linear_coefficient(n_linear_coefficient)
    , p_linear_coefficient(p_linear_coefficient)
    , parsed_constant_term(1.)
    , parsed_n_linear_coefficient(1.)
    , parsed_p_linear_coefficient(1.)
  {
    parsed_constant_term.initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      constant_term,
      Constants::constants);
    parsed_n_linear_coefficient.initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      n_linear_coefficient,
      Constants::constants);
    parsed_p_linear_coefficient.initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      p_linear_coefficient,
      Constants::constants);
  }



  template <int dim>
  LinearRecombinationTerm<dim>::LinearRecombinationTerm(
    const LinearRecombinationTerm<dim> &linear_recombination_term)
    : constant_term(linear_recombination_term.get_constant_term())
    , n_linear_coefficient(linear_recombination_term.get_n_linear_coefficient())
    , p_linear_coefficient(linear_recombination_term.get_p_linear_coefficient())
    , parsed_constant_term(1.)
    , parsed_n_linear_coefficient(1.)
    , parsed_p_linear_coefficient(1.)
  {
    parsed_constant_term.initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      constant_term,
      Constants::constants);
    parsed_n_linear_coefficient.initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      n_linear_coefficient,
      Constants::constants);
    parsed_p_linear_coefficient.initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      n_linear_coefficient,
      Constants::constants);
  }



  template <int dim>
  double
  LinearRecombinationTerm<dim>::compute_recombination_term(
    const double              n,
    const double              p,
    const dealii::Point<dim> &q) const
  {
    const double a = this->parsed_constant_term.value(q);
    const double b = this->parsed_n_linear_coefficient.value(q);
    const double c = this->parsed_p_linear_coefficient.value(q);
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
          return this->parsed_n_linear_coefficient.value(q);
        case Component::p:
          return this->parsed_p_linear_coefficient.value(q);
        default:
          Assert(false, UnknownComponent());
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
    std::vector<double> &                  r) const
  {
    const std::size_t n_of_points = P.size();

    Assert(n_of_points == n.size(),
           dealii::ExcDimensionMismatch(n_of_points, n.size()));
    Assert(n_of_points == p.size(),
           dealii::ExcDimensionMismatch(n_of_points, p.size()));
    Assert(n_of_points == r.size(),
           dealii::ExcDimensionMismatch(n_of_points, r.size()));

    std::vector<double> a(n_of_points);
    this->parsed_constant_term.value_list(P, a);

    std::vector<double> b(n_of_points);
    this->parsed_n_linear_coefficient.value_list(P, b);

    std::vector<double> c(n_of_points);
    this->parsed_p_linear_coefficient.value_list(P, c);

    for (std::size_t q = 0; q < n_of_points; q++)
      r[q] = a[q] + b[q] * n[q] + c[q] * p[q];
  }



  template <int dim>
  void
  LinearRecombinationTerm<dim>::
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<double> &            p,
      const std::vector<dealii::Point<dim>> &P,
      const Component                        c,
      std::vector<double> &                  r) const
  {
    (void)n;
    (void)p;
    Assert(P.size() == n.size(),
           dealii::ExcDimensionMismatch(P.size(), n.size()));
    Assert(P.size() == p.size(),
           dealii::ExcDimensionMismatch(P.size(), p.size()));
    Assert(P.size() == r.size(),
           dealii::ExcDimensionMismatch(P.size(), r.size()));

    switch (c)
      {
        case Component::n:
          this->parsed_n_linear_coefficient.value_list(P, r);
          break;
        case Component::p:
          this->parsed_p_linear_coefficient.value_list(P, r);
          break;
        default:
          Assert(false, UnknownComponent());
          break;
      }
  }


  template class RecombinationTerm<1>;
  template class RecombinationTerm<2>;
  template class RecombinationTerm<3>;

  template class LinearRecombinationTerm<1>;
  template class LinearRecombinationTerm<2>;
  template class LinearRecombinationTerm<3>;

} // namespace Ddhdg
