#include "recombination_term.h"

#include <utility>

#include "function_tools.h"


namespace Ddhdg
{
  template <int dim>
  void
  RecombinationTerm<dim>::compute_multiple_recombination_terms(
    const std::vector<double> &            n,
    const std::vector<double> &            p,
    const std::vector<dealii::Point<dim>> &P,
    bool                                   clear_vector,
    std::vector<double> &                  r)
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
    std::vector<double> &                  r)
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
    std::vector<double> &                  r)
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
      std::vector<double> &                  r)
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



  template <int dim>
  SuperimposedRecombinationTerm<dim>::SuperimposedRecombinationTerm(
    const std::vector<std::shared_ptr<RecombinationTerm<dim>>>
      &recombination_terms)
    : recombination_terms(recombination_terms)
  {}



  template <int dim>
  SuperimposedRecombinationTerm<dim>::SuperimposedRecombinationTerm(
    std::shared_ptr<RecombinationTerm<dim>> recombination_term1,
    std::shared_ptr<RecombinationTerm<dim>> recombination_term2)
    : SuperimposedRecombinationTerm(
        std::vector<std::shared_ptr<RecombinationTerm<dim>>>{
          recombination_term1,
          recombination_term2})
  {}



  template <int dim>
  SuperimposedRecombinationTerm<dim>::SuperimposedRecombinationTerm(
    std::shared_ptr<RecombinationTerm<dim>> recombination_term1,
    std::shared_ptr<RecombinationTerm<dim>> recombination_term2,
    std::shared_ptr<RecombinationTerm<dim>> recombination_term3)
    : SuperimposedRecombinationTerm(
        std::vector<std::shared_ptr<RecombinationTerm<dim>>>{
          recombination_term1,
          recombination_term2,
          recombination_term3})
  {}



  template <int dim>
  SuperimposedRecombinationTerm<dim>::SuperimposedRecombinationTerm(
    const SuperimposedRecombinationTerm<dim> &superimposed_recombination_term)
    : recombination_terms(superimposed_recombination_term.recombination_terms)
  {}



  template <int dim>
  double
  SuperimposedRecombinationTerm<dim>::compute_recombination_term(
    const double              n,
    const double              p,
    const dealii::Point<dim> &q) const
  {
    double k = 0;
    for (const auto &recombination_term : this->recombination_terms)
      k += recombination_term->compute_recombination_term(n, p, q);
    return k;
  }



  template <int dim>
  double
  SuperimposedRecombinationTerm<dim>::compute_derivative_of_recombination_term(
    const double              n,
    const double              p,
    const dealii::Point<dim> &q,
    Component                 c) const
  {
    double k = 0;
    for (const auto &recombination_term : this->recombination_terms)
      k += recombination_term->compute_derivative_of_recombination_term(n,
                                                                        p,
                                                                        q,
                                                                        c);
    return k;
  }



  template <int dim>
  void
  SuperimposedRecombinationTerm<dim>::compute_multiple_recombination_terms(
    const std::vector<double> &            n,
    const std::vector<double> &            p,
    const std::vector<dealii::Point<dim>> &P,
    bool                                   clear_vector,
    std::vector<double> &                  r)
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

    for (const auto &recombination_term : this->recombination_terms)
      recombination_term->compute_multiple_recombination_terms(
        n, p, P, false, r);
  }



  template <int dim>
  void
  SuperimposedRecombinationTerm<dim>::
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<double> &            p,
      const std::vector<dealii::Point<dim>> &P,
      const Component                        c,
      bool                                   clear_vector,
      std::vector<double> &                  r)
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

    for (const auto &recombination_term : this->recombination_terms)
      recombination_term->compute_multiple_derivatives_of_recombination_terms(
        n, p, P, c, false, r);
  }



  template <int dim>
  ShockleyReadHallFixedTemperature<dim>::ShockleyReadHallFixedTemperature(
    double conduction_band_density,
    double valence_band_density,
    double conduction_band_edge_energy,
    double valence_band_edge_energy,
    double temperature,
    double electron_life_time,
    double hole_life_time)
    : intrinsic_carrier_concentration(
        ShockleyReadHallFixedTemperature<dim>::
          compute_intrinsic_carrier_concentration(conduction_band_density,
                                                  valence_band_density,
                                                  conduction_band_edge_energy,
                                                  valence_band_edge_energy,
                                                  temperature))
    , electron_life_time(electron_life_time)
    , hole_life_time(hole_life_time)
  {}



  template <int dim>
  ShockleyReadHallFixedTemperature<dim>::ShockleyReadHallFixedTemperature(
    double intrinsic_carrier_concentration,
    double electron_life_time,
    double hole_life_time)
    : intrinsic_carrier_concentration(intrinsic_carrier_concentration)
    , electron_life_time(electron_life_time)
    , hole_life_time(hole_life_time)
  {}



  template <int dim>
  double
  ShockleyReadHallFixedTemperature<dim>::compute_recombination_term(
    const double              n,
    const double              p,
    const dealii::Point<dim> &q) const
  {
    (void)q;
    const double ni    = this->intrinsic_carrier_concentration;
    const double tau_n = this->electron_life_time;
    const double tau_p = this->hole_life_time;

    const double num = n * p - ni * ni;
    const double den = tau_p * (n + ni) + tau_n * (p + ni);
    return num / den;
  }



  template <int dim>
  double
  ShockleyReadHallFixedTemperature<
    dim>::compute_derivative_of_recombination_term(const double              n,
                                                   const double              p,
                                                   const dealii::Point<dim> &q,
                                                   Component c) const
  {
    (void)q;
    const double ni    = this->intrinsic_carrier_concentration;
    const double tau_n = this->electron_life_time;
    const double tau_p = this->hole_life_time;

    const double num = n * p - ni * ni;
    const double den = tau_p * (n + ni) + tau_n * (p + ni);

    switch (c)
      {
        case Component::n:
          return p / den - num * tau_p / (den * den);
        case Component::p:
          return n / den - num * tau_n / (den * den);
        default:
          Assert(false, InvalidComponent());
          return 9e99;
      }
  }



  template <int dim>
  void
  ShockleyReadHallFixedTemperature<dim>::compute_multiple_recombination_terms(
    const std::vector<double> &            n,
    const std::vector<double> &            p,
    const std::vector<dealii::Point<dim>> &P,
    bool                                   clear_vector,
    std::vector<double> &                  r)
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

    const double ni    = this->intrinsic_carrier_concentration;
    const double tau_n = this->electron_life_time;
    const double tau_p = this->hole_life_time;

    for (std::size_t q = 0; q < n_of_points; q++)
      {
        const double num = n[q] * p[q] - ni * ni;
        const double den = tau_p * (n[q] + ni) + tau_n * (p[q] + ni);
        r[q] += num / den;
      }
  }



  template <int dim>
  void
  ShockleyReadHallFixedTemperature<dim>::
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<double> &            p,
      const std::vector<dealii::Point<dim>> &P,
      const Component                        c,
      bool                                   clear_vector,
      std::vector<double> &                  r)
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

    const double ni    = this->intrinsic_carrier_concentration;
    const double tau_n = this->electron_life_time;
    const double tau_p = this->hole_life_time;

    switch (c)
      {
          case Component::n: {
            for (std::size_t q = 0; q < n_of_points; q++)
              {
                const double num = n[q] * p[q] - ni * ni;
                const double den = tau_p * (n[q] + ni) + tau_n * (p[q] + ni);
                r[q] += p[q] / den - num * tau_p / (den * den);
              }
            break;
          }
          case Component::p: {
            for (std::size_t q = 0; q < n_of_points; q++)
              {
                const double num = n[q] * p[q] - ni * ni;
                const double den = tau_p * (n[q] + ni) + tau_n * (p[q] + ni);
                r[q] += n[q] / den - num * tau_n / (den * den);
              }
            break;
          }
        default:
          Assert(false, InvalidComponent());
      }
  }



  template <int dim>
  AugerFixedTemperature<dim>::AugerFixedTemperature(
    double conduction_band_density,
    double valence_band_density,
    double conduction_band_edge_energy,
    double valence_band_edge_energy,
    double temperature,
    double n_coefficient,
    double p_coefficient)
    : intrinsic_carrier_concentration(
        conduction_band_density * valence_band_density *
        exp((valence_band_edge_energy - conduction_band_edge_energy) /
            (Constants::KB * temperature)))
    , n_coefficient(n_coefficient)
    , p_coefficient(p_coefficient)
  {}



  template <int dim>
  AugerFixedTemperature<dim>::AugerFixedTemperature(
    double intrinsic_carrier_concentration,
    double n_coefficient,
    double p_coefficient)
    : intrinsic_carrier_concentration(intrinsic_carrier_concentration)
    , n_coefficient(n_coefficient)
    , p_coefficient(p_coefficient)
  {}



  template <int dim>
  double
  AugerFixedTemperature<dim>::compute_recombination_term(
    const double              n,
    const double              p,
    const dealii::Point<dim> &q) const
  {
    (void)q;
    const double ni = this->intrinsic_carrier_concentration;
    return (this->n_coefficient * n + this->p_coefficient * p) * (n * p - ni);
  }



  template <int dim>
  double
  AugerFixedTemperature<dim>::compute_derivative_of_recombination_term(
    const double              n,
    const double              p,
    const dealii::Point<dim> &q,
    Component                 c) const
  {
    (void)q;
    const double ni  = this->intrinsic_carrier_concentration;
    const double C_n = this->n_coefficient;
    const double C_p = this->p_coefficient;

    switch (c)
      {
        case Component::n:
          return C_p * p * p + 2 * C_n * n * p - C_n * ni * ni;
        case Component::p:
          return C_n * n * n + 2 * C_n * n * p - C_p * ni * ni;
        default:
          Assert(false, InvalidComponent());
          return 9e99;
      }
  }



  template <int dim>
  void
  AugerFixedTemperature<dim>::compute_multiple_recombination_terms(
    const std::vector<double> &            n,
    const std::vector<double> &            p,
    const std::vector<dealii::Point<dim>> &P,
    bool                                   clear_vector,
    std::vector<double> &                  r)
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

    const double ni = this->intrinsic_carrier_concentration;

    for (std::size_t q = 0; q < n_of_points; q++)
      r[q] += (this->n_coefficient * n[q] + this->p_coefficient * p[q]) *
              (n[q] * p[q] - ni);
  }



  template <int dim>
  void
  AugerFixedTemperature<dim>::
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<double> &            p,
      const std::vector<dealii::Point<dim>> &P,
      const Component                        c,
      bool                                   clear_vector,
      std::vector<double> &                  r)
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

    const double ni  = this->intrinsic_carrier_concentration;
    const double C_n = this->n_coefficient;
    const double C_p = this->p_coefficient;

    switch (c)
      {
          case Component::n: {
            for (std::size_t q = 0; q < n_of_points; q++)
              r[q] += C_p * p[q] * p[q] + 2 * C_n * n[q] * p[q] - C_n * ni * ni;
            break;
          }
          case Component::p: {
            for (std::size_t q = 0; q < n_of_points; q++)
              r[q] += C_n * n[q] * n[q] + 2 * C_n * n[q] * p[q] - C_p * ni * ni;
            break;
          }
        default:
          Assert(false, InvalidComponent());
      }
  }



  template <int dim>
  ShockleyReadHall<dim>::ShockleyReadHall(
    double                                 conduction_band_density,
    double                                 valence_band_density,
    double                                 conduction_band_edge_energy,
    double                                 valence_band_edge_energy,
    std::shared_ptr<dealii::Function<dim>> temperature,
    double                                 electron_life_time,
    double                                 hole_life_time)
    : conduction_band_density(conduction_band_density)
    , valence_band_density(valence_band_density)
    , conduction_band_edge_energy(conduction_band_edge_energy)
    , valence_band_edge_energy(valence_band_edge_energy)
    , temperature(temperature)
    , electron_life_time(electron_life_time)
    , hole_life_time(hole_life_time)
  {
    Assert(temperature->n_components == 1, FunctionMustBeScalar());
  }



  template <int dim>
  double
  ShockleyReadHall<dim>::compute_recombination_term(
    const double              n,
    const double              p,
    const dealii::Point<dim> &q) const
  {
    const double T = this->temperature->value(q);
    const double ni =
      this->conduction_band_density * this->valence_band_density *
      exp((this->valence_band_edge_energy - this->conduction_band_edge_energy) /
          (Constants::KB * T));
    const double tau_n = this->electron_life_time;
    const double tau_p = this->hole_life_time;

    const double num = n * p - ni * ni;
    const double den = tau_p * (n + ni) + tau_n * (p + ni);
    return num / den;
  }



  template <int dim>
  double
  ShockleyReadHall<dim>::compute_derivative_of_recombination_term(
    const double              n,
    const double              p,
    const dealii::Point<dim> &q,
    Component                 c) const
  {
    const double T = this->temperature->value(q);
    const double ni =
      this->conduction_band_density * this->valence_band_density *
      exp((this->valence_band_edge_energy - this->conduction_band_edge_energy) /
          (Constants::KB * T));
    const double tau_n = this->electron_life_time;
    const double tau_p = this->hole_life_time;

    const double num = n * p - ni * ni;
    const double den = tau_p * (n + ni) + tau_n * (p + ni);

    switch (c)
      {
        case Component::n:
          return p / den - num * tau_p / (den * den);
        case Component::p:
          return n / den - num * tau_n / (den * den);
        default:
          Assert(false, InvalidComponent());
          return 9e99;
      }
  }



  template <int dim>
  void
  ShockleyReadHall<dim>::compute_multiple_recombination_terms(
    const std::vector<double> &            n,
    const std::vector<double> &            p,
    const std::vector<dealii::Point<dim>> &P,
    bool                                   clear_vector,
    std::vector<double> &                  r)
  {
    const std::size_t n_of_points = P.size();

    Assert(n_of_points == n.size(),
           dealii::ExcDimensionMismatch(n_of_points, n.size()));
    Assert(n_of_points == p.size(),
           dealii::ExcDimensionMismatch(n_of_points, p.size()));
    Assert(n_of_points == r.size(),
           dealii::ExcDimensionMismatch(n_of_points, r.size()));

    if (this->temperature_buffer.size() != n_of_points)
      this->temperature_buffer.resize(n_of_points);

    if (clear_vector)
      for (std::size_t q = 0; q < n_of_points; q++)
        r[q] = 0.;

    const double tau_n = this->electron_life_time;
    const double tau_p = this->hole_life_time;

    this->temperature->value_list(P, this->temperature_buffer);

    for (std::size_t q = 0; q < n_of_points; q++)
      {
        const double ni = this->conduction_band_density *
                          this->valence_band_density *
                          exp((this->valence_band_edge_energy -
                               this->conduction_band_edge_energy) /
                              (Constants::KB * temperature_buffer[q]));
        const double num = n[q] * p[q] - ni * ni;
        const double den = tau_p * (n[q] + ni) + tau_n * (p[q] + ni);
        r[q] += num / den;
      }
  }



  template <int dim>
  void
  ShockleyReadHall<dim>::compute_multiple_derivatives_of_recombination_terms(
    const std::vector<double> &            n,
    const std::vector<double> &            p,
    const std::vector<dealii::Point<dim>> &P,
    const Component                        c,
    bool                                   clear_vector,
    std::vector<double> &                  r)
  {
    const std::size_t n_of_points = P.size();

    Assert(n_of_points == n.size(),
           dealii::ExcDimensionMismatch(n_of_points, n.size()));
    Assert(n_of_points == p.size(),
           dealii::ExcDimensionMismatch(n_of_points, p.size()));
    Assert(n_of_points == r.size(),
           dealii::ExcDimensionMismatch(n_of_points, r.size()));

    if (this->temperature_buffer.size() != n_of_points)
      this->temperature_buffer.resize(n_of_points);

    if (clear_vector)
      for (std::size_t q = 0; q < n_of_points; q++)
        r[q] = 0.;

    const double tau_n = this->electron_life_time;
    const double tau_p = this->hole_life_time;

    this->temperature->value_list(P, temperature_buffer);

    switch (c)
      {
          case Component::n: {
            for (std::size_t q = 0; q < n_of_points; q++)
              {
                const double ni = this->conduction_band_density *
                                  this->valence_band_density *
                                  exp((this->valence_band_edge_energy -
                                       this->conduction_band_edge_energy) /
                                      (Constants::KB * temperature_buffer[q]));
                const double num = n[q] * p[q] - ni * ni;
                const double den = tau_p * (n[q] + ni) + tau_n * (p[q] + ni);
                r[q] += p[q] / den - num * tau_p / (den * den);
              }
            break;
          }
          case Component::p: {
            for (std::size_t q = 0; q < n_of_points; q++)
              {
                const double ni = this->conduction_band_density *
                                  this->valence_band_density *
                                  exp((this->valence_band_edge_energy -
                                       this->conduction_band_edge_energy) /
                                      (Constants::KB * temperature_buffer[q]));
                const double num = n[q] * p[q] - ni * ni;
                const double den = tau_p * (n[q] + ni) + tau_n * (p[q] + ni);
                r[q] += n[q] / den - num * tau_n / (den * den);
              }
            break;
          }
        default:
          Assert(false, InvalidComponent());
      }
  }



  template <int dim>
  Auger<dim>::Auger(double conduction_band_density,
                    double valence_band_density,
                    double conduction_band_edge_energy,
                    double valence_band_edge_energy,
                    std::shared_ptr<dealii::Function<dim>> temperature,
                    double                                 n_coefficient,
                    double                                 p_coefficient)
    : conduction_band_density(conduction_band_density)
    , valence_band_density(valence_band_density)
    , conduction_band_edge_energy(conduction_band_edge_energy)
    , valence_band_edge_energy(valence_band_edge_energy)
    , temperature(temperature)
    , n_coefficient(n_coefficient)
    , p_coefficient(p_coefficient)
  {}



  template <int dim>
  double
  Auger<dim>::compute_recombination_term(const double              n,
                                         const double              p,
                                         const dealii::Point<dim> &q) const
  {
    const double T = this->temperature->value(q);
    const double ni =
      this->conduction_band_density * this->valence_band_density *
      exp((this->valence_band_edge_energy - this->conduction_band_edge_energy) /
          (Constants::KB * T));
    return (this->n_coefficient * n + this->p_coefficient * p) * (n * p - ni);
  }



  template <int dim>
  double
  Auger<dim>::compute_derivative_of_recombination_term(
    const double              n,
    const double              p,
    const dealii::Point<dim> &q,
    Component                 c) const
  {
    const double T = this->temperature->value(q);
    const double ni =
      this->conduction_band_density * this->valence_band_density *
      exp((this->valence_band_edge_energy - this->conduction_band_edge_energy) /
          (Constants::KB * T));
    const double C_n = this->n_coefficient;
    const double C_p = this->p_coefficient;

    switch (c)
      {
        case Component::n:
          return C_p * p * p + 2 * C_n * n * p - C_n * ni * ni;
        case Component::p:
          return C_n * n * n + 2 * C_n * n * p - C_p * ni * ni;
        default:
          Assert(false, InvalidComponent());
          return 9e99;
      }
  }



  template <int dim>
  void
  Auger<dim>::compute_multiple_recombination_terms(
    const std::vector<double> &            n,
    const std::vector<double> &            p,
    const std::vector<dealii::Point<dim>> &P,
    bool                                   clear_vector,
    std::vector<double> &                  r)
  {
    const std::size_t n_of_points = P.size();

    Assert(n_of_points == n.size(),
           dealii::ExcDimensionMismatch(n_of_points, n.size()));
    Assert(n_of_points == p.size(),
           dealii::ExcDimensionMismatch(n_of_points, p.size()));
    Assert(n_of_points == r.size(),
           dealii::ExcDimensionMismatch(n_of_points, r.size()));

    if (this->temperature_buffer.size() != n_of_points)
      this->temperature_buffer.resize(n_of_points);

    if (clear_vector)
      for (std::size_t q = 0; q < n_of_points; q++)
        r[q] = 0.;

    this->temperature->value_list(P, this->temperature_buffer);

    for (std::size_t q = 0; q < n_of_points; q++)
      {
        const double ni = this->conduction_band_density *
                          this->valence_band_density *
                          exp((this->valence_band_edge_energy -
                               this->conduction_band_edge_energy) /
                              (Constants::KB * temperature_buffer[q]));
        r[q] += (this->n_coefficient * n[q] + this->p_coefficient * p[q]) *
                (n[q] * p[q] - ni);
      }
  }



  template <int dim>
  void
  Auger<dim>::compute_multiple_derivatives_of_recombination_terms(
    const std::vector<double> &            n,
    const std::vector<double> &            p,
    const std::vector<dealii::Point<dim>> &P,
    const Component                        c,
    bool                                   clear_vector,
    std::vector<double> &                  r)
  {
    const std::size_t n_of_points = P.size();

    Assert(n_of_points == n.size(),
           dealii::ExcDimensionMismatch(n_of_points, n.size()));
    Assert(n_of_points == p.size(),
           dealii::ExcDimensionMismatch(n_of_points, p.size()));
    Assert(n_of_points == r.size(),
           dealii::ExcDimensionMismatch(n_of_points, r.size()));

    if (this->temperature_buffer.size() != n_of_points)
      this->temperature_buffer.resize(n_of_points);

    if (clear_vector)
      for (std::size_t q = 0; q < n_of_points; q++)
        r[q] = 0.;

    const double C_n = this->n_coefficient;
    const double C_p = this->p_coefficient;

    double ni;

    switch (c)
      {
          case Component::n: {
            for (std::size_t q = 0; q < n_of_points; q++)
              {
                ni = this->conduction_band_density *
                     this->valence_band_density *
                     exp((this->valence_band_edge_energy -
                          this->conduction_band_edge_energy) /
                         (Constants::KB * temperature_buffer[q]));
                r[q] +=
                  C_p * p[q] * p[q] + 2 * C_n * n[q] * p[q] - C_n * ni * ni;
              }
            break;
          }
          case Component::p: {
            for (std::size_t q = 0; q < n_of_points; q++)
              {
                ni = this->conduction_band_density *
                     this->valence_band_density *
                     exp((this->valence_band_edge_energy -
                          this->conduction_band_edge_energy) /
                         (Constants::KB * temperature_buffer[q]));
                r[q] +=
                  C_n * n[q] * n[q] + 2 * C_n * n[q] * p[q] - C_p * ni * ni;
              }
            break;
          }
        default:
          Assert(false, InvalidComponent());
      }
  }



  template class RecombinationTerm<1>;
  template class RecombinationTerm<2>;
  template class RecombinationTerm<3>;

  template class LinearRecombinationTerm<1>;
  template class LinearRecombinationTerm<2>;
  template class LinearRecombinationTerm<3>;

  template class SuperimposedRecombinationTerm<1>;
  template class SuperimposedRecombinationTerm<2>;
  template class SuperimposedRecombinationTerm<3>;

  template class ShockleyReadHallFixedTemperature<1>;
  template class ShockleyReadHallFixedTemperature<2>;
  template class ShockleyReadHallFixedTemperature<3>;

  template class AugerFixedTemperature<1>;
  template class AugerFixedTemperature<2>;
  template class AugerFixedTemperature<3>;

  template class ShockleyReadHall<1>;
  template class ShockleyReadHall<2>;
  template class ShockleyReadHall<3>;

  template class Auger<1>;
  template class Auger<2>;
  template class Auger<3>;

} // namespace Ddhdg
