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
      bool                                   clear_vector,
      std::vector<double> &                  r);

    virtual void
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<double> &            p,
      const std::vector<dealii::Point<dim>> &P,
      Component                              c,
      bool                                   clear_vector,
      std::vector<double> &                  r);

    virtual std::unique_ptr<RecombinationTerm<dim>>
    copy() const = 0;

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
      bool                                   clear_vector,
      std::vector<double> &                  r) override;

    void
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<double> &            p,
      const std::vector<dealii::Point<dim>> &P,
      Component                              c,
      bool                                   clear_vector,
      std::vector<double> &                  r) override;

    std::unique_ptr<RecombinationTerm<dim>>
    copy() const override
    {
      return std::make_unique<LinearRecombinationTerm<dim>>(*this);
    }

    virtual ~LinearRecombinationTerm() = default;

  private:
    std::shared_ptr<dealii::Function<dim>> constant_term;
    std::shared_ptr<dealii::Function<dim>> n_linear_coefficient;
    std::shared_ptr<dealii::Function<dim>> p_linear_coefficient;
  };

  template <int dim>
  class SuperimposedRecombinationTerm : public RecombinationTerm<dim>
  {
  public:
    explicit SuperimposedRecombinationTerm(
      const std::vector<std::shared_ptr<RecombinationTerm<dim>>>
        &recombination_terms);

    SuperimposedRecombinationTerm(
      std::shared_ptr<RecombinationTerm<dim>> recombination_term1,
      std::shared_ptr<RecombinationTerm<dim>> recombination_term2);

    SuperimposedRecombinationTerm(
      std::shared_ptr<RecombinationTerm<dim>> recombination_term1,
      std::shared_ptr<RecombinationTerm<dim>> recombination_term2,
      std::shared_ptr<RecombinationTerm<dim>> recombination_term3);

    SuperimposedRecombinationTerm(const SuperimposedRecombinationTerm<dim>
                                    &superimposed_recombination_term);

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
      bool                                   clear_vector,
      std::vector<double> &                  r) override;

    void
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<double> &            p,
      const std::vector<dealii::Point<dim>> &P,
      Component                              c,
      bool                                   clear_vector,
      std::vector<double> &                  r) override;

    std::unique_ptr<RecombinationTerm<dim>>
    copy() const override
    {
      return std::make_unique<SuperimposedRecombinationTerm<dim>>(*this);
    }

    virtual ~SuperimposedRecombinationTerm() = default;

  private:
    const std::vector<std::shared_ptr<RecombinationTerm<dim>>>
      recombination_terms;
  };

  template <int dim>
  class ShockleyReadHallFixedTemperature : public RecombinationTerm<dim>
  {
  public:
    static inline double
    compute_intrinsic_carrier_concentration(double conduction_band_density,
                                            double valence_band_density,
                                            double conduction_band_edge_energy,
                                            double valence_band_edge_energy,
                                            double temperature)
    {
      return conduction_band_density * valence_band_density *
             exp((valence_band_edge_energy - conduction_band_edge_energy) /
                 (Constants::KB * temperature));
    }

    ShockleyReadHallFixedTemperature(double intrinsic_carrier_concentration,
                                     double electron_life_time,
                                     double hole_life_time);

    ShockleyReadHallFixedTemperature(double conduction_band_density,
                                     double valence_band_density,
                                     double conduction_band_edge_energy,
                                     double valence_band_edge_energy,
                                     double temperature,
                                     double electron_life_time,
                                     double hole_life_time);

    ShockleyReadHallFixedTemperature(const ShockleyReadHallFixedTemperature<dim>
                                       &shockley_read_hall) = default;

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
      bool                                   clear_vector,
      std::vector<double> &                  r) override;

    void
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<double> &            p,
      const std::vector<dealii::Point<dim>> &P,
      Component                              c,
      bool                                   clear_vector,
      std::vector<double> &                  r) override;

    std::unique_ptr<RecombinationTerm<dim>>
    copy() const override
    {
      return std::make_unique<ShockleyReadHallFixedTemperature<dim>>(*this);
    }

    virtual ~ShockleyReadHallFixedTemperature() = default;

    const double intrinsic_carrier_concentration;
    const double electron_life_time;
    const double hole_life_time;
  };

  template <int dim>
  class AugerFixedTemperature : public RecombinationTerm<dim>
  {
  public:
    static inline double
    compute_intrinsic_carrier_concentration(double conduction_band_density,
                                            double valence_band_density,
                                            double conduction_band_edge_energy,
                                            double valence_band_edge_energy,
                                            double temperature)
    {
      return conduction_band_density * valence_band_density *
             exp((valence_band_edge_energy - conduction_band_edge_energy) /
                 (Constants::KB * temperature));
    }

    AugerFixedTemperature(double intrinsic_carrier_concentration,
                          double n_coefficient,
                          double p_coefficient);

    AugerFixedTemperature(double conduction_band_density,
                          double valence_band_density,
                          double conduction_band_edge_energy,
                          double valence_band_edge_energy,
                          double temperature,
                          double n_coefficient,
                          double p_coefficient);

    AugerFixedTemperature(const AugerFixedTemperature<dim> &auger) = default;

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
      bool                                   clear_vector,
      std::vector<double> &                  r) override;

    void
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<double> &            p,
      const std::vector<dealii::Point<dim>> &P,
      Component                              c,
      bool                                   clear_vector,
      std::vector<double> &                  r) override;

    std::unique_ptr<RecombinationTerm<dim>>
    copy() const override
    {
      return std::make_unique<AugerFixedTemperature<dim>>(*this);
    }

    virtual ~AugerFixedTemperature() = default;

    const double intrinsic_carrier_concentration;
    const double n_coefficient;
    const double p_coefficient;
  };


  template <int dim>
  class ShockleyReadHall : public RecombinationTerm<dim>
  {
  public:
    ShockleyReadHall(double conduction_band_density,
                     double valence_band_density,
                     double conduction_band_edge_energy,
                     double valence_band_edge_energy,
                     std::shared_ptr<dealii::Function<dim>> temperature,
                     double                                 electron_life_time,
                     double                                 hole_life_time);

    ShockleyReadHall(const ShockleyReadHall<dim> &shockley_read_hall) = default;

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
      bool                                   clear_vector,
      std::vector<double> &                  r) override;

    void
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<double> &            p,
      const std::vector<dealii::Point<dim>> &P,
      Component                              c,
      bool                                   clear_vector,
      std::vector<double> &                  r) override;

    std::unique_ptr<RecombinationTerm<dim>>
    copy() const override
    {
      return std::make_unique<ShockleyReadHall<dim>>(*this);
    }

    virtual ~ShockleyReadHall() = default;

    const double                                 conduction_band_density;
    const double                                 valence_band_density;
    const double                                 conduction_band_edge_energy;
    const double                                 valence_band_edge_energy;
    const std::shared_ptr<dealii::Function<dim>> temperature;

    const double electron_life_time;
    const double hole_life_time;

  private:
    std::vector<double> temperature_buffer;
  };

  template <int dim>
  class Auger : public RecombinationTerm<dim>
  {
  public:
    Auger(double                                 conduction_band_density,
          double                                 valence_band_density,
          double                                 conduction_band_edge_energy,
          double                                 valence_band_edge_energy,
          std::shared_ptr<dealii::Function<dim>> temperature,
          double                                 n_coefficient,
          double                                 p_coefficient);

    Auger(const Auger<dim> &auger) = default;

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
      bool                                   clear_vector,
      std::vector<double> &                  r) override;

    void
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double> &            n,
      const std::vector<double> &            p,
      const std::vector<dealii::Point<dim>> &P,
      Component                              c,
      bool                                   clear_vector,
      std::vector<double> &                  r) override;

    std::unique_ptr<RecombinationTerm<dim>>
    copy() const override
    {
      return std::make_unique<Auger<dim>>(*this);
    }

    virtual ~Auger() = default;

    const double                                 conduction_band_density;
    const double                                 valence_band_density;
    const double                                 conduction_band_edge_energy;
    const double                                 valence_band_edge_energy;
    const std::shared_ptr<dealii::Function<dim>> temperature;

    const double n_coefficient;
    const double p_coefficient;

  private:
    std::vector<double> temperature_buffer;
  };

} // namespace Ddhdg
