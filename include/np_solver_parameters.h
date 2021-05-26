#pragma once

#include <deal.II/base/numbers.h>

#include <deal.II/dofs/dof_handler.h>

#include <boost/container_hash/hash.hpp>

#include <map>
#include <memory>

#include "adimensionalizer.h"
#include "components.h"

namespace Ddhdg
{
  struct PairHash
  {
    std::size_t
    operator()(const std::pair<unsigned int, unsigned int> &k) const
    {
      std::size_t seed = 0;
      boost::hash_combine(seed, std::hash<unsigned int>()(k.first));
      boost::hash_combine(seed, std::hash<unsigned int>()(k.second));
      return seed;
    }
  };

  using cell_face_tau_map =
    std::unordered_map<std::pair<unsigned int, unsigned int>,
                       std::vector<double>,
                       PairHash>;

  enum DDFluxType
  {
    use_cell,
    use_trace,
    qiu_shi_stabilization
  };

  enum TauComputerType
  {
    not_implemented = -1,
    fixed_tau_computer,
    cell_face_tau_computer
  };

  struct NonlinearSolverParameters
  {
    explicit NonlinearSolverParameters(double absolute_tolerance       = 1e-10,
                                       double relative_tolerance       = 1e-10,
                                       int    max_number_of_iterations = 100,
                                       double alpha                    = 1.)
      : absolute_tolerance(absolute_tolerance)
      , relative_tolerance(relative_tolerance)
      , max_number_of_iterations(max_number_of_iterations)
      , alpha(alpha)
    {}

    double absolute_tolerance;
    double relative_tolerance;
    int    max_number_of_iterations;
    double alpha;
  };

  class TauComputer
  {
  public:
    virtual std::unique_ptr<TauComputer>
    make_copy() const = 0;

    virtual TauComputerType
    get_tau_computer_type() const;

    virtual ~TauComputer(){};
  };

  class FixedTauComputer : public TauComputer
  {
  public:
    FixedTauComputer(const std::map<Component, double> &tau_vals,
                     const Adimensionalizer &           adimensionalizer);

    FixedTauComputer(const FixedTauComputer &fixed_tau_computer) = default;

    std::unique_ptr<TauComputer>
    make_copy() const override;

    TauComputerType
    get_tau_computer_type() const override;

    template <int dim, Component c>
    inline void
    compute_tau(
      const std::vector<dealii::Point<dim>> quadrature_points,
      const typename dealii::DoFHandler<dim, dim>::cell_iterator &cell,
      unsigned int                                                face,
      std::vector<double> &                                       tau) const
    {
      Assert(c == Component::V || c == Component::n || c == Component::p,
             InvalidComponent());

      const unsigned int n_of_points = quadrature_points.size();

      (void)cell;
      (void)face;

      const double tau_val = (c == Component::V) ? this->V_tau_rescaled :
                             (c == Component::n) ? this->n_tau_rescaled :
                             (c == Component::p) ? this->p_tau_rescaled :
                                                   0.;

      for (unsigned int i = 0; i < n_of_points; ++i)
        tau[i] = tau_val;
    }

    template <int dim>
    void
    compute_tau(
      const std::vector<dealii::Point<dim>> quadrature_points,
      const typename dealii::DoFHandler<dim, dim>::cell_iterator &cell,
      unsigned int                                                face,
      Component                                                   c,
      std::vector<double> &                                       tau) const;

    const double V_tau;
    const double n_tau;
    const double p_tau;

    const double V_rescaling_factor;
    const double n_rescaling_factor;
    const double p_rescaling_factor;

    const double V_tau_rescaled;
    const double n_tau_rescaled;
    const double p_tau_rescaled;
  };

  class CellFaceTauComputer : public TauComputer
  {
  public:
    CellFaceTauComputer(std::shared_ptr<cell_face_tau_map> V_tau,
                        std::shared_ptr<cell_face_tau_map> n_tau,
                        std::shared_ptr<cell_face_tau_map> p_tau,
                        const Adimensionalizer &           adimensionalizer);

    CellFaceTauComputer(const CellFaceTauComputer &fixed_tau_computer) =
      default;

    std::unique_ptr<TauComputer>
    make_copy() const override;

    TauComputerType
    get_tau_computer_type() const override;

    template <int dim, Component c>
    inline void
    compute_tau(
      const std::vector<dealii::Point<dim>> quadrature_points,
      const typename dealii::DoFHandler<dim, dim>::cell_iterator &cell,
      unsigned int                                                face,
      std::vector<double> &                                       tau)
    {
      Assert(c == Component::V || c == Component::n || c == Component::p,
             InvalidComponent());

      const unsigned int n_of_points = quadrature_points.size();

      unsigned int                          level = cell->level();
      unsigned int                          index = cell->index();
      std::pair<unsigned int, unsigned int> cell_id(level, index);

      if constexpr (c == Component::V)
        {
          if (this->current_V_cell != cell_id)
            {
              this->current_V_cell = cell_id;
              this->V_tau_values   = &(this->V_tau->at(cell_id));
              Assert(this->V_tau_values->size() ==
                       dealii::GeometryInfo<dim>::faces_per_cell,
                     dealii::ExcMessage(
                       "Found a vector for V_tau that has not enough entries"));
            }

          const double rescaled_tau =
            (*(this->V_tau_values))[face] / this->V_rescaling_factor;
          for (unsigned int i = 0; i < n_of_points; ++i)
            tau[i] = rescaled_tau;
        }

      if constexpr (c == Component::n)
        {
          if (this->current_n_cell != cell_id)
            {
              this->current_n_cell = cell_id;
              this->n_tau_values   = &(this->n_tau->at(cell_id));
              Assert(this->n_tau_values->size() ==
                       dealii::GeometryInfo<dim>::faces_per_cell,
                     dealii::ExcMessage(
                       "Found a vector for n_tau that has not enough entries"));
            }

          const double rescaled_tau =
            (*(this->n_tau_values))[face] / this->n_rescaling_factor;
          for (unsigned int i = 0; i < n_of_points; ++i)
            tau[i] = rescaled_tau;
        }

      if constexpr (c == Component::p)
        {
          if (this->current_p_cell != cell_id)
            {
              this->current_p_cell = cell_id;
              this->p_tau_values   = &(this->p_tau->at(cell_id));
              Assert(this->p_tau_values->size() ==
                       dealii::GeometryInfo<dim>::faces_per_cell,
                     dealii::ExcMessage(
                       "Found a vector for p_tau that has not enough entries"));
            }

          const double rescaled_tau =
            (*(this->p_tau_values))[face] / this->p_rescaling_factor;
          for (unsigned int i = 0; i < n_of_points; ++i)
            tau[i] = rescaled_tau;
        }
    }

    template <int dim>
    void
    compute_tau(
      const std::vector<dealii::Point<dim>> quadrature_points,
      const typename dealii::DoFHandler<dim, dim>::cell_iterator &cell,
      unsigned int                                                face,
      Component                                                   c,
      std::vector<double> &                                       tau);

  private:
    std::shared_ptr<cell_face_tau_map> V_tau;
    std::shared_ptr<cell_face_tau_map> n_tau;
    std::shared_ptr<cell_face_tau_map> p_tau;

    double V_rescaling_factor;
    double n_rescaling_factor;
    double p_rescaling_factor;

    std::pair<unsigned int, unsigned int> current_V_cell = {
      dealii::numbers::invalid_unsigned_int,
      dealii::numbers::invalid_unsigned_int};
    std::pair<unsigned int, unsigned int> current_n_cell = {
      dealii::numbers::invalid_unsigned_int,
      dealii::numbers::invalid_unsigned_int};
    std::pair<unsigned int, unsigned int> current_p_cell = {
      dealii::numbers::invalid_unsigned_int,
      dealii::numbers::invalid_unsigned_int};

    std::vector<double> *V_tau_values;
    std::vector<double> *n_tau_values;
    std::vector<double> *p_tau_values;
  };

  class NPSolverParameters
  {
  public:
    explicit NPSolverParameters(
      unsigned int                               V_degree,
      unsigned int                               n_degree,
      unsigned int                               p_degree,
      std::shared_ptr<NonlinearSolverParameters> nonlinear_parameters,
      bool                                       iterative_linear_solver,
      bool                                       multithreading,
      DDFluxType                                 dd_flux_type,
      bool                                       phi_linearize);

    virtual std::unique_ptr<NPSolverParameters>
    make_unique_copy() const = 0;

    virtual std::shared_ptr<NPSolverParameters>
    make_shared_copy() const = 0;

    virtual std::unique_ptr<TauComputer>
    get_tau_computer(const Adimensionalizer &adimensionalizer) const = 0;

    virtual TauComputerType
    get_tau_computer_type() const;

    virtual ~NPSolverParameters()
    {}

    const std::map<Component, unsigned int> degree;

    const std::shared_ptr<NonlinearSolverParameters> nonlinear_parameters;

    bool       iterative_linear_solver;
    bool       multithreading;
    DDFluxType dd_flux_type;
    bool       phi_linearize;
  };

  class FixedTauNPSolverParameters : public NPSolverParameters
  {
  public:
    explicit FixedTauNPSolverParameters(
      unsigned int                               V_degree = 1,
      unsigned int                               n_degree = 1,
      unsigned int                               p_degree = 1,
      std::shared_ptr<NonlinearSolverParameters> nonlinear_parameters =
        std::make_shared<NonlinearSolverParameters>(),
      double     V_tau                   = 1.,
      double     n_tau                   = 1.,
      double     p_tau                   = 1.,
      bool       iterative_linear_solver = false,
      bool       multithreading          = true,
      DDFluxType dd_flux_type            = DDFluxType::use_cell,
      bool       phi_linearize           = false);

    std::unique_ptr<NPSolverParameters>
    make_unique_copy() const override;

    std::shared_ptr<NPSolverParameters>
    make_shared_copy() const override;

    std::unique_ptr<TauComputer>
    get_tau_computer(const Adimensionalizer &adimensionalizer) const override;

    TauComputerType
    get_tau_computer_type() const override;

    [[nodiscard]] double
    get_tau(Component c) const;

  private:
    const std::map<Component, double> tau;
  };

  class CellFaceTauNPSolverParameters : public NPSolverParameters
  {
  public:
    explicit CellFaceTauNPSolverParameters(
      unsigned int                               V_degree = 1,
      unsigned int                               n_degree = 1,
      unsigned int                               p_degree = 1,
      std::shared_ptr<NonlinearSolverParameters> nonlinear_parameters =
        std::make_shared<NonlinearSolverParameters>(),
      bool       iterative_linear_solver = false,
      bool       multithreading          = true,
      DDFluxType dd_flux_type            = DDFluxType::use_cell,
      bool       phi_linearize           = false);

    CellFaceTauNPSolverParameters(
      const CellFaceTauNPSolverParameters &solver_parameters);

    std::unique_ptr<NPSolverParameters>
    make_unique_copy() const override;

    std::shared_ptr<NPSolverParameters>
    make_shared_copy() const override;

    std::unique_ptr<TauComputer>
    get_tau_computer(const Adimensionalizer &adimensionalizer) const override;

    TauComputerType
    get_tau_computer_type() const override;

    void
    clear();

    void
    set_face(unsigned int cell_level,
             unsigned int cell_index,
             unsigned int faces_per_cell,
             unsigned int face,
             double       face_V_tau,
             double       face_n_tau,
             double       face_p_tau);

    template <int dim>
    void
    set_face(const typename dealii::CellAccessor<dim> &cell,
             unsigned int                              face,
             double                                    face_V_tau,
             double                                    face_n_tau,
             double                                    face_p_tau)
    {
      constexpr unsigned int faces_per_cell =
        dealii::GeometryInfo<dim>::faces_per_cell;

      unsigned int level = cell.level();
      unsigned int index = cell.index();

      Assert(
        face < faces_per_cell,
        dealii::ExcMessage(
          "The current index for the face is bigger than the number of faces "
          "per cell"));

      this->set_face(
        level, index, faces_per_cell, face, face_V_tau, face_n_tau, face_p_tau);
    }

  private:
    std::shared_ptr<cell_face_tau_map> V_tau;
    std::shared_ptr<cell_face_tau_map> n_tau;
    std::shared_ptr<cell_face_tau_map> p_tau;
  };

} // namespace Ddhdg
