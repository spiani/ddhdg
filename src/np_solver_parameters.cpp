#include "np_solver_parameters.h"

namespace Ddhdg
{
  TauComputerType
  TauComputer::get_tau_computer_type() const
  {
    return TauComputerType::not_implemented;
  }



  FixedTauComputer::FixedTauComputer(
    const std::map<Component, double> &tau_vals,
    const Adimensionalizer            &adimensionalizer)
    : V_tau(tau_vals.at(Component::V))
    , n_tau(tau_vals.at(Component::n))
    , p_tau(tau_vals.at(Component::p))
    , V_rescaling_factor(
        adimensionalizer.get_tau_rescaling_factor<Component::V>())
    , n_rescaling_factor(
        adimensionalizer.get_tau_rescaling_factor<Component::n>())
    , p_rescaling_factor(
        adimensionalizer.get_tau_rescaling_factor<Component::p>())
    , V_tau_rescaled(this->V_tau / this->V_rescaling_factor)
    , n_tau_rescaled(this->n_tau / this->n_rescaling_factor)
    , p_tau_rescaled(this->p_tau / this->p_rescaling_factor)
  {}



  std::unique_ptr<TauComputer>
  FixedTauComputer::make_copy() const
  {
    return std::make_unique<FixedTauComputer>(*this);
  }



  TauComputerType
  FixedTauComputer::get_tau_computer_type() const
  {
    return TauComputerType::fixed_tau_computer;
  }



  template <int dim>
  void
  FixedTauComputer::compute_tau(
    const std::vector<dealii::Point<dim>> quadrature_points,
    const typename dealii::DoFHandler<dim, dim>::cell_iterator &cell,
    unsigned int                                                face,
    Component                                                   c,
    std::vector<double>                                        &tau) const
  {
    Assert(c == Component::V || c == Component::n || c == Component::p,
           InvalidComponent());

    if (c == Component::V)
      this->template compute_tau<dim, Component::V>(quadrature_points,
                                                    cell,
                                                    face,
                                                    tau);
    if (c == Component::n)
      this->template compute_tau<dim, Component::n>(quadrature_points,
                                                    cell,
                                                    face,
                                                    tau);
    if (c == Component::p)
      this->template compute_tau<dim, Component::p>(quadrature_points,
                                                    cell,
                                                    face,
                                                    tau);
  }



  CellFaceTauComputer::CellFaceTauComputer(
    std::shared_ptr<cell_face_tau_map> V_tau,
    std::shared_ptr<cell_face_tau_map> n_tau,
    std::shared_ptr<cell_face_tau_map> p_tau,
    const Adimensionalizer            &adimensionalizer)
    : V_tau(V_tau)
    , n_tau(n_tau)
    , p_tau(p_tau)
    , V_rescaling_factor(
        adimensionalizer.get_tau_rescaling_factor<Component::V>())
    , n_rescaling_factor(
        adimensionalizer.get_tau_rescaling_factor<Component::n>())
    , p_rescaling_factor(
        adimensionalizer.get_tau_rescaling_factor<Component::p>())
  {}



  std::unique_ptr<TauComputer>
  CellFaceTauComputer::make_copy() const
  {
    return std::make_unique<CellFaceTauComputer>(*this);
  }



  TauComputerType
  CellFaceTauComputer::get_tau_computer_type() const
  {
    return TauComputerType::cell_face_tau_computer;
  }



  template <int dim>
  void
  CellFaceTauComputer::compute_tau(
    const std::vector<dealii::Point<dim>> quadrature_points,
    const typename dealii::DoFHandler<dim, dim>::cell_iterator &cell,
    unsigned int                                                face,
    Component                                                   c,
    std::vector<double>                                        &tau)
  {
    Assert(c == Component::V || c == Component::n || c == Component::p,
           InvalidComponent());

    if (c == Component::V)
      this->template compute_tau<dim, Component::V>(quadrature_points,
                                                    cell,
                                                    face,
                                                    tau);
    if (c == Component::n)
      this->template compute_tau<dim, Component::n>(quadrature_points,
                                                    cell,
                                                    face,
                                                    tau);
    if (c == Component::p)
      this->template compute_tau<dim, Component::p>(quadrature_points,
                                                    cell,
                                                    face,
                                                    tau);
  }



  NPSolverParameters::NPSolverParameters(
    const unsigned int                               V_degree,
    const unsigned int                               n_degree,
    const unsigned int                               p_degree,
    const std::shared_ptr<NonlinearSolverParameters> nonlinear_parameters,
    const bool                                       iterative_linear_solver,
    const bool                                       multithreading,
    const DDFluxType                                 dd_flux_type,
    const bool                                       phi_linearize)
    : degree{{Component::V, V_degree},
             {Component::n, n_degree},
             {Component::p, p_degree}}
    , nonlinear_parameters(nonlinear_parameters)
    , iterative_linear_solver(iterative_linear_solver)
    , multithreading(multithreading)
    , dd_flux_type(dd_flux_type)
    , phi_linearize(phi_linearize)
  {}



  FixedTauNPSolverParameters::FixedTauNPSolverParameters(
    const unsigned int                               V_degree,
    const unsigned int                               n_degree,
    const unsigned int                               p_degree,
    const std::shared_ptr<NonlinearSolverParameters> nonlinear_parameters,
    const double                                     V_tau,
    const double                                     n_tau,
    const double                                     p_tau,
    const bool                                       iterative_linear_solver,
    const bool                                       multithreading,
    const DDFluxType                                 dd_flux_type,
    const bool                                       phi_linearize)
    : NPSolverParameters(V_degree,
                         n_degree,
                         p_degree,
                         nonlinear_parameters,
                         iterative_linear_solver,
                         multithreading,
                         dd_flux_type,
                         phi_linearize)
    , tau{{Component::V, V_tau}, {Component::n, n_tau}, {Component::p, p_tau}}
  {}



  TauComputerType
  NPSolverParameters::get_tau_computer_type() const
  {
    return TauComputerType::not_implemented;
  }



  std::unique_ptr<NPSolverParameters>
  FixedTauNPSolverParameters::make_unique_copy() const
  {
    return std::make_unique<FixedTauNPSolverParameters>(*this);
  }



  std::shared_ptr<NPSolverParameters>
  FixedTauNPSolverParameters::make_shared_copy() const
  {
    return std::make_shared<FixedTauNPSolverParameters>(*this);
  }



  std::unique_ptr<TauComputer>
  FixedTauNPSolverParameters::get_tau_computer(
    const Adimensionalizer &adimensionalizer) const
  {
    return std::make_unique<FixedTauComputer>(this->tau, adimensionalizer);
  }



  TauComputerType
  FixedTauNPSolverParameters::get_tau_computer_type() const
  {
    return TauComputerType::fixed_tau_computer;
  }



  double
  FixedTauNPSolverParameters::get_tau(const Component c) const
  {
    return this->tau.at(c);
  }



  CellFaceTauNPSolverParameters::CellFaceTauNPSolverParameters(
    const unsigned int                               V_degree,
    const unsigned int                               n_degree,
    const unsigned int                               p_degree,
    const std::shared_ptr<NonlinearSolverParameters> nonlinear_parameters,
    const bool                                       iterative_linear_solver,
    const bool                                       multithreading,
    const DDFluxType                                 dd_flux_type,
    const bool                                       phi_linearize)
    : NPSolverParameters(V_degree,
                         n_degree,
                         p_degree,
                         nonlinear_parameters,
                         iterative_linear_solver,
                         multithreading,
                         dd_flux_type,
                         phi_linearize)
    , V_tau(std::make_shared<cell_face_tau_map>())
    , n_tau(std::make_shared<cell_face_tau_map>())
    , p_tau(std::make_shared<cell_face_tau_map>())
  {}



  CellFaceTauNPSolverParameters::CellFaceTauNPSolverParameters(
    const CellFaceTauNPSolverParameters &solver_parameters)
    : NPSolverParameters(solver_parameters.degree.at(Component::V),
                         solver_parameters.degree.at(Component::n),
                         solver_parameters.degree.at(Component::p),
                         solver_parameters.nonlinear_parameters,
                         solver_parameters.iterative_linear_solver,
                         solver_parameters.multithreading,
                         solver_parameters.dd_flux_type,
                         solver_parameters.phi_linearize)
    , V_tau(std::make_shared<cell_face_tau_map>(*solver_parameters.V_tau))
    , n_tau(std::make_shared<cell_face_tau_map>(*solver_parameters.n_tau))
    , p_tau(std::make_shared<cell_face_tau_map>(*solver_parameters.p_tau))
  {
    std::cout << V_tau << std::endl;
  }



  std::unique_ptr<NPSolverParameters>
  CellFaceTauNPSolverParameters::make_unique_copy() const
  {
    return std::make_unique<CellFaceTauNPSolverParameters>(*this);
  }



  std::shared_ptr<NPSolverParameters>
  CellFaceTauNPSolverParameters::make_shared_copy() const
  {
    return std::make_shared<CellFaceTauNPSolverParameters>(*this);
  }



  std::unique_ptr<TauComputer>
  CellFaceTauNPSolverParameters::get_tau_computer(
    const Adimensionalizer &adimensionalizer) const
  {
    return std::make_unique<CellFaceTauComputer>(this->V_tau,
                                                 this->n_tau,
                                                 this->p_tau,
                                                 adimensionalizer);
  }



  TauComputerType
  CellFaceTauNPSolverParameters::get_tau_computer_type() const
  {
    return TauComputerType::cell_face_tau_computer;
  }



  void
  CellFaceTauNPSolverParameters::clear()
  {
    this->V_tau->clear();
    this->n_tau->clear();
    this->p_tau->clear();
  }



  void
  CellFaceTauNPSolverParameters::set_face(const unsigned int cell_level,
                                          const unsigned int cell_index,
                                          const unsigned int faces_per_cell,
                                          const unsigned int face,
                                          const double       face_V_tau,
                                          const double       face_n_tau,
                                          const double       face_p_tau)
  {
    std::pair<unsigned int, unsigned int> cell_id(cell_level, cell_index);

    Assert(
      face < faces_per_cell,
      dealii::ExcMessage(
        "The current index for the face is bigger than the number of faces "
        "per cell"));

    cell_face_tau_map &V_tau_map = *(this->V_tau);
    cell_face_tau_map &n_tau_map = *(this->n_tau);
    cell_face_tau_map &p_tau_map = *(this->p_tau);

    std::vector<double> *V_tau_cell = &(V_tau_map[cell_id]);
    std::vector<double> *n_tau_cell = &(n_tau_map[cell_id]);
    std::vector<double> *p_tau_cell = &(p_tau_map[cell_id]);

    if (V_tau_cell->size() != faces_per_cell)
      V_tau_cell->resize(faces_per_cell);
    if (n_tau_cell->size() != faces_per_cell)
      n_tau_cell->resize(faces_per_cell);
    if (p_tau_cell->size() != faces_per_cell)
      p_tau_cell->resize(faces_per_cell);

    (*V_tau_cell)[face] = face_V_tau;
    (*n_tau_cell)[face] = face_n_tau;
    (*p_tau_cell)[face] = face_p_tau;
  }



  template void
  FixedTauComputer::compute_tau<1>(
    const std::vector<dealii::Point<1>>                  quadrature_points,
    const typename dealii::DoFHandler<1>::cell_iterator &cell,
    unsigned int                                         face,
    Component                                            c,
    std::vector<double>                                 &tau) const;
  template void
  FixedTauComputer::compute_tau<2>(
    const std::vector<dealii::Point<2>>                  quadrature_points,
    const typename dealii::DoFHandler<2>::cell_iterator &cell,
    unsigned int                                         face,
    Component                                            c,
    std::vector<double>                                 &tau) const;
  template void
  FixedTauComputer::compute_tau<3>(
    const std::vector<dealii::Point<3>>                  quadrature_points,
    const typename dealii::DoFHandler<3>::cell_iterator &cell,
    unsigned int                                         face,
    Component                                            c,
    std::vector<double>                                 &tau) const;

  template void
  CellFaceTauComputer::compute_tau<1>(
    const std::vector<dealii::Point<1>>                  quadrature_points,
    const typename dealii::DoFHandler<1>::cell_iterator &cell,
    unsigned int                                         face,
    Component                                            c,
    std::vector<double>                                 &tau);
  template void
  CellFaceTauComputer::compute_tau<2>(
    const std::vector<dealii::Point<2>>                  quadrature_points,
    const typename dealii::DoFHandler<2>::cell_iterator &cell,
    unsigned int                                         face,
    Component                                            c,
    std::vector<double>                                 &tau);
  template void
  CellFaceTauComputer::compute_tau<3>(
    const std::vector<dealii::Point<3>>                  quadrature_points,
    const typename dealii::DoFHandler<3>::cell_iterator &cell,
    unsigned int                                         face,
    Component                                            c,
    std::vector<double>                                 &tau);

} // namespace Ddhdg
