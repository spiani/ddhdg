#include "pyddhdg/pyddhdg.h"

#include <stdexcept>

#include "function_tools.h"


namespace Ddhdg
{
  template <int dim>
  void
  PythonDefinedRecombinationTerm<dim>::redefine_function(
    const pybind11::object r_function,
    const pybind11::object dr_dn_function,
    const pybind11::object dr_dp_function)
  {
    this->r     = r_function;
    this->dr_dn = dr_dn_function;
    this->dr_dp = dr_dp_function;
  }



  template <int dim>
  double
  PythonDefinedRecombinationTerm<dim>::compute_recombination_term(
    double                    n,
    double                    p,
    const dealii::Point<dim> &q,
    double                    rescaling_factor) const
  {
    const unsigned int n_of_points = 1;

    double *points_data = new double[n_of_points * dim];
    for (unsigned int d = 0; d < dim; ++d)
      points_data[d] = q[d];

    pybind11::buffer_info points_buffer(
      points_data,
      sizeof(double), // itemsize
      pybind11::format_descriptor<double>::format(),
      1,                                  // ndim
      std::vector<size_t>{dim},           // shape
      std::vector<size_t>{sizeof(double)} // strides
    );

    pybind11::array_t<double, pybind11::array::c_style> points(points_buffer);
    pybind11::object                                    output_value =
      this->r(n * rescaling_factor, p * rescaling_factor, points);
    const double output_double = pybind11::cast<double>(output_value);

    delete[] points_data;

    return output_double;
  }



  template <int dim>
  double
  PythonDefinedRecombinationTerm<dim>::compute_derivative_of_recombination_term(
    double                    n,
    double                    p,
    const dealii::Point<dim> &q,
    double                    rescaling_factor,
    const Component           c) const
  {
    const unsigned int n_of_points = 1;

    double *points_data = new double[n_of_points * dim];
    for (unsigned int d = 0; d < dim; ++d)
      points_data[d] = q[d];

    pybind11::buffer_info points_buffer(
      points_data,
      sizeof(double), // itemsize
      pybind11::format_descriptor<double>::format(),
      1,                                  // ndim
      std::vector<size_t>{dim},           // shape
      std::vector<size_t>{sizeof(double)} // strides
    );

    pybind11::array_t<double, pybind11::array::c_style> points(points_buffer);

    if (c == Ddhdg::Component::n)
      {
        pybind11::object output_value =
          this->dr_dn(n * rescaling_factor, p * rescaling_factor, points);
        delete[] points_data;
        return pybind11::cast<double>(output_value);
      }
    else if (c == Ddhdg::Component::p)
      {
        pybind11::object output_value =
          this->dr_dp(n * rescaling_factor, p * rescaling_factor, points);
        delete[] points_data;
        return pybind11::cast<double>(output_value);
      }
    delete[] points_data;
    AssertThrow(false, dealii::ExcMessage("Invalid component"));
  }



  template <int dim>
  void
  PythonDefinedRecombinationTerm<dim>::compute_multiple_recombination_terms(
    const std::vector<double>             &n,
    const std::vector<double>             &p,
    const std::vector<dealii::Point<dim>> &P,
    double                                 rescaling_factor,
    bool                                   clear_vector,
    std::vector<double>                   &r)
  {
    const unsigned int n_of_points = P.size();

    double *points_data = new double[n_of_points * dim];
    for (unsigned int i = 0; i < n_of_points; ++i)
      for (unsigned int d = 0; d < dim; ++d)
        points_data[i * dim + d] = P[i][d];

    double *n_data = new double[n_of_points];
    double *p_data = new double[n_of_points];

    for (unsigned int i = 0; i < n_of_points; ++i)
      {
        n_data[i] = n[i] * rescaling_factor;
        p_data[i] = p[i] * rescaling_factor;
      }

    pybind11::buffer_info points_buffer(
      points_data,
      sizeof(double), // itemsize
      pybind11::format_descriptor<double>::format(),
      2,                                                        // ndim
      std::vector<size_t>{n_of_points, dim},                    // shape
      std::vector<size_t>{dim * sizeof(double), sizeof(double)} // strides
    );

    pybind11::buffer_info n_buffer(
      n_data,
      sizeof(double), // itemsize
      pybind11::format_descriptor<double>::format(),
      1,                                  // ndim
      std::vector<size_t>{n_of_points},   // shape
      std::vector<size_t>{sizeof(double)} // strides
    );

    pybind11::buffer_info p_buffer(
      p_data,
      sizeof(double), // itemsize
      pybind11::format_descriptor<double>::format(),
      1,                                  // ndim
      std::vector<size_t>{n_of_points},   // shape
      std::vector<size_t>{sizeof(double)} // strides
    );

    pybind11::array_t<double, pybind11::array::c_style> points(points_buffer);
    pybind11::array_t<double, pybind11::array::c_style> n_ndarray(n_buffer);
    pybind11::array_t<double, pybind11::array::c_style> p_ndarray(p_buffer);
    pybind11::array_t<double>                           output_value =
      this->r(n_ndarray, p_ndarray, points);

    pybind11::buffer_info output_buffer = output_value.request();
    if (output_buffer.ndim != 1)
      AssertThrow(false, ExcMessage("Number of dimensions must be one"));
    if (output_buffer.size != n_of_points)
      AssertThrow(
        false,
        ExcMessage(
          "Python function returned an array with an invalid number of points"));

    double *output_ptr = static_cast<double *>(output_buffer.ptr);

    if (clear_vector)
      for (unsigned int i = 0; i < n_of_points; ++i)
        r[i] = 0;

    for (unsigned int i = 0; i < n_of_points; ++i)
      r[i] += output_ptr[i];

    delete[] points_data;
    delete[] n_data;
    delete[] p_data;
  }



  template <int dim>
  void
  PythonDefinedRecombinationTerm<dim>::
    compute_multiple_derivatives_of_recombination_terms(
      const std::vector<double>             &n,
      const std::vector<double>             &p,
      const std::vector<dealii::Point<dim>> &P,
      const double                           rescaling_factor,
      const Component                        c,
      const bool                             clear_vector,
      std::vector<double>                   &r)
  {
    const unsigned int n_of_points = P.size();

    double *points_data = new double[n_of_points * dim];
    for (unsigned int i = 0; i < n_of_points; ++i)
      for (unsigned int d = 0; d < dim; ++d)
        points_data[i * dim + d] = P[i][d];

    double *n_data = new double[n_of_points];
    double *p_data = new double[n_of_points];

    for (unsigned int i = 0; i < n_of_points; ++i)
      {
        n_data[i] = n[i] * rescaling_factor;
        p_data[i] = p[i] * rescaling_factor;
      }

    pybind11::buffer_info points_buffer(
      points_data,
      sizeof(double), // itemsize
      pybind11::format_descriptor<double>::format(),
      2,                                                        // ndim
      std::vector<size_t>{n_of_points, dim},                    // shape
      std::vector<size_t>{dim * sizeof(double), sizeof(double)} // strides
    );

    pybind11::buffer_info n_buffer(
      n_data,
      sizeof(double), // itemsize
      pybind11::format_descriptor<double>::format(),
      1,                                  // ndim
      std::vector<size_t>{n_of_points},   // shape
      std::vector<size_t>{sizeof(double)} // strides
    );

    pybind11::buffer_info p_buffer(
      p_data,
      sizeof(double), // itemsize
      pybind11::format_descriptor<double>::format(),
      1,                                  // ndim
      std::vector<size_t>{n_of_points},   // shape
      std::vector<size_t>{sizeof(double)} // strides
    );

    pybind11::array_t<double, pybind11::array::c_style> points(points_buffer);
    pybind11::array_t<double, pybind11::array::c_style> n_ndarray(n_buffer);
    pybind11::array_t<double, pybind11::array::c_style> p_ndarray(p_buffer);
    pybind11::array_t<double>                           output_value;

    if (c == Ddhdg::Component::n)
      output_value = this->dr_dn(n_ndarray, p_ndarray, points);
    else if (c == Ddhdg::Component::p)
      output_value = this->dr_dp(n_ndarray, p_ndarray, points);
    else
      AssertThrow(false, ExcMessage("Invalid component specified"))

        pybind11::buffer_info output_buffer = output_value.request();
    if (output_buffer.ndim != 1)
      AssertThrow(false, ExcMessage("Number of dimensions must be one"));
    if (output_buffer.size != n_of_points)
      AssertThrow(
        false,
        ExcMessage(
          "Python function returned an array with an invalid number of points"));

    double *output_ptr = static_cast<double *>(output_buffer.ptr);

    if (clear_vector)
      for (unsigned int i = 0; i < n_of_points; ++i)
        r[i] = 0;

    for (unsigned int i = 0; i < n_of_points; ++i)
      r[i] += output_ptr[i];

    delete[] points_data;
    delete[] n_data;
    delete[] p_data;
  }



  template <int dim>
  double
  PythonDefinedSpacialRecombinationTerm<dim>::compute_recombination_term(
    double                    n,
    double                    p,
    const dealii::Point<dim> &q,
    double                    rescaling_factor) const
  {
    const unsigned int n_of_points = 1;

    double *points_data = new double[n_of_points * dim];
    for (unsigned int d = 0; d < dim; ++d)
      points_data[d] = q[d];

    pybind11::buffer_info points_buffer(
      points_data,
      sizeof(double), // itemsize
      pybind11::format_descriptor<double>::format(),
      1,                                  // ndim
      std::vector<size_t>{dim},           // shape
      std::vector<size_t>{sizeof(double)} // strides
    );

    pybind11::array_t<double, pybind11::array::c_style> points(points_buffer);

    (void)rescaling_factor;
    (void)n;
    (void)p;

    pybind11::object output_value  = this->r(points);
    const double     output_double = pybind11::cast<double>(output_value);

    delete[] points_data;

    return output_double;
  }



  template <int dim>
  void
  PythonDefinedSpacialRecombinationTerm<dim>::
    compute_multiple_recombination_terms(
      const std::vector<double>             &n,
      const std::vector<double>             &p,
      const std::vector<dealii::Point<dim>> &P,
      double                                 rescaling_factor,
      bool                                   clear_vector,
      std::vector<double>                   &r)
  {
    const unsigned int n_of_points = P.size();

    double *points_data = new double[n_of_points * dim];
    for (unsigned int i = 0; i < n_of_points; ++i)
      for (unsigned int d = 0; d < dim; ++d)
        points_data[i * dim + d] = P[i][d];

    pybind11::buffer_info points_buffer(
      points_data,
      sizeof(double), // itemsize
      pybind11::format_descriptor<double>::format(),
      2,                                                        // ndim
      std::vector<size_t>{n_of_points, dim},                    // shape
      std::vector<size_t>{dim * sizeof(double), sizeof(double)} // strides
    );

    (void)rescaling_factor;
    (void)n;
    (void)p;

    pybind11::array_t<double, pybind11::array::c_style> points(points_buffer);
    pybind11::array_t<double> output_value = this->r(points);

    pybind11::buffer_info output_buffer = output_value.request();
    if (output_buffer.ndim != 1)
      AssertThrow(false, ExcMessage("Number of dimensions must be one"));
    if (output_buffer.size != n_of_points)
      AssertThrow(
        false,
        ExcMessage(
          "Python function returned an array with an invalid number of points"));

    double *output_ptr = static_cast<double *>(output_buffer.ptr);

    if (clear_vector)
      for (unsigned int i = 0; i < n_of_points; ++i)
        r[i] = 0;

    for (unsigned int i = 0; i < n_of_points; ++i)
      r[i] += output_ptr[i];

    delete[] points_data;
  }



  template class PythonDefinedRecombinationTerm<1>;
  template class PythonDefinedRecombinationTerm<2>;
  template class PythonDefinedRecombinationTerm<3>;

  template class PythonDefinedSpacialRecombinationTerm<1>;
  template class PythonDefinedSpacialRecombinationTerm<2>;
  template class PythonDefinedSpacialRecombinationTerm<3>;
} // namespace Ddhdg


namespace pyddhdg
{
  template <int dim>
  HomogeneousPermittivity<dim>::HomogeneousPermittivity(const double epsilon)
    : epsilon(epsilon)
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::HomogeneousPermittivity<dim>>
  HomogeneousPermittivity<dim>::generate_ddhdg_permittivity()
  {
    return std::make_shared<Ddhdg::HomogeneousPermittivity<dim>>(this->epsilon);
  }



  template <int dim>
  HomogeneousMobility<dim>::HomogeneousMobility(const double           mu,
                                                const Ddhdg::Component c)
    : mu(mu)
    , cmp(c)
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::HomogeneousMobility<dim>>
  HomogeneousMobility<dim>::generate_ddhdg_mobility()
  {
    return std::make_shared<Ddhdg::HomogeneousMobility<dim>>(this->mu,
                                                             this->cmp);
  }



  template <int dim>
  DealIIFunction<dim>::DealIIFunction(
    const std::shared_ptr<dealii::Function<dim>> f)
    : f(f)
  {}


  template <int dim>
  DealIIFunction<dim>::DealIIFunction(const double f_const)
    : f((f_const == 0.) ?
          std::make_shared<dealii::Functions::ZeroFunction<dim>>() :
          std::make_shared<dealii::Functions::ConstantFunction<dim>>(f_const))
  {}



  template <int dim>
  std::shared_ptr<dealii::Function<dim>>
  DealIIFunction<dim>::get_dealii_function() const
  {
    return this->f;
  }



  template <int dim>
  std::shared_ptr<dealii::FunctionParser<dim>>
  AnalyticFunction<dim>::get_function_from_string(const std::string &f_expr)
  {
    const unsigned int n_of_components =
      dealii::Utilities::split_string_list(f_expr, ';').size();
    auto f = std::make_shared<dealii::FunctionParser<dim>>(n_of_components);
    f->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                  f_expr,
                  Ddhdg::Constants::constants);
    return f;
  }



  template <int dim>
  AnalyticFunction<dim>::AnalyticFunction(std::string f_expr)
    : DealIIFunction<dim>(get_function_from_string(f_expr))
    , f_expr(f_expr)
  {}



  template <int dim>
  std::string
  AnalyticFunction<dim>::get_expression() const
  {
    return this->f_expr;
  }



  template <int dim>
  PiecewiseFunction<dim>::PiecewiseFunction(
    const DealIIFunction<dim> &condition,
    const DealIIFunction<dim> &f1,
    const DealIIFunction<dim> &f2)
    : DealIIFunction<dim>(std::make_shared<Ddhdg::PiecewiseFunction<dim>>(
        condition.get_dealii_function(),
        f1.get_dealii_function(),
        f2.get_dealii_function()))
  {}



  template <int dim>
  PiecewiseFunction<dim>::PiecewiseFunction(const std::string &condition,
                                            const std::string &f1,
                                            const std::string &f2)
    : PiecewiseFunction(AnalyticFunction<dim>(condition),
                        AnalyticFunction<dim>(f1),
                        AnalyticFunction<dim>(f2))
  {}



  template <int dim>
  PiecewiseFunction<dim>::PiecewiseFunction(const std::string &condition,
                                            const std::string &f1,
                                            double             f2)
    : PiecewiseFunction(AnalyticFunction<dim>(condition),
                        AnalyticFunction<dim>(f1),
                        DealIIFunction<dim>(f2))
  {}



  template <int dim>
  PiecewiseFunction<dim>::PiecewiseFunction(const std::string &condition,
                                            double             f1,
                                            const std::string &f2)
    : PiecewiseFunction(AnalyticFunction<dim>(condition),
                        DealIIFunction<dim>(f1),
                        AnalyticFunction<dim>(f2))
  {}



  template <int dim>
  PiecewiseFunction<dim>::PiecewiseFunction(const std::string &condition,
                                            double             f1,
                                            double             f2)
    : PiecewiseFunction(AnalyticFunction<dim>(condition),
                        DealIIFunction<dim>(f1),
                        DealIIFunction<dim>(f2))
  {}



  template <int dim>
  LinearRecombinationTerm<dim>::LinearRecombinationTerm(
    const DealIIFunction<dim> &zero_term,
    const DealIIFunction<dim> &n_linear_coefficient,
    const DealIIFunction<dim> &p_linear_coefficient)
    : zero_term(zero_term)
    , n_linear_coefficient(n_linear_coefficient)
    , p_linear_coefficient(p_linear_coefficient)
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::RecombinationTerm<dim>>
  LinearRecombinationTerm<dim>::generate_ddhdg_recombination_term()
  {
    return std::make_shared<Ddhdg::LinearRecombinationTerm<dim>>(
      this->zero_term.get_dealii_function(),
      this->n_linear_coefficient.get_dealii_function(),
      this->p_linear_coefficient.get_dealii_function());
  }



  template <int dim>
  DealIIFunction<dim>
  LinearRecombinationTerm<dim>::get_constant_term() const
  {
    return this->zero_term;
  }



  template <int dim>
  DealIIFunction<dim>
  LinearRecombinationTerm<dim>::get_n_linear_coefficient() const
  {
    return this->p_linear_coefficient;
  }



  template <int dim>
  DealIIFunction<dim>
  LinearRecombinationTerm<dim>::get_p_linear_coefficient() const
  {
    return this->n_linear_coefficient;
  }



  template <int dim>
  ShockleyReadHallFixedTemperature<dim>::ShockleyReadHallFixedTemperature(
    const double intrinsic_carrier_concentration,
    const double electron_life_time,
    const double hole_life_time)
    : intrinsic_carrier_concentration(intrinsic_carrier_concentration)
    , electron_life_time(electron_life_time)
    , hole_life_time(hole_life_time)
  {}



  template <int dim>
  ShockleyReadHallFixedTemperature<dim>::ShockleyReadHallFixedTemperature(
    const double conduction_band_density,
    const double valence_band_density,
    const double conduction_band_edge_energy,
    const double valence_band_edge_energy,
    const double temperature,
    const double electron_life_time,
    const double hole_life_time)
    : intrinsic_carrier_concentration(
        Ddhdg::ShockleyReadHallFixedTemperature<dim>::
          compute_intrinsic_carrier_concentration(conduction_band_density,
                                                  valence_band_density,
                                                  conduction_band_edge_energy,
                                                  valence_band_edge_energy,
                                                  temperature))
    , electron_life_time(electron_life_time)
    , hole_life_time(hole_life_time)
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::RecombinationTerm<dim>>
  ShockleyReadHallFixedTemperature<dim>::generate_ddhdg_recombination_term()
  {
    return std::make_shared<Ddhdg::ShockleyReadHallFixedTemperature<dim>>(
      this->intrinsic_carrier_concentration,
      this->electron_life_time,
      this->hole_life_time);
  }



  template <int dim>
  AugerFixedTemperature<dim>::AugerFixedTemperature(
    const double intrinsic_carrier_concentration,
    const double n_coefficient,
    const double p_coefficient)
    : intrinsic_carrier_concentration(intrinsic_carrier_concentration)
    , n_coefficient(n_coefficient)
    , p_coefficient(p_coefficient)
  {}



  template <int dim>
  AugerFixedTemperature<dim>::AugerFixedTemperature(
    const double conduction_band_density,
    const double valence_band_density,
    const double conduction_band_edge_energy,
    const double valence_band_edge_energy,
    const double temperature,
    const double n_coefficient,
    const double p_coefficient)
    : intrinsic_carrier_concentration(
        Ddhdg::AugerFixedTemperature<dim>::
          compute_intrinsic_carrier_concentration(conduction_band_density,
                                                  valence_band_density,
                                                  conduction_band_edge_energy,
                                                  valence_band_edge_energy,
                                                  temperature))
    , n_coefficient(n_coefficient)
    , p_coefficient(p_coefficient)
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::RecombinationTerm<dim>>
  AugerFixedTemperature<dim>::generate_ddhdg_recombination_term()
  {
    return std::make_shared<Ddhdg::AugerFixedTemperature<dim>>(
      this->intrinsic_carrier_concentration,
      this->n_coefficient,
      this->p_coefficient);
  }



  template <int dim>
  ShockleyReadHall<dim>::ShockleyReadHall(
    const double              conduction_band_density,
    const double              valence_band_density,
    const double              conduction_band_edge_energy,
    const double              valence_band_edge_energy,
    const DealIIFunction<dim> temperature,
    const double              electron_life_time,
    const double              hole_life_time)
    : conduction_band_density(conduction_band_density)
    , valence_band_density(valence_band_density)
    , conduction_band_edge_energy(conduction_band_edge_energy)
    , valence_band_edge_energy(valence_band_edge_energy)
    , temperature(temperature)
    , electron_life_time(electron_life_time)
    , hole_life_time(hole_life_time)
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::RecombinationTerm<dim>>
  ShockleyReadHall<dim>::generate_ddhdg_recombination_term()
  {
    return std::make_shared<Ddhdg::ShockleyReadHall<dim>>(
      this->conduction_band_density,
      this->valence_band_density,
      this->conduction_band_edge_energy,
      this->valence_band_edge_energy,
      this->temperature.get_dealii_function(),
      this->electron_life_time,
      this->hole_life_time);
  }



  template <int dim>
  Auger<dim>::Auger(const double              conduction_band_density,
                    const double              valence_band_density,
                    const double              conduction_band_edge_energy,
                    const double              valence_band_edge_energy,
                    const DealIIFunction<dim> temperature,
                    const double              n_coefficient,
                    const double              p_coefficient)
    : conduction_band_density(conduction_band_density)
    , valence_band_density(valence_band_density)
    , conduction_band_edge_energy(conduction_band_edge_energy)
    , valence_band_edge_energy(valence_band_edge_energy)
    , temperature(temperature)
    , n_coefficient(n_coefficient)
    , p_coefficient(p_coefficient)
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::RecombinationTerm<dim>>
  Auger<dim>::generate_ddhdg_recombination_term()
  {
    return std::make_shared<Ddhdg::Auger<dim>>(
      this->conduction_band_density,
      this->valence_band_density,
      this->conduction_band_edge_energy,
      this->valence_band_edge_energy,
      this->temperature.get_dealii_function(),
      this->n_coefficient,
      this->p_coefficient);
  }



  template <int dim>
  pybind11::list
  SuperimposedRecombinationTerm<dim>::put_in_a_list(
    pybind11::object recombination_term1,
    pybind11::object recombination_term2)
  {
    auto l = pybind11::list();
    l.append(recombination_term1);
    l.append(recombination_term2);
    return l;
  }



  template <int dim>
  pybind11::list
  SuperimposedRecombinationTerm<dim>::put_in_a_list(
    pybind11::object recombination_term1,
    pybind11::object recombination_term2,
    pybind11::object recombination_term3)
  {
    auto l = pybind11::list();
    l.append(recombination_term1);
    l.append(recombination_term2);
    l.append(recombination_term3);
    return l;
  }



  template <int dim>
  SuperimposedRecombinationTerm<dim>::SuperimposedRecombinationTerm(
    const pybind11::list recombination_terms)
    : recombination_terms(recombination_terms)
  {}



  template <int dim>
  SuperimposedRecombinationTerm<dim>::SuperimposedRecombinationTerm(
    pybind11::object recombination_term1,
    pybind11::object recombination_term2)
    : recombination_terms(
        put_in_a_list(recombination_term1, recombination_term2))
  {}



  template <int dim>
  SuperimposedRecombinationTerm<dim>::SuperimposedRecombinationTerm(
    pybind11::object recombination_term1,
    pybind11::object recombination_term2,
    pybind11::object recombination_term3)
    : recombination_terms(put_in_a_list(recombination_term1,
                                        recombination_term2,
                                        recombination_term3))
  {}



  template <int dim>
  pybind11::list
  SuperimposedRecombinationTerm<dim>::get_recombination_terms() const
  {
    auto l = pybind11::list();
    for (pybind11::handle obj : this->recombination_terms)
      l.append(obj);
    return l;
  }



  template <int dim>
  std::shared_ptr<Ddhdg::RecombinationTerm<dim>>
  SuperimposedRecombinationTerm<dim>::generate_ddhdg_recombination_term()
  {
    std::vector<std::shared_ptr<Ddhdg::RecombinationTerm<dim>>>
      ddhdg_recombination_terms;
    for (pybind11::handle obj : this->recombination_terms)
      {
        const auto recombination_term = obj.cast<RecombinationTerm<dim> *>()
                                          ->generate_ddhdg_recombination_term();
        ddhdg_recombination_terms.push_back(recombination_term);
      }
    return std::make_shared<Ddhdg::SuperimposedRecombinationTerm<dim>>(
      ddhdg_recombination_terms);
  }



  template <int dim>
  BoundaryConditionHandler<dim>::BoundaryConditionHandler()
    : bc_handler(std::make_shared<Ddhdg::BoundaryConditionHandler<dim>>())
  {}



  template <int dim>
  std::shared_ptr<Ddhdg::BoundaryConditionHandler<dim>>
  BoundaryConditionHandler<dim>::get_ddhdg_boundary_condition_handler()
  {
    return this->bc_handler;
  }



  template <int dim>
  void
  BoundaryConditionHandler<dim>::add_boundary_condition(
    const dealii::types::boundary_id   id,
    const Ddhdg::BoundaryConditionType bc_type,
    const Ddhdg::Component             c,
    const DealIIFunction<dim>         &f)
  {
    this->bc_handler->add_boundary_condition(id,
                                             bc_type,
                                             c,
                                             f.get_dealii_function());
  }



  template <int dim>
  void
  BoundaryConditionHandler<dim>::add_boundary_condition(
    const dealii::types::boundary_id   id,
    const Ddhdg::BoundaryConditionType bc_type,
    const Ddhdg::Component             c,
    const std::string                 &f)
  {
    this->add_boundary_condition(id, bc_type, c, AnalyticFunction<dim>(f));
  }



  template <int dim>
  void
  BoundaryConditionHandler<dim>::add_boundary_condition(
    const dealii::types::boundary_id   id,
    const Ddhdg::BoundaryConditionType bc_type,
    const Ddhdg::Component             c,
    const double                       d)
  {
    this->add_boundary_condition(id, bc_type, c, DealIIFunction<dim>(d));
  }



  template <int dim>
  void
  BoundaryConditionHandler<dim>::replace_boundary_condition(
    const dealii::types::boundary_id   id,
    const Ddhdg::BoundaryConditionType bc_type,
    const Ddhdg::Component             c,
    const DealIIFunction<dim>         &f)
  {
    this->bc_handler->replace_boundary_condition(id,
                                                 bc_type,
                                                 c,
                                                 f.get_dealii_function());
  }



  template <int dim>
  void
  BoundaryConditionHandler<dim>::replace_boundary_condition(
    const dealii::types::boundary_id   id,
    const Ddhdg::BoundaryConditionType bc_type,
    const Ddhdg::Component             c,
    const std::string                 &f)
  {
    this->replace_boundary_condition(id, bc_type, c, AnalyticFunction<dim>(f));
  }



  template <int dim>
  void
  BoundaryConditionHandler<dim>::replace_boundary_condition(
    const dealii::types::boundary_id   id,
    const Ddhdg::BoundaryConditionType bc_type,
    const Ddhdg::Component             c,
    const double                       d)
  {
    this->replace_boundary_condition(id, bc_type, c, DealIIFunction<dim>(d));
  }



  template <int dim>
  bool
  BoundaryConditionHandler<dim>::has_dirichlet_boundary_conditions() const
  {
    return this->bc_handler->has_dirichlet_boundary_conditions();
  }



  template <int dim>
  bool
  BoundaryConditionHandler<dim>::has_neumann_boundary_conditions() const
  {
    return this->bc_handler->has_neumann_boundary_conditions();
  }



  template <int dim>
  Problem<dim>::Problem(const double                   left,
                        const double                   right,
                        HomogeneousPermittivity<dim>  &permittivity,
                        HomogeneousMobility<dim>      &electron_mobility,
                        HomogeneousMobility<dim>      &hole_mobility,
                        RecombinationTerm<dim>        &recombination_term,
                        DealIIFunction<dim>           &temperature,
                        DealIIFunction<dim>           &doping,
                        BoundaryConditionHandler<dim> &bc_handler,
                        const double                   conduction_band_density,
                        const double                   valence_band_density,
                        const double conduction_band_edge_energy,
                        const double valence_band_edge_energy)
    : ddhdg_problem(std::make_shared<Ddhdg::HomogeneousProblem<dim>>(
        generate_triangulation(left, right),
        permittivity.generate_ddhdg_permittivity(),
        electron_mobility.generate_ddhdg_mobility(),
        hole_mobility.generate_ddhdg_mobility(),
        recombination_term.generate_ddhdg_recombination_term(),
        temperature.get_dealii_function(),
        doping.get_dealii_function(),
        bc_handler.get_ddhdg_boundary_condition_handler(),
        conduction_band_density,
        valence_band_density,
        conduction_band_edge_energy,
        valence_band_edge_energy))
  {}


  template <int dim>
  Problem<dim>::Problem(
    const dealii::python::TriangulationWrapper &triangulation,
    HomogeneousPermittivity<dim>               &permittivity,
    HomogeneousMobility<dim>                   &electron_mobility,
    HomogeneousMobility<dim>                   &hole_mobility,
    RecombinationTerm<dim>                     &recombination_term,
    DealIIFunction<dim>                        &temperature,
    DealIIFunction<dim>                        &doping,
    BoundaryConditionHandler<dim>              &bc_handler,
    const double                                conduction_band_density,
    const double                                valence_band_density,
    const double                                conduction_band_edge_energy,
    const double                                valence_band_edge_energy)
    : ddhdg_problem(std::make_shared<Ddhdg::HomogeneousProblem<dim>>(
        copy_triangulation(triangulation),
        permittivity.generate_ddhdg_permittivity(),
        electron_mobility.generate_ddhdg_mobility(),
        hole_mobility.generate_ddhdg_mobility(),
        recombination_term.generate_ddhdg_recombination_term(),
        temperature.get_dealii_function(),
        doping.get_dealii_function(),
        bc_handler.get_ddhdg_boundary_condition_handler(),
        conduction_band_density,
        valence_band_density,
        conduction_band_edge_energy,
        valence_band_edge_energy))
  {}



  template <int dim>
  Problem<dim>::Problem(const Problem<dim> &problem)
    : ddhdg_problem(problem.ddhdg_problem)
  {}



  template <int dim>
  std::shared_ptr<const Ddhdg::HomogeneousProblem<dim>>
  Problem<dim>::get_ddhdg_problem() const
  {
    return this->ddhdg_problem;
  }



  template <int dim>
  std::shared_ptr<dealii::Triangulation<dim>>
  Problem<dim>::generate_triangulation(const double left, const double right)
  {
    std::shared_ptr<dealii::Triangulation<dim>> triangulation =
      std::make_shared<dealii::Triangulation<dim>>();

    dealii::Point<dim>        p1, p2;
    std::vector<unsigned int> subdivisions(dim);
    for (unsigned int i = 0; i < dim; ++i)
      {
        p1[i]           = left;
        p2[i]           = right;
        subdivisions[i] = 1;
      }

    dealii::GridGenerator::subdivided_hyper_rectangle(
      *triangulation, subdivisions, p1, p2, true);

    return triangulation;
  }



  template <int dim>
  std::shared_ptr<dealii::Triangulation<dim>>
  Problem<dim>::copy_triangulation(
    const dealii::python::TriangulationWrapper &triangulation)
  {
    auto problem_triangulation = std::make_shared<dealii::Triangulation<dim>>();
    const int triang_dim       = triangulation.get_dim();
    const int triang_spacedim  = triangulation.get_spacedim();
    AssertThrow(triang_dim == dim,
                dealii::ExcMessage(
                  "The dimension of the triangulation is different from the "
                  "dimension of the problem"));
    AssertThrow(
      triang_spacedim == dim,
      dealii::ExcMessage(
        "The dimension of the space of the triangulation is different from the "
        "dimension of the problem"));

    const auto triang_raw_pointer = static_cast<dealii::Triangulation<dim> *>(
      triangulation.get_triangulation());

    problem_triangulation->copy_triangulation(*triang_raw_pointer);

    return problem_triangulation;
  }



  ErrorPerCell::ErrorPerCell(const unsigned int size)
  {
    this->data_vector = std::make_shared<dealii::Vector<float>>(size);
  }



  ErrorPerCell::ErrorPerCell(const ErrorPerCell &other)
  {
    this->data_vector = other.data_vector;
  }



  template <int dim>
  NPSolver<dim>::NPSolver(
    const Problem<dim>                              &problem,
    const std::shared_ptr<Ddhdg::NPSolverParameters> parameters,
    const Ddhdg::Adimensionalizer                   &adimensionalizer,
    const bool                                       verbose)
    : ddhdg_solver(
        std::make_shared<Ddhdg::NPSolver<dim, Ddhdg::HomogeneousProblem<dim>>>(
          problem.get_ddhdg_problem(),
          parameters,
          std::make_shared<const Ddhdg::Adimensionalizer>(adimensionalizer),
          verbose))
  {}



  template <int dim>
  void
  NPSolver<dim>::set_verbose(const bool verbose)
  {
    if (verbose)
      this->ddhdg_solver->log_standard_level =
        Ddhdg::Logging::severity_level::info;
    else
      this->ddhdg_solver->log_standard_level =
        Ddhdg::Logging::severity_level::debug;
  }



  template <int dim>
  void
  NPSolver<dim>::refine_grid(const unsigned int i, const bool preserve_solution)
  {
    this->ddhdg_solver->refine_grid(i, preserve_solution);
  }



  template <int dim>
  void
  NPSolver<dim>::refine_and_coarsen_fixed_fraction(
    const ErrorPerCell error_per_cell,
    const double       top_fraction,
    const double       bottom_fraction,
    const unsigned int max_n_cells)
  {
    this->ddhdg_solver->refine_and_coarsen_fixed_fraction(
      *(error_per_cell.data_vector),
      top_fraction,
      bottom_fraction,
      max_n_cells);
  }



  template <int dim>
  unsigned int
  NPSolver<dim>::n_of_triangulation_levels() const
  {
    return this->ddhdg_solver->n_of_triangulation_levels();
  }


  template <int dim>
  unsigned int
  NPSolver<dim>::get_n_dofs(const bool for_trace) const
  {
    return this->ddhdg_solver->get_n_dofs(for_trace);
  }


  template <int dim>
  unsigned int
  NPSolver<dim>::get_n_active_cells() const
  {
    return this->ddhdg_solver->get_n_active_cells();
  }


  template <int dim>
  void
  NPSolver<dim>::get_cell_vertices(double vertices[]) const
  {
    const unsigned int vertices_per_cell =
      dealii::GeometryInfo<dim>::vertices_per_cell;
    dealii::Point<dim> *p;
    unsigned int        cell_number = 0;
    for (const auto &cell :
         this->ddhdg_solver->dof_handler_cell.active_cell_iterators())
      {
        for (unsigned int v = 0; v < vertices_per_cell; v++)
          {
            p = &(cell->vertex(v));
            for (unsigned int i = 0; i < dim; i++)
              {
                const double       p_i = (*p)[i];
                const unsigned int k =
                  cell_number * (vertices_per_cell * dim) + v * dim + i;
                vertices[k] = p_i;
              }
          }
        ++cell_number;
      }
  }



  template <int dim>
  void
  NPSolver<dim>::set_component(const Ddhdg::Component c,
                               const std::string     &f,
                               const bool             use_projection)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> c_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    c_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      f,
      Ddhdg::Constants::constants);
    this->ddhdg_solver->set_component(c, c_function, use_projection);
  }



  template <int dim>
  void
  NPSolver<dim>::set_component(const Ddhdg::Component    c,
                               const DealIIFunction<dim> f,
                               const bool                use_projection)
  {
    this->ddhdg_solver->set_component(c,
                                      f.get_dealii_function(),
                                      use_projection);
  }



  template <int dim>
  void
  NPSolver<dim>::set_current_solution(const std::string &v_f,
                                      const std::string &n_f,
                                      const std::string &p_f,
                                      const bool         use_projection)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> v_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> n_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> p_function =
      std::make_shared<dealii::FunctionParser<dim>>();
    v_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      v_f,
      Ddhdg::Constants::constants);
    n_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      n_f,
      Ddhdg::Constants::constants);
    p_function->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      p_f,
      Ddhdg::Constants::constants);
    this->ddhdg_solver->set_current_solution(v_function,
                                             n_function,
                                             p_function,
                                             use_projection);
  }



  template <int dim>
  void
  NPSolver<dim>::set_multithreading(const bool multithreading)
  {
    this->ddhdg_solver->set_multithreading(multithreading);
  }



  template <int dim>
  bool
  NPSolver<dim>::is_enabled(Ddhdg::Component c) const
  {
    return this->ddhdg_solver->is_enabled(c);
  }



  template <int dim>
  void
  NPSolver<dim>::enable_component(Ddhdg::Component c)
  {
    this->ddhdg_solver->enable_component(c);
  }



  template <int dim>
  void
  NPSolver<dim>::disable_component(Ddhdg::Component c)
  {
    this->ddhdg_solver->disable_component(c);
  }



  template <int dim>
  void
  NPSolver<dim>::set_enabled_components(const bool V_enabled,
                                        const bool n_enabled,
                                        const bool p_enabled)
  {
    this->ddhdg_solver->set_enabled_components(V_enabled, n_enabled, p_enabled);
  }



  template <int dim>
  void
  NPSolver<dim>::copy_triangulation_from(NPSolver<dim> other)
  {
    this->ddhdg_solver->copy_triangulation_from(*(other.ddhdg_solver));
  }



  template <int dim>
  void
  NPSolver<dim>::copy_solution_from(NPSolver<dim> other)
  {
    this->ddhdg_solver->copy_solution_from(*(other.ddhdg_solver));
  }



  template <int dim>
  std::shared_ptr<Ddhdg::NPSolverParameters>
  NPSolver<dim>::get_parameters() const
  {
    return this->ddhdg_solver->parameters;
  }



  template <int dim>
  void
  NPSolver<dim>::assemble_system()
  {
    if (!this->ddhdg_solver->initialized)
      this->ddhdg_solver->setup_overall_system();

    this->ddhdg_solver->system_matrix = 0;
    this->ddhdg_solver->system_rhs    = 0;

    this->ddhdg_solver->assemble_system(
      false, false, this->ddhdg_solver->parameters->multithreading);
  }



  template <int dim>
  std::map<Ddhdg::Component,
           std::pair<pybind11::array_t<unsigned int, pybind11::array::c_style>,
                     pybind11::array_t<double, pybind11::array::c_style>>>
  NPSolver<dim>::get_dirichlet_boundary_dofs()
  {
    if (!this->ddhdg_solver->initialized)
      this->ddhdg_solver->setup_overall_system();

    const unsigned int n_dofs =
      this->ddhdg_solver->dof_handler_trace_restricted.n_dofs();

    std::vector<Ddhdg::Component> dof_to_component_map(n_dofs);
    std::vector<Ddhdg::DofType>   dof_to_dof_type_map(n_dofs);

    this->ddhdg_solver->generate_dof_to_component_map(dof_to_component_map,
                                                      dof_to_dof_type_map,
                                                      true,
                                                      true);

    // Now we create a map that associates to each global dof its index for
    // the current component
    std::vector<dealii::types::global_dof_index> dof_to_local_dof_map(
      n_dofs, dealii::numbers::invalid_dof_index);

    // This one, instead, associates for each component its number of dofs
    std::map<Ddhdg::Component, unsigned int> n_of_component_dofs;

    // Let us fill the previous maps
    for (const auto c : this->ddhdg_solver->enabled_components)
      {
        unsigned int n_of_current_component_dofs = 0;
        for (unsigned int i = 0; i < n_dofs; ++i)
          if (dof_to_component_map[i] == c)
            dof_to_local_dof_map[i] = n_of_current_component_dofs++;

        Assert(n_of_current_component_dofs > 0,
               dealii::ExcMessage("No dofs for current component"));

        n_of_component_dofs.insert({c, n_of_current_component_dofs});
      }

    std::vector<bool>  dirichlet_constraints(n_dofs, false);
    const unsigned int n_dirichlet_constraints =
      this->ddhdg_solver->get_dofs_constrained_by_dirichlet_conditions(
        dirichlet_constraints);

    // This map is the final output
    std::map<
      Ddhdg::Component,
      std::pair<pybind11::array_t<unsigned int, pybind11::array::c_style>,
                pybind11::array_t<double, pybind11::array::c_style>>>
      dirichlet_bc_dofs;

    // Check if the values have been computed or not
    if (n_dirichlet_constraints > 0 and
        this->ddhdg_solver->constrained_dof_indices[0] ==
          dealii::numbers::invalid_dof_index)
      {
        throw std::runtime_error(
          "Dirichlet boundary conditions have not been computed yet!");
      }

    for (const auto c : this->ddhdg_solver->enabled_components)
      {
        // Now we count the dofs constrained by Dirichlet boundary conditions
        // on the current component
        unsigned int constrained_dofs = 0;
        for (unsigned int i = 0; i < n_dofs; ++i)
          if (dirichlet_constraints[i] && dof_to_component_map[i] == c)
            ++constrained_dofs;

        auto *component_dirichlet_bc_indices =
          new unsigned int[constrained_dofs];
        auto *component_dirichlet_bc_values = new double[constrained_dofs];

        unsigned int k = 0;
        for (unsigned int i = 0; i < n_dirichlet_constraints; ++i)
          {
            dealii::types::global_dof_index current_index =
              this->ddhdg_solver->constrained_dof_indices[i];
            if (dof_to_component_map[current_index] == c)
              {
                const unsigned int local_dof_index =
                  dof_to_local_dof_map[current_index];
                component_dirichlet_bc_indices[k] = local_dof_index;
                component_dirichlet_bc_values[k] =
                  this->ddhdg_solver->constrained_dof_values[i];
                ++k;
              }
          }

        Assert(
          k == constrained_dofs,
          dealii::ExcMessage(
            "Copied a number of elements that is different from constrained_dofs"));

        const auto vector_shape         = std::vector<long>{constrained_dofs};
        const auto vector_stride_index  = std::vector<long>{4};
        const auto vector_stride_values = std::vector<long>{8};

        pybind11::capsule free_indices(component_dirichlet_bc_indices,
                                       [](void *f) {
                                         auto *data =
                                           reinterpret_cast<unsigned int *>(f);
                                         delete[] data;
                                       });
        pybind11::capsule free_values(component_dirichlet_bc_values,
                                      [](void *f) {
                                        auto *data =
                                          reinterpret_cast<double *>(f);
                                        delete[] data;
                                      });

        auto indices =
          pybind11::array_t<unsigned int, pybind11::array::c_style>(
            vector_shape,
            vector_stride_index,
            component_dirichlet_bc_indices,
            free_indices);
        auto values = pybind11::array_t<double, pybind11::array::c_style>(
          vector_shape,
          vector_stride_values,
          component_dirichlet_bc_values,
          free_values);

        dirichlet_bc_dofs.insert({c, {indices, values}});
      }

    return dirichlet_bc_dofs;
  }



  template <int dim>
  std::map<Ddhdg::Component,
           pybind11::array_t<double, pybind11::array::c_style>>
  NPSolver<dim>::get_residual()
  {
    if (!this->ddhdg_solver->initialized)
      this->ddhdg_solver->setup_overall_system();

    std::map<Ddhdg::Component,
             pybind11::array_t<double, pybind11::array::c_style>>
      residual_map;

    const unsigned int n_dofs =
      this->ddhdg_solver->dof_handler_trace_restricted.n_dofs();

    std::vector<Ddhdg::Component> dof_to_component_map(n_dofs);
    std::vector<Ddhdg::DofType>   dof_to_dof_type_map(n_dofs);

    this->ddhdg_solver->generate_dof_to_component_map(dof_to_component_map,
                                                      dof_to_dof_type_map,
                                                      true,
                                                      true);
    for (const auto c : this->ddhdg_solver->enabled_components)
      {
        unsigned int n_of_component_dofs = 0;
        for (unsigned int i = 0; i < n_dofs; ++i)
          if (dof_to_component_map[i] == c)
            n_of_component_dofs += 1;

        Assert(n_of_component_dofs > 0,
               dealii::ExcMessage("No dofs for current component"));

        auto *component_residual_data = new double[n_of_component_dofs];

        unsigned int k = 0;
        for (unsigned int i = 0; i < n_dofs; ++i)
          if (dof_to_component_map[i] == c)
            component_residual_data[k++] = this->ddhdg_solver->system_rhs[i];

        Assert(
          k == n_of_component_dofs,
          dealii::ExcMessage(
            "Copied a number of elements that is different from n_of_component_dofs"));

        const auto vector_shape  = std::vector<long>{n_of_component_dofs};
        const auto vector_stride = std::vector<long>{8};

        // Create a Python object that will free the allocated
        // memory when destroyed:
        pybind11::capsule free_when_done(component_residual_data, [](void *f) {
          auto *data = reinterpret_cast<double *>(f);
          delete[] data;
        });

        auto component_residual =
          pybind11::array_t<double, pybind11::array::c_style>(
            vector_shape,
            vector_stride,
            component_residual_data,
            free_when_done);

        residual_map.insert({c, component_residual});
      }

    return residual_map;
  }



  template <int dim>
  std::map<Ddhdg::Component,
           pybind11::array_t<double, pybind11::array::c_style>>
  NPSolver<dim>::get_linear_system_solution_vector()
  {
    if (!this->ddhdg_solver->initialized)
      this->ddhdg_solver->setup_overall_system();

    std::map<Ddhdg::Component,
             pybind11::array_t<double, pybind11::array::c_style>>
      ls_solution_map;

    const unsigned int n_dofs =
      this->ddhdg_solver->dof_handler_trace_restricted.n_dofs();

    std::vector<Ddhdg::Component> dof_to_component_map(n_dofs);
    std::vector<Ddhdg::DofType>   dof_to_dof_type_map(n_dofs);

    this->ddhdg_solver->generate_dof_to_component_map(dof_to_component_map,
                                                      dof_to_dof_type_map,
                                                      true,
                                                      true);
    for (const auto c : this->ddhdg_solver->enabled_components)
      {
        unsigned int n_of_component_dofs = 0;
        for (unsigned int i = 0; i < n_dofs; ++i)
          if (dof_to_component_map[i] == c)
            n_of_component_dofs += 1;

        Assert(n_of_component_dofs > 0,
               dealii::ExcMessage("No dofs for current component"));

        auto *component_ls_solution_data = new double[n_of_component_dofs];

        unsigned int k = 0;
        for (unsigned int i = 0; i < n_dofs; ++i)
          if (dof_to_component_map[i] == c)
            component_ls_solution_data[k++] =
              this->ddhdg_solver->system_solution[i];

        Assert(
          k == n_of_component_dofs,
          dealii::ExcMessage(
            "Copied a number of elements that is different from n_of_component_dofs"));

        const auto vector_shape  = std::vector<long>{n_of_component_dofs};
        const auto vector_stride = std::vector<long>{8};

        // Create a Python object that will free the allocated
        // memory when destroyed:
        pybind11::capsule free_when_done(component_ls_solution_data,
                                         [](void *f) {
                                           auto *data =
                                             reinterpret_cast<double *>(f);
                                           delete[] data;
                                         });

        auto component_ls_solution =
          pybind11::array_t<double, pybind11::array::c_style>(
            vector_shape,
            vector_stride,
            component_ls_solution_data,
            free_when_done);

        ls_solution_map.insert({c, component_ls_solution});
      }

    return ls_solution_map;
  }



  template <int dim>
  std::map<Ddhdg::Component,
           pybind11::array_t<double, pybind11::array::c_style>>
  NPSolver<dim>::get_current_trace_vector()
  {
    if (!this->ddhdg_solver->initialized)
      this->ddhdg_solver->setup_overall_system();

    std::map<Ddhdg::Component,
             pybind11::array_t<double, pybind11::array::c_style>>
      trace_vector_map;

    const unsigned int n_dofs = this->ddhdg_solver->dof_handler_trace.n_dofs();

    std::vector<Ddhdg::Component> dof_to_component_map(n_dofs);
    std::vector<Ddhdg::DofType>   dof_to_dof_type_map(n_dofs);

    this->ddhdg_solver->generate_dof_to_component_map(dof_to_component_map,
                                                      dof_to_dof_type_map,
                                                      true,
                                                      false);
    for (const auto c : Ddhdg::all_primary_components())
      {
        unsigned int n_of_component_dofs = 0;
        for (unsigned int i = 0; i < n_dofs; ++i)
          if (dof_to_component_map[i] == c)
            n_of_component_dofs += 1;

        Assert(n_of_component_dofs > 0,
               dealii::ExcMessage("No dofs for current component"));

        auto *component_trace_vector_data = new double[n_of_component_dofs];

        unsigned int k = 0;
        for (unsigned int i = 0; i < n_dofs; ++i)
          if (dof_to_component_map[i] == c)
            component_trace_vector_data[k++] =
              this->ddhdg_solver->current_solution_trace[i];

        Assert(
          k == n_of_component_dofs,
          dealii::ExcMessage(
            "Copied a number of elements that is different from n_of_component_dofs"));

        const auto vector_shape  = std::vector<long>{n_of_component_dofs};
        const auto vector_stride = std::vector<long>{8};

        // Create a Python object that will free the allocated
        // memory when destroyed:
        pybind11::capsule free_when_done(component_trace_vector_data,
                                         [](void *f) {
                                           auto *data =
                                             reinterpret_cast<double *>(f);
                                           delete[] data;
                                         });

        auto component_trace_vector =
          pybind11::array_t<double, pybind11::array::c_style>(
            vector_shape,
            vector_stride,
            component_trace_vector_data,
            free_when_done);

        trace_vector_map.insert({c, component_trace_vector});
      }

    return trace_vector_map;
  }


  template <int dim>
  void
  NPSolver<dim>::set_current_trace_vector(
    const std::map<Ddhdg::Component,
                   pybind11::array_t<double, pybind11::array::c_style>>
      &trace_vector_map)
  {
    if (!this->ddhdg_solver->initialized)
      this->ddhdg_solver->setup_overall_system();

    const unsigned int n_dofs = this->ddhdg_solver->dof_handler_trace.n_dofs();

    std::vector<Ddhdg::Component> dof_to_component_map(n_dofs);
    std::vector<Ddhdg::DofType>   dof_to_dof_type_map(n_dofs);

    this->ddhdg_solver->generate_dof_to_component_map(dof_to_component_map,
                                                      dof_to_dof_type_map,
                                                      true,
                                                      false);

    for (const auto &in_vector : trace_vector_map)
      {
        const auto   &c      = in_vector.first;
        const auto    buffer = in_vector.second.request();
        const double *vector = static_cast<const double *>(buffer.ptr);
        unsigned int  k      = 0;
        for (unsigned int i = 0; i < n_dofs; ++i)
          if (dof_to_component_map[i] == c)
            {
              AssertIndexRange(k, buffer.shape[0]);
              this->ddhdg_solver->current_solution_trace[i] = vector[k++];
            }
      }
  }



  template <int dim>
  std::map<std::pair<Ddhdg::Component, Ddhdg::Component>,
           Eigen::SparseMatrix<double>>
  NPSolver<dim>::get_jacobian()
  {
    if (!this->ddhdg_solver->initialized)
      this->ddhdg_solver->setup_overall_system();

    std::map<std::pair<Ddhdg::Component, Ddhdg::Component>,
             Eigen::SparseMatrix<double>>
      jacobian_map;

    const unsigned int n_dofs =
      this->ddhdg_solver->dof_handler_trace_restricted.n_dofs();

    std::vector<Ddhdg::Component> dof_to_component_map(n_dofs);
    std::vector<Ddhdg::DofType>   dof_to_dof_type_map(n_dofs);

    this->ddhdg_solver->generate_dof_to_component_map(dof_to_component_map,
                                                      dof_to_dof_type_map,
                                                      true,
                                                      true);

    std::map<Ddhdg::Component, unsigned int> n_of_component_dofs;

    for (const auto c : this->ddhdg_solver->enabled_components)
      {
        unsigned int n_of_current_component_dofs = 0;
        for (unsigned int i = 0; i < n_dofs; ++i)
          if (dof_to_component_map[i] == c)
            n_of_current_component_dofs += 1;

        Assert(n_of_current_component_dofs > 0,
               dealii::ExcMessage("No dofs for current component"));

        n_of_component_dofs.insert({c, n_of_current_component_dofs});
      }

    for (const auto c1 : this->ddhdg_solver->enabled_components)
      for (const auto c2 : this->ddhdg_solver->enabled_components)
        {
          unsigned int c1_dofs = n_of_component_dofs.at(c1);
          unsigned int c2_dofs = n_of_component_dofs.at(c2);

          // Count number of entries
          unsigned int n_of_entries = 0;
          for (unsigned int i = 0; i < n_dofs; ++i)
            if (dof_to_component_map[i] == c1)
              for (unsigned int j = 0; j < n_dofs; ++j)
                if (dof_to_component_map[j] == c2)
                  ++n_of_entries;

          std::vector<Eigen::Triplet<double>> triplet_list;
          triplet_list.reserve(n_of_entries);

          int k1 = 0;
          int k2;
          for (unsigned int i = 0; i < n_dofs; ++i)
            if (dof_to_component_map[i] == c1)
              {
                k2 = 0;
                for (unsigned int j = 0; j < n_dofs; ++j)
                  if (dof_to_component_map[j] == c2)
                    {
                      if (this->ddhdg_solver->sparsity_pattern.exists(i, j))
                        triplet_list.push_back(
                          {k1, k2, this->ddhdg_solver->system_matrix(i, j)});
                      ++k2;
                    }
                ++k1;
              }
          Assert(k1 == (int)c1_dofs,
                 dealii::ExcMessage(
                   "Copied a number of rows that is different from c1_dofs"));
          Assert(
            k2 == (int)c2_dofs,
            dealii::ExcMessage(
              "Copied a number of columns that is different from c2_dofs"));

          jacobian_map.insert(
            {{c1, c2}, Eigen::SparseMatrix<double>(c1_dofs, c2_dofs)});
          jacobian_map.at({c1, c2}).setFromTriplets(triplet_list.begin(),
                                                    triplet_list.end());
        }

    return jacobian_map;
  }



  template <int dim>
  Ddhdg::NonlinearIterationResults
  NPSolver<dim>::run()
  {
    return this->ddhdg_solver->run();
  }



  template <int dim>
  Ddhdg::NonlinearIterationResults
  NPSolver<dim>::run(const std::optional<double> absolute_tol,
                     const std::optional<double> relative_tol,
                     const std::optional<int>    max_number_of_iterations)
  {
    double abs_tol =
      (absolute_tol.has_value()) ?
        absolute_tol.value() :
        this->get_parameters()->nonlinear_parameters->absolute_tolerance;
    double rel_tol =
      (relative_tol.has_value()) ?
        relative_tol.value() :
        this->get_parameters()->nonlinear_parameters->relative_tolerance;
    int iterations =
      (max_number_of_iterations.has_value()) ?
        max_number_of_iterations.value() :
        this->get_parameters()->nonlinear_parameters->max_number_of_iterations;

    return this->ddhdg_solver->run(abs_tol, rel_tol, iterations);
  }



  template <int dim>
  void
  NPSolver<dim>::compute_local_charge_neutrality_on_trace(
    const bool only_at_boundary)
  {
    return this->ddhdg_solver->compute_local_charge_neutrality_on_trace(
      only_at_boundary);
  }



  template <int dim>
  void
  NPSolver<dim>::compute_local_charge_neutrality()
  {
    return this->ddhdg_solver->compute_local_charge_neutrality();
  }



  template <int dim>
  Ddhdg::NonlinearIterationResults
  NPSolver<dim>::compute_thermodynamic_equilibrium(bool generate_first_guess)
  {
    return this->ddhdg_solver->compute_thermodynamic_equilibrium(
      generate_first_guess);
  }



  template <int dim>
  Ddhdg::NonlinearIterationResults
  NPSolver<dim>::compute_thermodynamic_equilibrium(double absolute_tol,
                                                   double relative_tol,
                                                   int max_number_of_iterations,
                                                   bool generate_first_guess)
  {
    return this->ddhdg_solver->compute_thermodynamic_equilibrium(
      absolute_tol,
      relative_tol,
      max_number_of_iterations,
      generate_first_guess);
  }



  template <int dim>
  void
  NPSolver<dim>::replace_boundary_condition(
    dealii::types::boundary_id   id,
    Ddhdg::BoundaryConditionType bc_type,
    Ddhdg::Component             c,
    DealIIFunction<dim>          f)
  {
    this->ddhdg_solver->replace_boundary_condition(id,
                                                   bc_type,
                                                   c,
                                                   f.get_dealii_function());
  }



  template <int dim>
  void
  NPSolver<dim>::replace_boundary_condition(
    dealii::types::boundary_id   id,
    Ddhdg::BoundaryConditionType bc_type,
    Ddhdg::Component             c,
    const std::string           &f_string)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> f =
      std::make_shared<dealii::FunctionParser<dim>>();
    f->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                  f_string,
                  Ddhdg::Constants::constants);
    this->ddhdg_solver->replace_boundary_condition(id, bc_type, c, f);
  }



  template <int dim>
  void
  NPSolver<dim>::replace_boundary_condition(
    dealii::types::boundary_id   id,
    Ddhdg::BoundaryConditionType bc_type,
    Ddhdg::Component             c,
    const double                 k)
  {
    this->ddhdg_solver->replace_boundary_condition(
      id,
      bc_type,
      c,
      std::make_shared<dealii::Functions::ConstantFunction<dim>>(k));
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_error_per_cell(const Ddhdg::Component c) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(c,
                                                *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_error_per_cell(const Ddhdg::Displacement d) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(d,
                                                *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_l2_error_per_cell(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Component    c) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      expected_solution.get_dealii_function(),
      c,
      dealii::VectorTools::NormType::L2_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_l2_error_per_cell(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Displacement d) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      expected_solution.get_dealii_function(),
      d,
      dealii::VectorTools::NormType::L2_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_l2_error_per_cell(const NPSolver<dim>    other,
                                            const Ddhdg::Component c) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      *(other.ddhdg_solver),
      c,
      dealii::VectorTools::NormType::L2_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_l2_error_per_cell(const NPSolver<dim>       other,
                                            const Ddhdg::Displacement d) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      *(other.ddhdg_solver),
      d,
      dealii::VectorTools::NormType::L2_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_h1_error_per_cell(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Component    c) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      expected_solution.get_dealii_function(),
      c,
      dealii::VectorTools::NormType::H1_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_h1_error_per_cell(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Displacement d) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      expected_solution.get_dealii_function(),
      d,
      dealii::VectorTools::NormType::H1_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_h1_error_per_cell(const NPSolver<dim>    other,
                                            const Ddhdg::Component c) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      *(other.ddhdg_solver),
      c,
      dealii::VectorTools::NormType::H1_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_h1_error_per_cell(const NPSolver<dim>       other,
                                            const Ddhdg::Displacement d) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      *(other.ddhdg_solver),
      d,
      dealii::VectorTools::NormType::H1_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_linfty_error_per_cell(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Component    c) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      expected_solution.get_dealii_function(),
      c,
      dealii::VectorTools::NormType::Linfty_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_linfty_error_per_cell(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Displacement d) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      expected_solution.get_dealii_function(),
      d,
      dealii::VectorTools::NormType::Linfty_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_linfty_error_per_cell(const NPSolver<dim>    other,
                                                const Ddhdg::Component c) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      *(other.ddhdg_solver),
      c,
      dealii::VectorTools::NormType::Linfty_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  ErrorPerCell
  NPSolver<dim>::estimate_linfty_error_per_cell(
    const NPSolver<dim>       other,
    const Ddhdg::Displacement d) const
  {
    ErrorPerCell error_per_cell(this->get_n_active_cells());
    this->ddhdg_solver->estimate_error_per_cell(
      *(other.ddhdg_solver),
      d,
      dealii::VectorTools::NormType::Linfty_norm,
      *(error_per_cell.data_vector));
    return error_per_cell;
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(
    const DealIIFunction<dim>                expected_solution,
    const Ddhdg::Component                   c,
    const dealii::python::QuadratureWrapper &q) const
  {
    const int q_dim = q.get_dim();
    AssertThrow(
      q_dim == dim,
      dealii::ExcMessage(
        "Quadrature rule dimension and solver dimension are different"));
    const dealii::Quadrature<dim> *quadrature_formula =
      static_cast<dealii::Quadrature<dim> *>(q.get_quadrature());

    return this->ddhdg_solver->estimate_l2_error(
      expected_solution.get_dealii_function(), c, *quadrature_formula);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(const DealIIFunction<dim> expected_solution,
                                   const Ddhdg::Component    c) const
  {
    return this->ddhdg_solver->estimate_l2_error(
      expected_solution.get_dealii_function(), c);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(
    const DealIIFunction<dim>                expected_solution,
    const Ddhdg::Displacement                d,
    const dealii::python::QuadratureWrapper &q) const
  {
    const int q_dim = q.get_dim();
    AssertThrow(
      q_dim == dim,
      dealii::ExcMessage(
        "Quadrature rule dimension and solver dimension are different"));
    const dealii::Quadrature<dim> *quadrature_formula =
      static_cast<dealii::Quadrature<dim> *>(q.get_quadrature());

    return this->ddhdg_solver->estimate_l2_error(
      expected_solution.get_dealii_function(), d, *quadrature_formula);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(const DealIIFunction<dim> expected_solution,
                                   const Ddhdg::Displacement d) const
  {
    return this->ddhdg_solver->estimate_l2_error(
      expected_solution.get_dealii_function(), d);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(
    const NPSolver<dim>                      solver,
    const Ddhdg::Component                   c,
    const dealii::python::QuadratureWrapper &q) const
  {
    const int q_dim = q.get_dim();
    AssertThrow(
      q_dim == dim,
      dealii::ExcMessage(
        "Quadrature rule dimension and solver dimension are different"));
    const dealii::Quadrature<dim> *quadrature_formula =
      static_cast<dealii::Quadrature<dim> *>(q.get_quadrature());

    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              c,
                                              dealii::VectorTools::L2_norm,
                                              *quadrature_formula);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(const NPSolver<dim>    solver,
                                   const Ddhdg::Component c) const
  {
    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              c,
                                              dealii::VectorTools::L2_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(
    const NPSolver<dim>                      solver,
    const Ddhdg::Displacement                d,
    const dealii::python::QuadratureWrapper &q) const
  {
    const int q_dim = q.get_dim();
    AssertThrow(
      q_dim == dim,
      dealii::ExcMessage(
        "Quadrature rule dimension and solver dimension are different"));
    const dealii::Quadrature<dim> *quadrature_formula =
      static_cast<dealii::Quadrature<dim> *>(q.get_quadrature());

    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              d,
                                              dealii::VectorTools::L2_norm,
                                              *quadrature_formula);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error(const NPSolver<dim>       solver,
                                   const Ddhdg::Displacement d) const
  {
    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              d,
                                              dealii::VectorTools::L2_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(
    const DealIIFunction<dim>                expected_solution,
    const Ddhdg::Component                   c,
    const dealii::python::QuadratureWrapper &q) const
  {
    const int q_dim = q.get_dim();
    AssertThrow(
      q_dim == dim,
      dealii::ExcMessage(
        "Quadrature rule dimension and solver dimension are different"));
    const dealii::Quadrature<dim> *quadrature_formula =
      static_cast<dealii::Quadrature<dim> *>(q.get_quadrature());

    return this->ddhdg_solver->estimate_h1_error(
      expected_solution.get_dealii_function(), c, *quadrature_formula);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(const DealIIFunction<dim> expected_solution,
                                   const Ddhdg::Component    c) const
  {
    return this->ddhdg_solver->estimate_h1_error(
      expected_solution.get_dealii_function(), c);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(
    const DealIIFunction<dim>                expected_solution,
    const Ddhdg::Displacement                d,
    const dealii::python::QuadratureWrapper &q) const
  {
    const int q_dim = q.get_dim();
    AssertThrow(
      q_dim == dim,
      dealii::ExcMessage(
        "Quadrature rule dimension and solver dimension are different"));
    const dealii::Quadrature<dim> *quadrature_formula =
      static_cast<dealii::Quadrature<dim> *>(q.get_quadrature());

    return this->ddhdg_solver->estimate_h1_error(
      expected_solution.get_dealii_function(), d, *quadrature_formula);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(const DealIIFunction<dim> expected_solution,
                                   const Ddhdg::Displacement d) const
  {
    return this->ddhdg_solver->estimate_h1_error(
      expected_solution.get_dealii_function(), d);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(
    const NPSolver<dim>                      solver,
    const Ddhdg::Component                   c,
    const dealii::python::QuadratureWrapper &q) const
  {
    const int q_dim = q.get_dim();
    AssertThrow(
      q_dim == dim,
      dealii::ExcMessage(
        "Quadrature rule dimension and solver dimension are different"));
    const dealii::Quadrature<dim> *quadrature_formula =
      static_cast<dealii::Quadrature<dim> *>(q.get_quadrature());

    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              c,
                                              dealii::VectorTools::H1_norm,
                                              *quadrature_formula);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(const NPSolver<dim>    solver,
                                   const Ddhdg::Component c) const
  {
    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              c,
                                              dealii::VectorTools::H1_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(
    const NPSolver<dim>                      solver,
    const Ddhdg::Displacement                d,
    const dealii::python::QuadratureWrapper &q) const
  {
    const int q_dim = q.get_dim();
    AssertThrow(
      q_dim == dim,
      dealii::ExcMessage(
        "Quadrature rule dimension and solver dimension are different"));
    const dealii::Quadrature<dim> *quadrature_formula =
      static_cast<dealii::Quadrature<dim> *>(q.get_quadrature());

    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              d,
                                              dealii::VectorTools::H1_norm,
                                              *quadrature_formula);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_h1_error(const NPSolver<dim>       solver,
                                   const Ddhdg::Displacement d) const
  {
    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              d,
                                              dealii::VectorTools::H1_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(
    const DealIIFunction<dim>                expected_solution,
    const Ddhdg::Component                   c,
    const dealii::python::QuadratureWrapper &q) const
  {
    const int q_dim = q.get_dim();
    AssertThrow(
      q_dim == dim,
      dealii::ExcMessage(
        "Quadrature rule dimension and solver dimension are different"));
    const dealii::Quadrature<dim> *quadrature_formula =
      static_cast<dealii::Quadrature<dim> *>(q.get_quadrature());

    return this->ddhdg_solver->estimate_linfty_error(
      expected_solution.get_dealii_function(), c, *quadrature_formula);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Component    c) const
  {
    return this->ddhdg_solver->estimate_linfty_error(
      expected_solution.get_dealii_function(), c);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(
    const DealIIFunction<dim>                expected_solution,
    const Ddhdg::Displacement                d,
    const dealii::python::QuadratureWrapper &q) const
  {
    const int q_dim = q.get_dim();
    AssertThrow(
      q_dim == dim,
      dealii::ExcMessage(
        "Quadrature rule dimension and solver dimension are different"));
    const dealii::Quadrature<dim> *quadrature_formula =
      static_cast<dealii::Quadrature<dim> *>(q.get_quadrature());

    return this->ddhdg_solver->estimate_linfty_error(
      expected_solution.get_dealii_function(), d, *quadrature_formula);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Displacement d) const
  {
    return this->ddhdg_solver->estimate_linfty_error(
      expected_solution.get_dealii_function(), d);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(
    const NPSolver<dim>                      solver,
    const Ddhdg::Component                   c,
    const dealii::python::QuadratureWrapper &q) const
  {
    const int q_dim = q.get_dim();
    AssertThrow(
      q_dim == dim,
      dealii::ExcMessage(
        "Quadrature rule dimension and solver dimension are different"));
    const dealii::Quadrature<dim> *quadrature_formula =
      static_cast<dealii::Quadrature<dim> *>(q.get_quadrature());

    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              c,
                                              dealii::VectorTools::Linfty_norm,
                                              *quadrature_formula);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(const NPSolver<dim>    solver,
                                       const Ddhdg::Component c) const
  {
    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              c,
                                              dealii::VectorTools::Linfty_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(
    const NPSolver<dim>                      solver,
    const Ddhdg::Displacement                d,
    const dealii::python::QuadratureWrapper &q) const
  {
    const int q_dim = q.get_dim();
    AssertThrow(
      q_dim == dim,
      dealii::ExcMessage(
        "Quadrature rule dimension and solver dimension are different"));
    const dealii::Quadrature<dim> *quadrature_formula =
      static_cast<dealii::Quadrature<dim> *>(q.get_quadrature());

    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              d,
                                              dealii::VectorTools::Linfty_norm,
                                              *quadrature_formula);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error(const NPSolver<dim>       solver,
                                       const Ddhdg::Displacement d) const
  {
    return this->ddhdg_solver->estimate_error(*(solver.ddhdg_solver),
                                              d,
                                              dealii::VectorTools::Linfty_norm);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error_on_trace(
    const std::string                       &expected_solution,
    const Ddhdg::Component                   c,
    const dealii::python::QuadratureWrapper &q) const
  {
    const int q_dim = q.get_dim();
    AssertThrow(
      q_dim == dim - 1,
      dealii::ExcMessage(
        "Quadrature rule dimension must be 1 less than the solver dimension"));
    const dealii::Quadrature<dim - 1> *face_quadrature_formula =
      static_cast<dealii::Quadrature<dim - 1> *>(q.get_quadrature());

    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_solution,
      Ddhdg::Constants::constants);
    return this->ddhdg_solver->estimate_l2_error_on_trace(
      expected_solution_f, c, *face_quadrature_formula);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error_on_trace(
    const std::string     &expected_solution,
    const Ddhdg::Component c) const
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_solution,
      Ddhdg::Constants::constants);
    return this->ddhdg_solver->estimate_l2_error_on_trace(expected_solution_f,
                                                          c);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error_on_trace(
    const DealIIFunction<dim>                expected_solution,
    const Ddhdg::Component                   c,
    const dealii::python::QuadratureWrapper &q) const
  {
    const int q_dim = q.get_dim();
    AssertThrow(
      q_dim == dim - 1,
      dealii::ExcMessage(
        "Quadrature rule dimension must be 1 less than the solver dimension"));
    const dealii::Quadrature<dim - 1> *face_quadrature_formula =
      static_cast<dealii::Quadrature<dim - 1> *>(q.get_quadrature());

    return this->ddhdg_solver->estimate_l2_error_on_trace(
      expected_solution.get_dealii_function(), c, *face_quadrature_formula);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_l2_error_on_trace(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Component    c) const
  {
    return this->ddhdg_solver->estimate_l2_error_on_trace(
      expected_solution.get_dealii_function(), c);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error_on_trace(
    const std::string                       &expected_solution,
    const Ddhdg::Component                   c,
    const dealii::python::QuadratureWrapper &q) const
  {
    const int q_dim = q.get_dim();
    AssertThrow(
      q_dim == dim - 1,
      dealii::ExcMessage(
        "Quadrature rule dimension must be 1 less than the solver dimension"));
    const dealii::Quadrature<dim - 1> *face_quadrature_formula =
      static_cast<dealii::Quadrature<dim - 1> *>(q.get_quadrature());

    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_solution,
      Ddhdg::Constants::constants);
    return this->ddhdg_solver->estimate_linfty_error_on_trace(
      expected_solution_f, c, *face_quadrature_formula);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error_on_trace(
    const std::string     &expected_solution,
    const Ddhdg::Component c) const
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_solution,
      Ddhdg::Constants::constants);
    return this->ddhdg_solver->estimate_linfty_error_on_trace(
      expected_solution_f, c);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error_on_trace(
    const DealIIFunction<dim>                expected_solution,
    const Ddhdg::Component                   c,
    const dealii::python::QuadratureWrapper &q) const
  {
    const int q_dim = q.get_dim();
    AssertThrow(
      q_dim == dim - 1,
      dealii::ExcMessage(
        "Quadrature rule dimension must be 1 less than the solver dimension"));
    const dealii::Quadrature<dim - 1> *face_quadrature_formula =
      static_cast<dealii::Quadrature<dim - 1> *>(q.get_quadrature());

    return this->ddhdg_solver->estimate_linfty_error_on_trace(
      expected_solution.get_dealii_function(), c, *face_quadrature_formula);
  }



  template <int dim>
  double
  NPSolver<dim>::estimate_linfty_error_on_trace(
    const DealIIFunction<dim> expected_solution,
    const Ddhdg::Component    c) const
  {
    return this->ddhdg_solver->estimate_linfty_error_on_trace(
      expected_solution.get_dealii_function(), c);
  }



  template <int dim>
  DealIIFunction<dim>
  NPSolver<dim>::get_solution(const Ddhdg::Component c) const
  {
    return DealIIFunction(this->ddhdg_solver->get_solution(c));
  }



  template <int dim>
  double
  NPSolver<dim>::get_solution_on_a_point(const dealii::Point<dim> p,
                                         const Ddhdg::Component   c) const
  {
    return this->ddhdg_solver->get_solution_on_a_point(p, c);
  }



  template <int dim>
  void
  NPSolver<dim>::output_results(const std::string &solution_filename,
                                const bool         save_update,
                                const bool redimensionalize_quantities) const
  {
    this->ddhdg_solver->output_results(solution_filename,
                                       save_update,
                                       redimensionalize_quantities);
  }



  template <int dim>
  void
  NPSolver<dim>::output_results(const std::string &solution_filename,
                                const std::string &trace_filename,
                                const bool         save_update,
                                const bool redimensionalize_quantities) const
  {
    this->ddhdg_solver->output_results(solution_filename,
                                       trace_filename,
                                       save_update,
                                       redimensionalize_quantities);
  }



  template <int dim>
  std::string
  NPSolver<dim>::print_convergence_table(
    DealIIFunction<dim> expected_V_solution,
    DealIIFunction<dim> expected_n_solution,
    DealIIFunction<dim> expected_p_solution,
    const unsigned int  n_cycles,
    const unsigned int  initial_refinements)
  {
    std::ostringstream stream;
    this->ddhdg_solver->print_convergence_table(
      std::make_shared<Ddhdg::ConvergenceTable>(dim),
      expected_V_solution.get_dealii_function(),
      expected_n_solution.get_dealii_function(),
      expected_p_solution.get_dealii_function(),
      n_cycles,
      initial_refinements,
      stream);
    return stream.str();
  }



  template <int dim>
  std::string
  NPSolver<dim>::print_convergence_table(
    DealIIFunction<dim> expected_V_solution,
    DealIIFunction<dim> expected_n_solution,
    DealIIFunction<dim> expected_p_solution,
    DealIIFunction<dim> initial_V_function,
    DealIIFunction<dim> initial_n_function,
    DealIIFunction<dim> initial_p_function,
    const unsigned int  n_cycles,
    const unsigned int  initial_refinements)
  {
    std::ostringstream stream;
    this->ddhdg_solver->print_convergence_table(
      std::make_shared<Ddhdg::ConvergenceTable>(dim),
      expected_V_solution.get_dealii_function(),
      expected_n_solution.get_dealii_function(),
      expected_p_solution.get_dealii_function(),
      initial_V_function.get_dealii_function(),
      initial_n_function.get_dealii_function(),
      initial_p_function.get_dealii_function(),
      n_cycles,
      initial_refinements,
      stream);
    return stream.str();
  }



  template <int dim>
  std::string
  NPSolver<dim>::print_convergence_table(const std::string &expected_V_solution,
                                         const std::string &expected_n_solution,
                                         const std::string &expected_p_solution,
                                         const unsigned int n_cycles,
                                         const unsigned int initial_refinements)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_V_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> expected_n_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> expected_p_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_V_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_V_solution,
      Ddhdg::Constants::constants);
    expected_n_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_n_solution,
      Ddhdg::Constants::constants);
    expected_p_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_p_solution,
      Ddhdg::Constants::constants);

    std::ostringstream stream;
    this->ddhdg_solver->print_convergence_table(
      std::make_shared<Ddhdg::ConvergenceTable>(dim),
      expected_V_solution_f,
      expected_n_solution_f,
      expected_p_solution_f,
      n_cycles,
      initial_refinements,
      stream);
    return stream.str();
  }



  template <int dim>
  std::string
  NPSolver<dim>::print_convergence_table(const std::string &expected_V_solution,
                                         const std::string &expected_n_solution,
                                         const std::string &expected_p_solution,
                                         const std::string &initial_V_function,
                                         const std::string &initial_n_function,
                                         const std::string &initial_p_function,
                                         const unsigned int n_cycles,
                                         const unsigned int initial_refinements)
  {
    std::shared_ptr<dealii::FunctionParser<dim>> expected_V_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> expected_n_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> expected_p_solution_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> initial_V_function_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> initial_n_function_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    std::shared_ptr<dealii::FunctionParser<dim>> initial_p_function_f =
      std::make_shared<dealii::FunctionParser<dim>>();
    expected_V_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_V_solution,
      Ddhdg::Constants::constants);
    expected_n_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_n_solution,
      Ddhdg::Constants::constants);
    expected_p_solution_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      expected_p_solution,
      Ddhdg::Constants::constants);
    initial_V_function_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      initial_V_function,
      Ddhdg::Constants::constants);
    initial_n_function_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      initial_n_function,
      Ddhdg::Constants::constants);
    initial_p_function_f->initialize(
      dealii::FunctionParser<dim>::default_variable_names(),
      initial_p_function,
      Ddhdg::Constants::constants);

    std::ostringstream stream;
    this->ddhdg_solver->print_convergence_table(
      std::make_shared<Ddhdg::ConvergenceTable>(dim),
      expected_V_solution_f,
      expected_n_solution_f,
      expected_p_solution_f,
      initial_V_function_f,
      initial_n_function_f,
      initial_p_function_f,
      n_cycles,
      initial_refinements,
      stream);
    return stream.str();
  }



  template <int dim>
  double
  NPSolver<dim>::compute_quasi_fermi_potential(
    const double           density,
    const double           potential,
    const double           temperature,
    const Ddhdg::Component component) const
  {
    switch (component)
      {
        case Ddhdg::Component::n:
          return this->ddhdg_solver
            ->template compute_quasi_fermi_potential<Ddhdg::Component::n>(
              density, potential, temperature);
        case Ddhdg::Component::p:
          return this->ddhdg_solver
            ->template compute_quasi_fermi_potential<Ddhdg::Component::p>(
              density, potential, temperature);
        default:
          Assert(false, Ddhdg::InvalidComponent());
          return 9e99;
      }
  }



  template <int dim>
  double
  NPSolver<dim>::compute_density(const double           qf_potential,
                                 const double           potential,
                                 const double           temperature,
                                 const Ddhdg::Component component) const
  {
    switch (component)
      {
        case Ddhdg::Component::n:
          return this->ddhdg_solver
            ->template compute_density<Ddhdg::Component::n>(qf_potential,
                                                            potential,
                                                            temperature);
        case Ddhdg::Component::p:
          return this->ddhdg_solver
            ->template compute_density<Ddhdg::Component::p>(qf_potential,
                                                            potential,
                                                            temperature);
        default:
          Assert(false, Ddhdg::InvalidComponent());
          return 9e99;
      }
  }



  template class HomogeneousPermittivity<1>;
  template class HomogeneousPermittivity<2>;
  template class HomogeneousPermittivity<3>;

  template class HomogeneousMobility<1>;
  template class HomogeneousMobility<2>;
  template class HomogeneousMobility<3>;

  template class DealIIFunction<1>;
  template class DealIIFunction<2>;
  template class DealIIFunction<3>;

  template class AnalyticFunction<1>;
  template class AnalyticFunction<2>;
  template class AnalyticFunction<3>;

  template class PiecewiseFunction<1>;
  template class PiecewiseFunction<2>;
  template class PiecewiseFunction<3>;

  template class LinearRecombinationTerm<1>;
  template class LinearRecombinationTerm<2>;
  template class LinearRecombinationTerm<3>;

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

  template class PythonDefinedRecombinationTerm<1>;
  template class PythonDefinedRecombinationTerm<2>;
  template class PythonDefinedRecombinationTerm<3>;

  template class PythonDefinedSpacialRecombinationTerm<1>;
  template class PythonDefinedSpacialRecombinationTerm<2>;
  template class PythonDefinedSpacialRecombinationTerm<3>;

  template class SuperimposedRecombinationTerm<1>;
  template class SuperimposedRecombinationTerm<2>;
  template class SuperimposedRecombinationTerm<3>;

  template class BoundaryConditionHandler<1>;
  template class BoundaryConditionHandler<2>;
  template class BoundaryConditionHandler<3>;

  template class Problem<1>;
  template class Problem<2>;
  template class Problem<3>;

  template class NPSolver<1>;
  template class NPSolver<2>;
  template class NPSolver<3>;
} // namespace pyddhdg
