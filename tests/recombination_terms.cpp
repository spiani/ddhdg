#include <gtest/gtest.h>
#include <recombination_term.h>

using dimensions = ::testing::Types<std::integral_constant<unsigned int, 1>,
                                    std::integral_constant<unsigned int, 2>,
                                    std::integral_constant<unsigned int, 3>>;

template <typename T>
class ShockleyReadHallFixedTemperatureTest : public ::testing::Test
{
public:
  static constexpr unsigned int dim = T::value;
};

TYPED_TEST_SUITE(ShockleyReadHallFixedTemperatureTest, dimensions, );

TYPED_TEST(ShockleyReadHallFixedTemperatureTest,
           StandardConstructor) // NOLINT
{
  constexpr unsigned int dim = TypeParam::value;

  constexpr double n = .123;
  constexpr double p = .42;

  constexpr double ni    = 1;
  constexpr double tau_n = 2;
  constexpr double tau_p = 3.;

  constexpr double EXPECTED_VALUE =
    (n * p - ni) / (tau_p * (n + ni) + tau_n * (p + ni));

  const auto r = Ddhdg::ShockleyReadHallFixedTemperature<dim>(ni, tau_n, tau_p);

  const double value =
    r.compute_recombination_term(n, p, dealii::Point<dim>(), 1);

  EXPECT_DOUBLE_EQ(value, EXPECTED_VALUE);
}



TYPED_TEST(ShockleyReadHallFixedTemperatureTest,
           PhysicalParametersConstructor) // NOLINT
{
  constexpr unsigned int dim = TypeParam::value;

  constexpr double n = .123;
  constexpr double p = .42;

  constexpr double Nc    = 1;
  constexpr double Nv    = 1;
  constexpr double Ec    = 1;
  constexpr double Ev    = 1;
  constexpr double T     = 1;
  constexpr double tau_n = 2;
  constexpr double tau_p = 3.;

  const double ni = Nc * Nv * exp((Ev - Ec) / (Ddhdg::Constants::KB * T));

  const auto r1 = Ddhdg::ShockleyReadHallFixedTemperature<dim>(
    Nc, Nv, Ec, Ev, T, tau_n, tau_p);
  const auto r2 =
    Ddhdg::ShockleyReadHallFixedTemperature<dim>(ni, tau_n, tau_p);

  const double value1 =
    r1.compute_recombination_term(n, p, dealii::Point<dim>(), 1);
  const double value2 =
    r2.compute_recombination_term(n, p, dealii::Point<dim>(), 1);

  EXPECT_DOUBLE_EQ(value1, value2);
}


template <typename T>
class ShockleyReadHall : public ::testing::Test
{
public:
  static constexpr unsigned int dim = T::value;
};

TYPED_TEST_SUITE(ShockleyReadHall, dimensions, );



TYPED_TEST(ShockleyReadHall, OutputTest) // NOLINT
{
  // Compare that, for a fixed temperature, this class returns the same values
  // of ShockleyReadHallFixedTemperature
  constexpr unsigned int dim = TypeParam::value;

  constexpr double n = .123;
  constexpr double p = .42;

  constexpr double Nc = 1.5;
  constexpr double Nv = 1.3;
  constexpr double Ec = .5e-10;
  constexpr double Ev = .7e-11;

  constexpr double tau_n = 2;
  constexpr double tau_p = 3.;

  double T;

  double v1;
  double v2;

  double v1_der_n;
  double v2_der_n;

  double v1_der_p;
  double v2_der_p;

  for (unsigned int i = 0; i < 3; i++)
    {
      T = 100 * i;
      const std::shared_ptr<dealii::Function<dim>> temperature =
        std::make_shared<dealii::Functions::ConstantFunction<dim>>(T);

      const auto r1 =
        Ddhdg::ShockleyReadHall<dim>(Nc, Nv, Ec, Ev, temperature, tau_n, tau_p);
      const auto r2 = Ddhdg::ShockleyReadHallFixedTemperature<dim>(
        Nc, Nv, Ec, Ev, T, tau_n, tau_p);

      v1       = r1.compute_recombination_term(n, p, dealii::Point<dim>(), 1);
      v2       = r2.compute_recombination_term(n, p, dealii::Point<dim>(), 1);
      v1_der_n = r1.compute_derivative_of_recombination_term(
        n, p, dealii::Point<dim>(), 1, Ddhdg::Component::n);
      v2_der_n = r2.compute_derivative_of_recombination_term(
        n, p, dealii::Point<dim>(), 1, Ddhdg::Component::n);
      v1_der_p = r1.compute_derivative_of_recombination_term(
        n, p, dealii::Point<dim>(), 1, Ddhdg::Component::p);
      v2_der_p = r2.compute_derivative_of_recombination_term(
        n, p, dealii::Point<dim>(), 1, Ddhdg::Component::p);

      EXPECT_DOUBLE_EQ(v1, v2);
      EXPECT_DOUBLE_EQ(v1_der_n, v2_der_n);
      EXPECT_DOUBLE_EQ(v1_der_p, v2_der_p);
    }
}



TYPED_TEST(ShockleyReadHall, Buffer) // NOLINT
{
  // Compute multiple values of the recombination term with different size;
  // ensure that the buffer is resized accordingly
  constexpr unsigned int dim = TypeParam::value;

  constexpr double                             Nc = 1.5;
  constexpr double                             Nv = 1.3;
  constexpr double                             Ec = .5e-10;
  constexpr double                             Ev = .7e-11;
  const std::shared_ptr<dealii::Function<dim>> temperature =
    std::make_shared<dealii::Functions::ConstantFunction<dim>>(300.);

  constexpr double tau_n = 2;
  constexpr double tau_p = 3.;

  std::vector<double> n_1(100);
  std::vector<double> n_2(30);
  std::vector<double> n_3(70);

  std::vector<double> p_1(100);
  std::vector<double> p_2(30);
  std::vector<double> p_3(70);

  std::vector<dealii::Point<dim>> points_1(100);
  std::vector<dealii::Point<dim>> points_2(30);
  std::vector<dealii::Point<dim>> points_3(70);

  std::vector<double> value_1(100);
  std::vector<double> value_2(30);
  std::vector<double> value_3(70);

  std::vector<double> der_n_1(100);
  std::vector<double> der_n_2(30);
  std::vector<double> der_n_3(70);

  std::vector<double> der_p_1(100);
  std::vector<double> der_p_2(30);
  std::vector<double> der_p_3(70);

  for (unsigned int i = 0; i < 100; ++i)
    {
      const double n = 100. * i / 37;
      const double p = 4 - 100. * i / 37;
      n_1[i]         = n;
      p_1[i]         = p;

      if (i < 30)
        {
          n_2[i] = n;
          p_2[i] = p;
        }
      else
        {
          n_3[i - 30] = n;
          p_3[i - 30] = p;
        }
    }

  auto recombination_term =
    Ddhdg::ShockleyReadHall<dim>(Nc, Nv, Ec, Ev, temperature, tau_n, tau_p);

  recombination_term.compute_multiple_recombination_terms(
    n_1, p_1, points_1, 1, true, value_1);
  recombination_term.compute_multiple_recombination_terms(
    n_2, p_2, points_2, 1, true, value_2);
  recombination_term.compute_multiple_recombination_terms(
    n_3, p_3, points_3, 1, true, value_3);

  recombination_term.compute_multiple_derivatives_of_recombination_terms(
    n_1, p_1, points_1, 1, Ddhdg::Component::n, true, der_n_1);
  recombination_term.compute_multiple_derivatives_of_recombination_terms(
    n_2, p_2, points_2, 1, Ddhdg::Component::n, true, der_n_2);
  recombination_term.compute_multiple_derivatives_of_recombination_terms(
    n_3, p_3, points_3, 1, Ddhdg::Component::n, true, der_n_3);

  recombination_term.compute_multiple_derivatives_of_recombination_terms(
    n_1, p_1, points_1, 1, Ddhdg::Component::p, true, der_p_1);
  recombination_term.compute_multiple_derivatives_of_recombination_terms(
    n_2, p_2, points_2, 1, Ddhdg::Component::p, true, der_p_2);
  recombination_term.compute_multiple_derivatives_of_recombination_terms(
    n_3, p_3, points_3, 1, Ddhdg::Component::p, true, der_p_3);

  for (unsigned int i = 0; i < 100; ++i)
    {
      if (i < 30)
        {
          EXPECT_DOUBLE_EQ(value_1[i], value_2[i]);
          EXPECT_DOUBLE_EQ(der_n_1[i], der_n_2[i]);
          EXPECT_DOUBLE_EQ(der_p_1[i], der_p_2[i]);
        }
      else
        {
          EXPECT_DOUBLE_EQ(value_1[i], value_3[i - 30]);
          EXPECT_DOUBLE_EQ(der_n_1[i], der_n_3[i - 30]);
          EXPECT_DOUBLE_EQ(der_p_1[i], der_p_3[i - 30]);
        }
    }
}



template <typename T>
class Auger : public ::testing::Test
{
public:
  static constexpr unsigned int dim = T::value;
};

TYPED_TEST_SUITE(Auger, dimensions, );



TYPED_TEST(Auger, OutputTest) // NOLINT
{
  // Compare that, for a fixed temperature, this class returns the same values
  // of AugerFixedTemperature
  constexpr unsigned int dim = TypeParam::value;

  constexpr double n = .123;
  constexpr double p = .42;

  constexpr double Nc = 1.5;
  constexpr double Nv = 1.3;
  constexpr double Ec = .5e-10;
  constexpr double Ev = .7e-11;

  constexpr double n_coefficient = 2;
  constexpr double p_coefficient = 3.;

  double T;

  double v1;
  double v2;

  double v1_der_n;
  double v2_der_n;

  double v1_der_p;
  double v2_der_p;

  for (unsigned int i = 0; i < 3; i++)
    {
      T = 50 * i;
      const std::shared_ptr<dealii::Function<dim>> temperature =
        std::make_shared<dealii::Functions::ConstantFunction<dim>>(T);

      const auto r1 = Ddhdg::Auger<dim>(
        Nc, Nv, Ec, Ev, temperature, n_coefficient, p_coefficient);
      const auto r2 = Ddhdg::AugerFixedTemperature<dim>(
        Nc, Nv, Ec, Ev, T, n_coefficient, p_coefficient);

      v1       = r1.compute_recombination_term(n, p, dealii::Point<dim>(), 1);
      v2       = r2.compute_recombination_term(n, p, dealii::Point<dim>(), 1);
      v1_der_n = r1.compute_derivative_of_recombination_term(
        n, p, dealii::Point<dim>(), 1, Ddhdg::Component::n);
      v2_der_n = r2.compute_derivative_of_recombination_term(
        n, p, dealii::Point<dim>(), 1, Ddhdg::Component::n);
      v1_der_p = r1.compute_derivative_of_recombination_term(
        n, p, dealii::Point<dim>(), 1, Ddhdg::Component::p);
      v2_der_p = r2.compute_derivative_of_recombination_term(
        n, p, dealii::Point<dim>(), 1, Ddhdg::Component::p);

      EXPECT_DOUBLE_EQ(v1, v2);
      EXPECT_DOUBLE_EQ(v1_der_n, v2_der_n);
      EXPECT_DOUBLE_EQ(v1_der_p, v2_der_p);
    }
}



TYPED_TEST(Auger, Buffer) // NOLINT
{
  // Compute multiple values of the recombination term with different size;
  // ensure that the buffer is resized accordingly
  constexpr unsigned int dim = TypeParam::value;

  constexpr double                             Nc = 1.5;
  constexpr double                             Nv = 1.3;
  constexpr double                             Ec = .5e-10;
  constexpr double                             Ev = .7e-11;
  const std::shared_ptr<dealii::Function<dim>> temperature =
    std::make_shared<dealii::Functions::ConstantFunction<dim>>(300.);

  constexpr double n_coefficient = 2;
  constexpr double p_coefficient = 3.;

  std::vector<double> n_1(100);
  std::vector<double> n_2(30);
  std::vector<double> n_3(70);

  std::vector<double> p_1(100);
  std::vector<double> p_2(30);
  std::vector<double> p_3(70);

  std::vector<dealii::Point<dim>> points_1(100);
  std::vector<dealii::Point<dim>> points_2(30);
  std::vector<dealii::Point<dim>> points_3(70);

  std::vector<double> value_1(100);
  std::vector<double> value_2(30);
  std::vector<double> value_3(70);

  std::vector<double> der_n_1(100);
  std::vector<double> der_n_2(30);
  std::vector<double> der_n_3(70);

  std::vector<double> der_p_1(100);
  std::vector<double> der_p_2(30);
  std::vector<double> der_p_3(70);

  for (unsigned int i = 0; i < 100; ++i)
    {
      const double n = 100. * i / 37;
      const double p = 4 - 100. * i / 37;
      n_1[i]         = n;
      p_1[i]         = p;

      if (i < 30)
        {
          n_2[i] = n;
          p_2[i] = p;
        }
      else
        {
          n_3[i - 30] = n;
          p_3[i - 30] = p;
        }
    }

  auto recombination_term = Ddhdg::Auger<dim>(
    Nc, Nv, Ec, Ev, temperature, n_coefficient, p_coefficient);

  recombination_term.compute_multiple_recombination_terms(
    n_1, p_1, points_1, 1, true, value_1);
  recombination_term.compute_multiple_recombination_terms(
    n_2, p_2, points_2, 1, true, value_2);
  recombination_term.compute_multiple_recombination_terms(
    n_3, p_3, points_3, 1, true, value_3);

  recombination_term.compute_multiple_derivatives_of_recombination_terms(
    n_1, p_1, points_1, 1, Ddhdg::Component::n, true, der_n_1);
  recombination_term.compute_multiple_derivatives_of_recombination_terms(
    n_2, p_2, points_2, 1, Ddhdg::Component::n, true, der_n_2);
  recombination_term.compute_multiple_derivatives_of_recombination_terms(
    n_3, p_3, points_3, 1, Ddhdg::Component::n, true, der_n_3);

  recombination_term.compute_multiple_derivatives_of_recombination_terms(
    n_1, p_1, points_1, 1, Ddhdg::Component::p, true, der_p_1);
  recombination_term.compute_multiple_derivatives_of_recombination_terms(
    n_2, p_2, points_2, 1, Ddhdg::Component::p, true, der_p_2);
  recombination_term.compute_multiple_derivatives_of_recombination_terms(
    n_3, p_3, points_3, 1, Ddhdg::Component::p, true, der_p_3);

  for (unsigned int i = 0; i < 100; ++i)
    {
      if (i < 30)
        {
          EXPECT_DOUBLE_EQ(value_1[i], value_2[i]);
          EXPECT_DOUBLE_EQ(der_n_1[i], der_n_2[i]);
          EXPECT_DOUBLE_EQ(der_p_1[i], der_p_2[i]);
        }
      else
        {
          EXPECT_DOUBLE_EQ(value_1[i], value_3[i - 30]);
          EXPECT_DOUBLE_EQ(der_n_1[i], der_n_3[i - 30]);
          EXPECT_DOUBLE_EQ(der_p_1[i], der_p_3[i - 30]);
        }
    }
}



template <typename T>
class SuperimposedRecombinationTest : public ::testing::Test
{
public:
  static constexpr unsigned int dim = T::value;
};

TYPED_TEST_SUITE(SuperimposedRecombinationTest, dimensions, );



TYPED_TEST(SuperimposedRecombinationTest, sum_of_two) // NOLINT
{
  constexpr unsigned int dim = TypeParam::value;

  constexpr double n = .123;
  constexpr double p = .42;

  constexpr double ni    = 1;
  constexpr double tau_n = 2;
  constexpr double tau_p = 3.;

  constexpr double n_coefficient = 1.5;
  constexpr double p_coefficient = 2.5;


  std::shared_ptr<Ddhdg::RecombinationTerm<dim>> r1 =
    std::make_shared<Ddhdg::ShockleyReadHallFixedTemperature<dim>>(ni,
                                                                   tau_n,
                                                                   tau_p);
  std::shared_ptr<Ddhdg::RecombinationTerm<dim>> r2 =
    std::make_shared<Ddhdg::AugerFixedTemperature<dim>>(ni,
                                                        n_coefficient,
                                                        p_coefficient);

  auto superimposed_r =
    std::make_shared<Ddhdg::SuperimposedRecombinationTerm<dim>>(r1, r2);

  const double value1 =
    r1->compute_recombination_term(n, p, dealii::Point<dim>(), 1);
  const double value2 =
    r2->compute_recombination_term(n, p, dealii::Point<dim>(), 1);

  const auto tot_value =
    superimposed_r->compute_recombination_term(n, p, dealii::Point<dim>(), 1);

  EXPECT_DOUBLE_EQ(value1 + value2, tot_value);

  const double der_n_1 = r1->compute_derivative_of_recombination_term(
    n, p, dealii::Point<dim>(), 1, Ddhdg::Component::n);
  const double der_n_2 = r2->compute_derivative_of_recombination_term(
    n, p, dealii::Point<dim>(), 1, Ddhdg::Component::n);

  const auto tot_der_n =
    superimposed_r->compute_derivative_of_recombination_term(
      n, p, dealii::Point<dim>(), 1, Ddhdg::Component::n);

  EXPECT_DOUBLE_EQ(der_n_1 + der_n_2, tot_der_n);

  const double der_p_1 = r1->compute_derivative_of_recombination_term(
    n, p, dealii::Point<dim>(), 1, Ddhdg::Component::p);
  const double der_p_2 = r2->compute_derivative_of_recombination_term(
    n, p, dealii::Point<dim>(), 1, Ddhdg::Component::p);

  const auto tot_der_p =
    superimposed_r->compute_derivative_of_recombination_term(
      n, p, dealii::Point<dim>(), 1, Ddhdg::Component::p);

  EXPECT_DOUBLE_EQ(der_p_1 + der_p_2, tot_der_p);
}



TYPED_TEST(SuperimposedRecombinationTest, sum_of_three) // NOLINT
{
  constexpr unsigned int dim = TypeParam::value;

  constexpr double n = 313;
  constexpr double p = 1241;

  constexpr double ni    = 1;
  constexpr double tau_n = 10;
  constexpr double tau_p = 11;

  constexpr double n_coefficient_1 = 12;
  constexpr double p_coefficient_1 = 9;

  constexpr double n_coefficient_2 = 15;
  constexpr double p_coefficient_2 = 5;


  std::shared_ptr<Ddhdg::RecombinationTerm<dim>> r1 =
    std::make_shared<Ddhdg::ShockleyReadHallFixedTemperature<dim>>(ni,
                                                                   tau_n,
                                                                   tau_p);
  std::shared_ptr<Ddhdg::RecombinationTerm<dim>> r2 =
    std::make_shared<Ddhdg::AugerFixedTemperature<dim>>(ni,
                                                        n_coefficient_1,
                                                        p_coefficient_1);
  std::shared_ptr<Ddhdg::RecombinationTerm<dim>> r3 =
    std::make_shared<Ddhdg::AugerFixedTemperature<dim>>(ni,
                                                        n_coefficient_2,
                                                        p_coefficient_2);

  auto superimposed_r =
    std::make_shared<Ddhdg::SuperimposedRecombinationTerm<dim>>(r1, r2, r3);

  const double value1 =
    r1->compute_recombination_term(n, p, dealii::Point<dim>(), 1);
  const double value2 =
    r2->compute_recombination_term(n, p, dealii::Point<dim>(), 1);
  const double value3 =
    r3->compute_recombination_term(n, p, dealii::Point<dim>(), 1);

  const auto tot_value =
    superimposed_r->compute_recombination_term(n, p, dealii::Point<dim>(), 1);

  EXPECT_DOUBLE_EQ(value1 + value2 + value3, tot_value);

  const double der_n_1 = r1->compute_derivative_of_recombination_term(
    n, p, dealii::Point<dim>(), 1, Ddhdg::Component::n);
  const double der_n_2 = r2->compute_derivative_of_recombination_term(
    n, p, dealii::Point<dim>(), 1, Ddhdg::Component::n);
  const double der_n_3 = r3->compute_derivative_of_recombination_term(
    n, p, dealii::Point<dim>(), 1, Ddhdg::Component::n);

  const auto tot_der_n =
    superimposed_r->compute_derivative_of_recombination_term(
      n, p, dealii::Point<dim>(), 1, Ddhdg::Component::n);

  EXPECT_DOUBLE_EQ(der_n_1 + der_n_2 + der_n_3, tot_der_n);

  const double der_p_1 = r1->compute_derivative_of_recombination_term(
    n, p, dealii::Point<dim>(), 1, Ddhdg::Component::p);
  const double der_p_2 = r2->compute_derivative_of_recombination_term(
    n, p, dealii::Point<dim>(), 1, Ddhdg::Component::p);
  const double der_p_3 = r3->compute_derivative_of_recombination_term(
    n, p, dealii::Point<dim>(), 1, Ddhdg::Component::p);

  const auto tot_der_p =
    superimposed_r->compute_derivative_of_recombination_term(
      n, p, dealii::Point<dim>(), 1, Ddhdg::Component::p);

  EXPECT_DOUBLE_EQ(der_p_1 + der_p_2 + der_p_3, tot_der_p);
}


template <typename T>
class RecombinationTerms : public ::testing::Test
{
public:
  static constexpr unsigned int dim = T::value;
};

TYPED_TEST_SUITE(RecombinationTerms, dimensions, );



TYPED_TEST(RecombinationTerms, multiple_evaluation) // NOLINT
{
  // Check that compute multiple terms return the same values that we can obtain
  // calling multiple times the function on a single point
  constexpr unsigned int dim = TypeParam::value;

  std::vector<std::shared_ptr<Ddhdg::RecombinationTerm<dim>>>
    recombination_terms;

  std::vector<double> n(100);
  std::vector<double> p(100);

  std::vector<dealii::Point<dim>> points(100);

  std::vector<double> values(100);
  std::vector<double> der_n(100);
  std::vector<double> der_p(100);

  for (unsigned int i = 0; i < 100; ++i)
    {
      n[i] = i;
      p[i] = 99. - i;

      points[i][0] = (i * i) / 1000.;

      if (dim > 1)
        points[i][1] = 1 - i / 100.;

      if (dim > 2)
        points[i][2] = (i - 50) * (i - 50) * (i - 50) / 100000;
    }

  constexpr double                       Nc = 1.5;
  constexpr double                       Nv = 1.3;
  constexpr double                       Ec = .5e-10;
  constexpr double                       Ev = .7e-11;
  std::shared_ptr<dealii::Function<dim>> temperature =
    std::make_shared<dealii::Functions::ConstantFunction<dim>>(300);

  auto shockley_read_hall_fixed_temperature =
    std::make_shared<Ddhdg::ShockleyReadHallFixedTemperature<dim>>(1., 2., 3.);
  auto auger_fixed_temperature =
    std::make_shared<Ddhdg::AugerFixedTemperature<dim>>(1., 2., 3.);

  auto shockley_read_hall = std::make_shared<Ddhdg::ShockleyReadHall<dim>>(
    Nc, Nv, Ec, Ev, temperature, 1., 5.);

  auto auger =
    std::make_shared<Ddhdg::Auger<dim>>(Nc, Nv, Ec, Ev, temperature, 2., 3.);

  auto superimposed =
    std::make_shared<Ddhdg::SuperimposedRecombinationTerm<dim>>(
      shockley_read_hall_fixed_temperature, auger_fixed_temperature);

  recombination_terms.push_back(shockley_read_hall_fixed_temperature);
  recombination_terms.push_back(auger_fixed_temperature);
  recombination_terms.push_back(shockley_read_hall);
  recombination_terms.push_back(auger);
  recombination_terms.push_back(superimposed);

  for (auto r : recombination_terms)
    {
      r->compute_multiple_recombination_terms(n, p, points, 1, true, values);
      r->compute_multiple_derivatives_of_recombination_terms(
        n, p, points, 1, Ddhdg::Component::n, true, der_n);
      r->compute_multiple_derivatives_of_recombination_terms(
        n, p, points, 1, Ddhdg::Component::p, true, der_p);

      for (unsigned int q = 0; q < 100; ++q)
        {
          const double current_value =
            r->compute_recombination_term(n[q], p[q], points[q], 1);
          EXPECT_DOUBLE_EQ(values[q], current_value);

          const double current_der_n =
            r->compute_derivative_of_recombination_term(
              n[q], p[q], points[q], 1, Ddhdg::Component::n);
          EXPECT_DOUBLE_EQ(der_n[q], current_der_n);

          const double current_der_p =
            r->compute_derivative_of_recombination_term(
              n[q], p[q], points[q], 1, Ddhdg::Component::p);
          EXPECT_DOUBLE_EQ(der_p[q], current_der_p);
        }
    }
}
