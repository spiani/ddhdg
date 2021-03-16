// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#ifndef dealii_function_wrapper_h
#define dealii_function_wrapper_h

#include <deal.II/base/config.h>

#include <deal.II/base/function.h>

#ifdef USE_BOOST_PYTHON
#  include <boost/python.hpp>
#  define PYSPACE boost::python
#endif

#ifdef USE_PYBIND11
#  include <pybind11/pybind11.h>
#  define PYSPACE pybind11
#endif

DEAL_II_NAMESPACE_OPEN

namespace python
{
  template <int dim>
  class FunctionWrapper : public Function<dim, double>
  {
  public:
    FunctionWrapper(PYSPACE::object &python_function, unsigned n_components)
      : Function<dim, double>(n_components)
      , python_function(python_function)
    {}

    FunctionWrapper(const FunctionWrapper &other)
    {
      python_function = other.python_function;
    }

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      PYSPACE::list p_list_in;

      for (int d = 0; d < dim; ++d)
        p_list_in.append(p[d]);
#ifdef USE_BOOST_PYTHON
      PYSPACE::list p_list_out =
        boost::python::extract<boost::python::list>(python_function(p_list_in));
#endif
#ifdef USE_PYBIND11
      PYSPACE::list p_list_out = python_function(p_list_in);
#endif
      for (size_t i = 0; i < this->n_components; ++i)
#ifdef USE_BOOST_PYTHON
        values[i] = boost::python::extract<double>(p_list_out[i]);
#endif
#ifdef USE_PYBIND11
      values[i] = pybind11::cast<double>(p_list_out[i]);
#endif
    }

  private:
    /**
     * A callback to a python function
     */
    PYSPACE::object python_function;
  };

} // namespace python

DEAL_II_NAMESPACE_CLOSE

#endif
