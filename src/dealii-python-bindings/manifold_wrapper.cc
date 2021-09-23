// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2020 by the deal.II authors
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

#include <deal.II/grid/manifold_lib.h>

#include <dealii-python-bindings/manifold_wrapper.h>
#include <dealii-python-bindings/point_wrapper.h>

DEAL_II_NAMESPACE_OPEN

namespace python
{
  namespace internal
  {
    template <int dim, int spacedim>
    Manifold<dim, spacedim> *
    create_spherical_manifold(const PointWrapper &center)
    {
      const Point<spacedim> *point =
        static_cast<const Point<spacedim> *>(center.get_point());
      return new SphericalManifold<dim, spacedim>(*point);
    }



    template <int dim, int spacedim>
    Manifold<dim, spacedim> *
    create_polar_manifold(const PointWrapper &center)
    {
      const Point<spacedim> *point =
        static_cast<const Point<spacedim> *>(center.get_point());
      return new PolarManifold<dim, spacedim>(*point);
    }



    template <int dim, int spacedim>
    Manifold<dim, spacedim> *
    create_cylindrical_manifold(const int axis, const double tolerance)
    {
      return new CylindricalManifold<dim, spacedim>(axis, tolerance);
    }



    template <int dim, int spacedim>
    Manifold<dim, spacedim> *
    create_cylindrical_manifold(const PYSPACE::list &direction_list,
                                const PYSPACE::list &axial_point_list)
    {
      Tensor<1, spacedim> direction;
      for (int d = 0; d < spacedim; ++d)
#ifdef USE_BOOST_PYTHON
        direction[d] = boost::python::extract<double>(direction_list[d]);
#endif
#ifdef USE_PYBIND11
      direction[d] = pybind11::cast<double>(direction_list[d]);
#endif

      Point<spacedim> axial_point;
      for (int d = 0; d < spacedim; ++d)
#ifdef USE_BOOST_PYTHON
        axial_point[d] = boost::python::extract<double>(axial_point_list[d]);
#endif
#ifdef USE_PYBIND11
      axial_point[d] = pybind11::cast<double>(axial_point_list[d]);
#endif
      return new CylindricalManifold<dim, spacedim>(direction, axial_point);
    }



    template <int dim, int spacedim>
    Manifold<dim, spacedim> *
    create_function_manifold(const std::string &push_forward,
                             const std::string &pull_back)
    {
      return new FunctionManifold<dim, spacedim>(push_forward, pull_back);
    }



    template <int dim, int spacedim>
    Manifold<dim, spacedim> *
    create_function_manifold(PYSPACE::object &push_forward,
                             PYSPACE::object &pull_back)
    {
      return new FunctionManifold<dim, spacedim>(
        std::make_unique<FunctionWrapper<dim>>(push_forward, spacedim),
        std::make_unique<FunctionWrapper<spacedim>>(pull_back, dim));
    }



    template <int dim, int spacedim>
    Manifold<dim, spacedim> *
    clone(void *manifold_ptr)
    {
      const Manifold<dim, spacedim> *manifold =
        static_cast<const Manifold<dim, spacedim> *>(manifold_ptr);

      if (const SphericalManifold<dim, spacedim> *d =
            dynamic_cast<const SphericalManifold<dim, spacedim> *>(manifold))
        {
          return new SphericalManifold<dim, spacedim>(*d);
        }
      else if (const PolarManifold<dim, spacedim> *d =
                 dynamic_cast<const PolarManifold<dim, spacedim> *>(manifold))
        {
          return new PolarManifold<dim, spacedim>(*d);
        }
      else if (const CylindricalManifold<dim, spacedim> *d =
                 dynamic_cast<const CylindricalManifold<dim, spacedim> *>(
                   manifold))
        {
          return new CylindricalManifold<dim, spacedim>(*d);
        }
      else if (const FunctionManifold<dim, spacedim> *d =
                 dynamic_cast<const FunctionManifold<dim, spacedim> *>(
                   manifold))
        {
          return new FunctionManifold<dim, spacedim>(*d);
        }
      else
        ExcMessage("Unsupported manifold type in clone.");

      return nullptr;
    }
  } // namespace internal



  ManifoldWrapper::ManifoldWrapper(const int dim, const int spacedim)
    : dim(dim)
    , spacedim(spacedim)
    , manifold_ptr(nullptr)
  {
    AssertThrow(((dim == 2) && (spacedim == 2)) ||
                  ((dim == 2) && (spacedim == 3)) ||
                  ((dim == 3) && (spacedim == 3)),
                ExcMessage("Wrong dim-spacedim combination."));
  }



  ManifoldWrapper::ManifoldWrapper(const ManifoldWrapper &other)
  {
    dim      = other.dim;
    spacedim = other.spacedim;

    AssertThrow(other.manifold_ptr != nullptr,
                ExcMessage("Underlying manifold does not exist."));

    if ((dim == 2) && (spacedim == 2))
      {
        manifold_ptr = internal::clone<2, 2>(other.manifold_ptr);
      }
    else if ((dim == 2) && (spacedim == 3))
      {
        manifold_ptr = internal::clone<2, 3>(other.manifold_ptr);
      }
    else if ((dim == 3) && (spacedim == 3))
      {
        manifold_ptr = internal::clone<3, 3>(other.manifold_ptr);
      }
    else
      AssertThrow(false,
                  ExcMessage(
                    "Given dim-spacedim combination is not implemented."));
  }



  ManifoldWrapper::~ManifoldWrapper()
  {
    if (dim != -1)
      {
        if ((dim == 2) && (spacedim == 2))
          {
            // We cannot call delete on a void pointer so cast the void pointer
            // back first.
            Manifold<2, 2> *tmp = static_cast<Manifold<2, 2> *>(manifold_ptr);
            delete tmp;
          }
        else if ((dim == 2) && (spacedim == 3))
          {
            Manifold<2, 3> *tmp = static_cast<Manifold<2, 3> *>(manifold_ptr);
            delete tmp;
          }
        else
          {
            Manifold<3, 3> *tmp = static_cast<Manifold<3, 3> *>(manifold_ptr);
            delete tmp;
          }

        dim          = -1;
        spacedim     = -1;
        manifold_ptr = nullptr;
      }
  }



  void
  ManifoldWrapper::create_spherical(const PointWrapper center)
  {
    if ((dim == 2) && (spacedim == 2))
      {
        manifold_ptr = internal::create_spherical_manifold<2, 2>(center);
      }
    else if ((dim == 2) && (spacedim == 3))
      {
        manifold_ptr = internal::create_spherical_manifold<2, 3>(center);
      }
    else if ((dim == 3) && (spacedim == 3))
      {
        manifold_ptr = internal::create_spherical_manifold<3, 3>(center);
      }
    else
      AssertThrow(false,
                  ExcMessage(
                    "Given dim-spacedim combination is not implemented."));
  }



  void
  ManifoldWrapper::create_polar(const PointWrapper center)
  {
    if ((dim == 2) && (spacedim == 2))
      {
        manifold_ptr = internal::create_polar_manifold<2, 2>(center);
      }
    else if ((dim == 2) && (spacedim == 3))
      {
        manifold_ptr = internal::create_polar_manifold<2, 3>(center);
      }
    else if ((dim == 3) && (spacedim == 3))
      {
        manifold_ptr = internal::create_polar_manifold<3, 3>(center);
      }
    else
      AssertThrow(false,
                  ExcMessage(
                    "Given dim-spacedim combination is not implemented."));
  }



  void
  ManifoldWrapper::create_cylindrical(const int axis, const double tolerance)
  {
    if ((dim == 2) && (spacedim == 3))
      {
        manifold_ptr =
          internal::create_cylindrical_manifold<2, 3>(axis, tolerance);
      }
    else if ((dim == 3) && (spacedim == 3))
      {
        manifold_ptr =
          internal::create_cylindrical_manifold<3, 3>(axis, tolerance);
      }
    else
      AssertThrow(false,
                  ExcMessage(
                    "Given dim-spacedim combination is not implemented."));
  }



  void
  ManifoldWrapper::create_cylindrical(const PYSPACE::list &direction,
                                      const PYSPACE::list &axial_point)
  {
    if ((dim == 2) && (spacedim == 3))
      {
        manifold_ptr =
          internal::create_cylindrical_manifold<2, 3>(direction, axial_point);
      }
    else if ((dim == 3) && (spacedim == 3))
      {
        manifold_ptr =
          internal::create_cylindrical_manifold<3, 3>(direction, axial_point);
      }
    else
      AssertThrow(false,
                  ExcMessage(
                    "Given dim-spacedim combination is not implemented."));
  }



  void
  ManifoldWrapper::create_function_string(const std::string &push_forward,
                                          const std::string &pull_back)
  {
    if ((dim == 2) && (spacedim == 2))
      {
        manifold_ptr =
          internal::create_function_manifold<2, 2>(push_forward, pull_back);
      }
    else if ((dim == 2) && (spacedim == 3))
      {
        manifold_ptr =
          internal::create_function_manifold<2, 3>(push_forward, pull_back);
      }
    else if ((dim == 3) && (spacedim == 3))
      {
        manifold_ptr =
          internal::create_function_manifold<3, 3>(push_forward, pull_back);
      }
    else
      AssertThrow(false,
                  ExcMessage(
                    "Given dim-spacedim combination is not implemented."));
  }



  void
  ManifoldWrapper::create_function(PYSPACE::object &push_forward,
                                   PYSPACE::object &pull_back)
  {
    if ((dim == 2) && (spacedim == 2))
      {
        manifold_ptr =
          internal::create_function_manifold<2, 2>(push_forward, pull_back);
      }
    else if ((dim == 2) && (spacedim == 3))
      {
        manifold_ptr =
          internal::create_function_manifold<2, 3>(push_forward, pull_back);
      }
    else if ((dim == 3) && (spacedim == 3))
      {
        manifold_ptr =
          internal::create_function_manifold<3, 3>(push_forward, pull_back);
      }
    else
      AssertThrow(false,
                  ExcMessage(
                    "Given dim-spacedim combination is not implemented."));
  }



  void *
  ManifoldWrapper::get_manifold() const
  {
    return manifold_ptr;
  }



  int
  ManifoldWrapper::get_dim() const
  {
    return dim;
  }



  int
  ManifoldWrapper::get_spacedim() const
  {
    return spacedim;
  }


} // namespace python

DEAL_II_NAMESPACE_CLOSE
