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

#ifdef USE_BOOST_PYTHON
#  include <boost/python.hpp>
#  define PYSPACE boost::python
#endif

#ifdef USE_PYBIND11
#  include <pybind11/pybind11.h>
#  define PYSPACE pybind11
#endif

#include <dealii-python-bindings/point_wrapper.h>
#include <dealii-python-bindings/tria_accessor_wrapper.h>
#include <dealii-python-bindings/triangulation_wrapper.h>

DEAL_II_NAMESPACE_OPEN

namespace python
{
  namespace internal
  {
    template <int structdim, int dim, int spacedim>
    PointWrapper
    get_barycenter(const void *tria_accessor)
    {
      const TriaAccessor<structdim, dim, spacedim> *accessor =
        static_cast<const TriaAccessor<structdim, dim, spacedim> *>(
          tria_accessor);
      Point<spacedim> barycenter = accessor->barycenter();
      PYSPACE::list   barycenter_list;
      for (int i = 0; i < spacedim; ++i)
        barycenter_list.append(barycenter[i]);

      return PointWrapper(barycenter_list);
    }



    template <int structdim, int dim, int spacedim>
    PointWrapper
    get_center(const bool  respect_manifold,
               const bool  interpolate_from_surrounding,
               const void *tria_accessor)
    {
      const TriaAccessor<structdim, dim, spacedim> *accessor =
        static_cast<const TriaAccessor<structdim, dim, spacedim> *>(
          tria_accessor);
      Point<spacedim> center =
        accessor->center(respect_manifold, interpolate_from_surrounding);
      PYSPACE::list center_list;
      for (int i = 0; i < spacedim; ++i)
        center_list.append(center[i]);

      return PointWrapper(center_list);
    }



    template <int structdim, int dim, int spacedim>
    void
    set_boundary_id(const int boundary_id, void *tria_accessor)
    {
      TriaAccessor<structdim, dim, spacedim> *accessor =
        static_cast<TriaAccessor<structdim, dim, spacedim> *>(tria_accessor);
      accessor->set_boundary_id(boundary_id);
    }



    template <int structdim, int dim, int spacedim>
    int
    get_boundary_id(const void *tria_accessor)
    {
      const TriaAccessor<structdim, dim, spacedim> *accessor =
        static_cast<const TriaAccessor<structdim, dim, spacedim> *>(
          tria_accessor);

      return accessor->boundary_id();
    }



    template <int structdim, int dim, int spacedim>
    void
    set_all_boundary_ids(const int boundary_id, void *tria_accessor)
    {
      TriaAccessor<structdim, dim, spacedim> *accessor =
        static_cast<TriaAccessor<structdim, dim, spacedim> *>(tria_accessor);
      accessor->set_all_boundary_ids(boundary_id);
    }


    template <int structdim, int dim, int spacedim>
    void
    set_vertex(const int i, PointWrapper &point_wrapper, void *tria_accessor)
    {
      TriaAccessor<structdim, dim, spacedim> *accessor =
        static_cast<TriaAccessor<structdim, dim, spacedim> *>(tria_accessor);
      Point<spacedim> *point =
        static_cast<Point<spacedim> *>(point_wrapper.get_point());

      accessor->vertex(i) = *point;
    }



    template <int structdim, int dim, int spacedim>
    PointWrapper
    get_vertex(const int i, const void *tria_accessor)
    {
      const TriaAccessor<structdim, dim, spacedim> *accessor =
        static_cast<const TriaAccessor<structdim, dim, spacedim> *>(
          tria_accessor);
      Point<spacedim> vertex = accessor->vertex(i);

      PYSPACE::list coordinates;
      for (int i = 0; i < spacedim; ++i)
        coordinates.append(vertex[i]);

      return PointWrapper(coordinates);
    }



    template <int structdim, int dim, int spacedim>
    void
    set_manifold_id(const int manifold_id, void *tria_accessor)
    {
      TriaAccessor<structdim, dim, spacedim> *accessor =
        static_cast<TriaAccessor<structdim, dim, spacedim> *>(tria_accessor);
      accessor->set_manifold_id(manifold_id);
    }



    template <int structdim, int dim, int spacedim>
    int
    get_manifold_id(const void *tria_accessor)
    {
      const TriaAccessor<structdim, dim, spacedim> *accessor =
        static_cast<const TriaAccessor<structdim, dim, spacedim> *>(
          tria_accessor);
      return accessor->manifold_id();
    }



    template <int structdim, int dim, int spacedim>
    bool
    at_boundary(const void *tria_accessor)
    {
      const TriaAccessor<structdim, dim, spacedim> *accessor =
        static_cast<const TriaAccessor<structdim, dim, spacedim> *>(
          tria_accessor);
      return accessor->at_boundary();
    }


    template <int structdim, int dim, int spacedim>
    double
    measure(const void *tria_accessor)
    {
      const TriaAccessor<structdim, dim, spacedim> *accessor =
        static_cast<const TriaAccessor<structdim, dim, spacedim> *>(
          tria_accessor);
      return accessor->measure();
    }
  } // namespace internal



  TriaAccessorWrapper::TriaAccessorWrapper(const TriaAccessorWrapper &other)
    : structdim(other.structdim)
    , dim(other.dim)
    , spacedim(other.spacedim)
  {
    if ((dim == 2) && (spacedim == 2) && (structdim == 1))
      {
        TriaAccessor<1, 2, 2> *other_accessor =
          static_cast<TriaAccessor<1, 2, 2> *>(other.tria_accessor);
        tria_accessor = new TriaAccessor<1, 2, 2>(*other_accessor);
      }
    else if ((dim == 2) && (spacedim == 3) && (structdim == 1))
      {
        TriaAccessor<1, 2, 3> *other_accessor =
          static_cast<TriaAccessor<1, 2, 3> *>(other.tria_accessor);
        tria_accessor = new TriaAccessor<1, 2, 3>(*other_accessor);
      }
    else if ((dim == 3) && (spacedim == 3) && (structdim == 2))
      {
        TriaAccessor<2, 3, 3> *other_accessor =
          static_cast<TriaAccessor<2, 3, 3> *>(other.tria_accessor);
        tria_accessor = new TriaAccessor<2, 3, 3>(*other_accessor);
      }
    else
      AssertThrow(false,
                  ExcMessage("Wrong structdim-dim-spacedim combination."));
  }

  TriaAccessorWrapper::TriaAccessorWrapper(void *    tria_accessor,
                                           const int structdim,
                                           const int dim,
                                           const int spacedim)
    : structdim(structdim)
    , dim(dim)
    , spacedim(spacedim)
    , tria_accessor(tria_accessor)
  {}


  TriaAccessorWrapper::~TriaAccessorWrapper()
  {
    if (dim != -1)
      {
        if ((dim == 2) && (spacedim == 2) && (structdim == 1))
          {
            // We cannot call delete on a void pointer so cast the void pointer
            // back first.
            TriaAccessor<1, 2, 2> *tmp =
              static_cast<TriaAccessor<1, 2, 2> *>(tria_accessor);
            delete tmp;
          }
        else if ((dim == 2) && (spacedim == 3) && (structdim == 1))
          {
            TriaAccessor<1, 2, 3> *tmp =
              static_cast<TriaAccessor<1, 2, 3> *>(tria_accessor);
            delete tmp;
          }
        else
          {
            TriaAccessor<2, 3, 3> *tmp =
              static_cast<TriaAccessor<2, 3, 3> *>(tria_accessor);
            delete tmp;
          }

        dim           = -1;
        spacedim      = -1;
        structdim     = -1;
        tria_accessor = nullptr;
      }
  }



  PointWrapper
  TriaAccessorWrapper::get_barycenter() const
  {
    if ((dim == 2) && (spacedim == 2) && (structdim == 1))
      return internal::get_barycenter<1, 2, 2>(tria_accessor);
    else if ((dim == 2) && (spacedim == 3) && (structdim == 1))
      return internal::get_barycenter<1, 2, 3>(tria_accessor);
    else
      return internal::get_barycenter<2, 3, 3>(tria_accessor);
  }



  PointWrapper
  TriaAccessorWrapper::get_center(const bool respect_manifold,
                                  const bool interpolate_from_surrounding) const
  {
    if ((dim == 2) && (spacedim == 2) && (structdim == 1))
      return internal::get_center<1, 2, 2>(respect_manifold,
                                           interpolate_from_surrounding,
                                           tria_accessor);
    else if ((dim == 2) && (spacedim == 3) && (structdim == 1))
      return internal::get_center<1, 2, 3>(respect_manifold,
                                           interpolate_from_surrounding,
                                           tria_accessor);
    else
      return internal::get_center<2, 3, 3>(respect_manifold,
                                           interpolate_from_surrounding,
                                           tria_accessor);
  }



  void
  TriaAccessorWrapper::set_boundary_id(const int boundary_id)
  {
    if ((dim == 2) && (spacedim == 2) && (structdim == 1))
      internal::set_boundary_id<1, 2, 2>(boundary_id, tria_accessor);
    else if ((dim == 2) && (spacedim == 3) && (structdim == 1))
      internal::set_boundary_id<1, 2, 3>(boundary_id, tria_accessor);
    else
      internal::set_boundary_id<2, 3, 3>(boundary_id, tria_accessor);
  }



  void
  TriaAccessorWrapper::set_all_boundary_ids(const int boundary_id)
  {
    if ((dim == 2) && (spacedim == 2) && (structdim == 1))
      internal::set_all_boundary_ids<1, 2, 2>(boundary_id, tria_accessor);
    else if ((dim == 2) && (spacedim == 3) && (structdim == 1))
      internal::set_all_boundary_ids<1, 2, 3>(boundary_id, tria_accessor);
    else
      internal::set_all_boundary_ids<2, 3, 3>(boundary_id, tria_accessor);
  }



  int
  TriaAccessorWrapper::get_boundary_id() const
  {
    if ((dim == 2) && (spacedim == 2) && (structdim == 1))
      return internal::get_boundary_id<1, 2, 2>(tria_accessor);
    else if ((dim == 2) && (spacedim == 3) && (structdim == 1))
      return internal::get_boundary_id<1, 2, 3>(tria_accessor);
    else
      return internal::get_boundary_id<2, 3, 3>(tria_accessor);
  }



  void
  TriaAccessorWrapper::set_vertex(const int i, PointWrapper &point_wrapper)
  {
    AssertThrow(i < static_cast<int>(Utilities::pow(2, dim)),
                ExcVertexDoesNotExist(i, Utilities::pow(2, dim)));
    if ((dim == 2) && (spacedim == 2) && (structdim == 1))
      internal::set_vertex<1, 2, 2>(i, point_wrapper, tria_accessor);
    else if ((dim == 2) && (spacedim == 3) && (structdim == 1))
      internal::set_vertex<1, 2, 3>(i, point_wrapper, tria_accessor);
    else
      internal::set_vertex<2, 3, 3>(i, point_wrapper, tria_accessor);
  }



  PointWrapper
  TriaAccessorWrapper::get_vertex(const int i) const
  {
    AssertThrow(i < static_cast<int>(Utilities::pow(2, dim)),
                ExcVertexDoesNotExist(i, Utilities::pow(2, dim)));
    if ((dim == 2) && (spacedim == 2) && (structdim == 1))
      return internal::get_vertex<1, 2, 2>(i, tria_accessor);
    else if ((dim == 2) && (spacedim == 3) && (structdim == 1))
      return internal::get_vertex<1, 2, 3>(i, tria_accessor);
    else
      return internal::get_vertex<2, 3, 3>(i, tria_accessor);
  }



  void
  TriaAccessorWrapper::set_manifold_id(const int manifold_id)
  {
    if ((dim == 2) && (spacedim == 2) && (structdim == 1))
      internal::set_manifold_id<1, 2, 2>(manifold_id, tria_accessor);
    else if ((dim == 2) && (spacedim == 3) && (structdim == 1))
      internal::set_manifold_id<1, 2, 3>(manifold_id, tria_accessor);
    else
      internal::set_manifold_id<2, 3, 3>(manifold_id, tria_accessor);
  }



  int
  TriaAccessorWrapper::get_manifold_id() const
  {
    if ((dim == 2) && (spacedim == 2) && (structdim == 1))
      return internal::get_manifold_id<1, 2, 2>(tria_accessor);
    else if ((dim == 2) && (spacedim == 3) && (structdim == 1))
      return internal::get_manifold_id<1, 2, 3>(tria_accessor);
    else
      return internal::get_manifold_id<2, 3, 3>(tria_accessor);
  }



  bool
  TriaAccessorWrapper::at_boundary() const
  {
    if ((dim == 2) && (spacedim == 2) && (structdim == 1))
      return internal::at_boundary<1, 2, 2>(tria_accessor);
    else if ((dim == 2) && (spacedim == 3) && (structdim == 1))
      return internal::at_boundary<1, 2, 3>(tria_accessor);
    else
      return internal::at_boundary<2, 3, 3>(tria_accessor);
  }



  double
  TriaAccessorWrapper::measure() const
  {
    if ((dim == 2) && (spacedim == 2) && (structdim == 1))
      return internal::measure<1, 2, 2>(tria_accessor);
    else if ((dim == 2) && (spacedim == 3) && (structdim == 1))
      return internal::measure<1, 2, 3>(tria_accessor);
    else
      return internal::measure<2, 3, 3>(tria_accessor);
  }

} // namespace python

DEAL_II_NAMESPACE_CLOSE