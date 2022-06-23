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



    template <int d, int sd>
    PointWrapper
    get_barycenter_wrapper(const void *tria_accessor,
                           const int   structdim,
                           const int   dim,
                           const int   spacedim)
    {
      if constexpr (sd < d)
        AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));

      if constexpr (d > 1)
        if (dim != d)
          return get_barycenter_wrapper<d - 1, sd>(tria_accessor,
                                                   structdim,
                                                   dim,
                                                   spacedim);
      if (dim != d)
        AssertThrow(false, ExcMessage("Unsupported dimension."));

      if constexpr (sd > 1)
        if (spacedim != sd)
          return get_barycenter_wrapper<d, sd - 1>(tria_accessor,
                                                   structdim,
                                                   dim,
                                                   spacedim);
      if (spacedim != sd)
        AssertThrow(false, ExcMessage("Unsupported space dimension"));

      if constexpr (d == 1)
        {
          Assert(structdim <= 1,
                 ExcMessage("structdim must be smaller than dim (1)"));
          if (structdim == 1)
            return get_barycenter<1, 1, sd>(tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr (d == 2)
        {
          Assert(structdim <= 2,
                 ExcMessage("structdim must be smaller than dim (2)"));
          if (structdim == 1)
            return get_barycenter<1, 2, sd>(tria_accessor);
          else if (structdim == 2)
            return get_barycenter<2, 2, sd>(tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr (d == 3)
        {
          Assert(structdim <= 3,
                 ExcMessage("structdim must be smaller than dim (3)"));
          if (structdim == 1)
            return get_barycenter<1, 3, sd>(tria_accessor);
          else if (structdim == 2)
            return get_barycenter<2, 3, sd>(tria_accessor);
          else if (structdim == 3)
            return get_barycenter<3, 3, sd>(tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
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
      if constexpr (structdim == 0)
        {
          (void)respect_manifold;
          (void)interpolate_from_surrounding;
          Point<spacedim> center = accessor->center();
          PYSPACE::list   center_list;
          for (int i = 0; i < spacedim; ++i)
            center_list.append(center[i]);

          return PointWrapper(center_list);
        }

      if constexpr (structdim > 0)
        {
          Point<spacedim> center =
            accessor->center(respect_manifold, interpolate_from_surrounding);
          PYSPACE::list center_list;
          for (int i = 0; i < spacedim; ++i)
            center_list.append(center[i]);

          return PointWrapper(center_list);
        }
    }



    template <int d, int sd>
    PointWrapper
    get_center_wrapper(const bool  respect_manifold,
                       const bool  interpolate_from_surrounding,
                       const void *tria_accessor,
                       const int   structdim,
                       const int   dim,
                       const int   spacedim)
    {
      if constexpr (sd < d)
        AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));

      if constexpr (sd >= d)
        {
          if constexpr (d > 1)
            if (dim != d)
              return get_center_wrapper<d - 1, sd>(respect_manifold,
                                                   interpolate_from_surrounding,
                                                   tria_accessor,
                                                   structdim,
                                                   dim,
                                                   spacedim);
          if (dim != d)
            AssertThrow(false, ExcMessage("Unsupported dimension."));

          if constexpr (sd > 1)
            if (spacedim != sd)
              return get_center_wrapper<d, sd - 1>(respect_manifold,
                                                   interpolate_from_surrounding,
                                                   tria_accessor,
                                                   structdim,
                                                   dim,
                                                   spacedim);
          if (spacedim != sd)
            AssertThrow(false, ExcMessage("Unsupported space dimension"));

          if constexpr (d == 1)
            {
              Assert(structdim <= 1,
                     ExcMessage("structdim must be smaller than dim (1)"));
              if (structdim == 0)
                return get_center<0, 1, sd>(respect_manifold,
                                            interpolate_from_surrounding,
                                            tria_accessor);
              else if (structdim == 1)
                return get_center<1, 1, sd>(respect_manifold,
                                            interpolate_from_surrounding,
                                            tria_accessor);
              else
                AssertThrow(false, ExcMessage("Invalid structdim"));
            }
          else if constexpr (d == 2)
            {
              Assert(structdim <= 2,
                     ExcMessage("structdim must be smaller than dim (2)"));
              if (structdim == 0)
                return get_center<0, 2, sd>(respect_manifold,
                                            interpolate_from_surrounding,
                                            tria_accessor);
              else if (structdim == 1)
                return get_center<1, 2, sd>(respect_manifold,
                                            interpolate_from_surrounding,
                                            tria_accessor);
              else if (structdim == 2)
                return get_center<2, 2, sd>(respect_manifold,
                                            interpolate_from_surrounding,
                                            tria_accessor);
              else
                AssertThrow(false, ExcMessage("Invalid structdim"));
            }
          else if constexpr (d == 3)
            {
              Assert(structdim <= 3,
                     ExcMessage("structdim must be smaller than dim (3)"));
              if (structdim == 0)
                return get_center<0, 3, sd>(respect_manifold,
                                            interpolate_from_surrounding,
                                            tria_accessor);
              else if (structdim == 1)
                return get_center<1, 3, sd>(respect_manifold,
                                            interpolate_from_surrounding,
                                            tria_accessor);
              else if (structdim == 2)
                return get_center<2, 3, sd>(respect_manifold,
                                            interpolate_from_surrounding,
                                            tria_accessor);
              else if (structdim == 3)
                return get_center<3, 3, sd>(respect_manifold,
                                            interpolate_from_surrounding,
                                            tria_accessor);
              else
                AssertThrow(false, ExcMessage("Invalid structdim"));
            }
        }
    }



    template <int structdim, int dim, int spacedim>
    void
    set_boundary_id(const int boundary_id, void *tria_accessor)
    {
      TriaAccessor<structdim, dim, spacedim> *accessor =
        static_cast<TriaAccessor<structdim, dim, spacedim> *>(tria_accessor);
      accessor->set_boundary_id(boundary_id);
    }



    template <int d, int sd>
    void
    set_boundary_id_wrapper(const int boundary_id,
                            void     *tria_accessor,
                            const int structdim,
                            const int dim,
                            const int spacedim)
    {
      if constexpr (sd < d)
        AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));

      if constexpr (d > 1)
        if (dim != d)
          return set_boundary_id_wrapper<d - 1, sd>(
            boundary_id, tria_accessor, structdim, dim, spacedim);
      if (dim != d)
        AssertThrow(false, ExcMessage("Unsupported dimension."));

      if constexpr (sd > 1)
        if (spacedim != sd)
          return set_boundary_id_wrapper<d, sd - 1>(
            boundary_id, tria_accessor, structdim, dim, spacedim);
      if (spacedim != sd)
        AssertThrow(false, ExcMessage("Unsupported space dimension"));

      if constexpr (d == 1)
        {
          Assert(structdim <= 1,
                 ExcMessage("structdim must be smaller than dim (1)"));
          if (structdim == 1)
            return set_boundary_id<1, 1, sd>(boundary_id, tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 2) && (sd >= 2))
        {
          Assert(structdim <= 2,
                 ExcMessage("structdim must be smaller than dim (2)"));
          if (structdim == 1)
            return set_boundary_id<1, 2, sd>(boundary_id, tria_accessor);
          else if (structdim == 2)
            return set_boundary_id<2, 2, sd>(boundary_id, tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 3) && (sd >= 3))
        {
          Assert(structdim <= 3,
                 ExcMessage("structdim must be smaller than dim (3)"));
          if (structdim == 1)
            return set_boundary_id<1, 3, sd>(boundary_id, tria_accessor);
          else if (structdim == 2)
            return set_boundary_id<2, 3, sd>(boundary_id, tria_accessor);
          else if (structdim == 3)
            return set_boundary_id<3, 3, sd>(boundary_id, tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
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



    template <int d, int sd>
    int
    get_boundary_id_wrapper(const void *tria_accessor,
                            const int   structdim,
                            const int   dim,
                            const int   spacedim)
    {
      if constexpr (sd < d)
        AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));

      if constexpr (d > 1)
        if (dim != d)
          return get_boundary_id_wrapper<d - 1, sd>(tria_accessor,
                                                    structdim,
                                                    dim,
                                                    spacedim);
      if (dim != d)
        AssertThrow(false, ExcMessage("Unsupported dimension."));

      if constexpr (sd > 1)
        if (spacedim != sd)
          return get_boundary_id_wrapper<d, sd - 1>(tria_accessor,
                                                    structdim,
                                                    dim,
                                                    spacedim);
      if (spacedim != sd)
        AssertThrow(false, ExcMessage("Unsupported space dimension"));

      if constexpr (d == 1)
        {
          Assert(structdim <= 1,
                 ExcMessage("structdim must be smaller than dim (1)"));
          if (structdim == 1)
            return get_boundary_id<1, 1, sd>(tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 2) && (sd >= 2))
        {
          Assert(structdim <= 2,
                 ExcMessage("structdim must be smaller than dim (2)"));
          if (structdim == 1)
            return get_boundary_id<1, 2, sd>(tria_accessor);
          else if (structdim == 2)
            return get_boundary_id<2, 2, sd>(tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 3) && (sd >= 3))
        {
          Assert(structdim <= 3,
                 ExcMessage("structdim must be smaller than dim (3)"));
          if (structdim == 1)
            return get_boundary_id<1, 3, sd>(tria_accessor);
          else if (structdim == 2)
            return get_boundary_id<2, 3, sd>(tria_accessor);
          else if (structdim == 3)
            return get_boundary_id<3, 3, sd>(tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
    }



    template <int structdim, int dim, int spacedim>
    void
    set_all_boundary_ids(const int boundary_id, void *tria_accessor)
    {
      TriaAccessor<structdim, dim, spacedim> *accessor =
        static_cast<TriaAccessor<structdim, dim, spacedim> *>(tria_accessor);
      accessor->set_all_boundary_ids(boundary_id);
    }



    template <int d, int sd>
    void
    set_all_boundary_ids_wrapper(const int boundary_id,
                                 void     *tria_accessor,
                                 const int structdim,
                                 const int dim,
                                 const int spacedim)
    {
      if constexpr (sd < d)
        AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));

      if constexpr (d > 1)
        if (dim != d)
          return set_all_boundary_ids_wrapper<d - 1, sd>(
            boundary_id, tria_accessor, structdim, dim, spacedim);
      if (dim != d)
        AssertThrow(false, ExcMessage("Unsupported dimension."));

      if constexpr (sd > 1)
        if (spacedim != sd)
          return set_all_boundary_ids_wrapper<d, sd - 1>(
            boundary_id, tria_accessor, structdim, dim, spacedim);
      if (spacedim != sd)
        AssertThrow(false, ExcMessage("Unsupported space dimension"));

      if constexpr ((d == 1) && (sd >= 1))
        {
          Assert(structdim <= 1,
                 ExcMessage("structdim must be smaller than dim (1)"));
          if (structdim == 1)
            return set_all_boundary_ids<1, 1, sd>(boundary_id, tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 2) && (sd >= 2))
        {
          Assert(structdim <= 2,
                 ExcMessage("structdim must be smaller than dim (2)"));
          if (structdim == 1)
            return set_all_boundary_ids<1, 2, sd>(boundary_id, tria_accessor);
          else if (structdim == 2)
            return set_all_boundary_ids<2, 2, sd>(boundary_id, tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 3) && (sd >= 3))
        {
          Assert(structdim <= 3,
                 ExcMessage("structdim must be smaller than dim (3)"));
          if (structdim == 1)
            return set_all_boundary_ids<1, 3, sd>(boundary_id, tria_accessor);
          else if (structdim == 2)
            return set_all_boundary_ids<2, 3, sd>(boundary_id, tria_accessor);
          else if (structdim == 3)
            return set_all_boundary_ids<3, 3, sd>(boundary_id, tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
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



    template <int d, int sd>
    void
    set_vertex_wrapper(const int     i,
                       PointWrapper &point_wrapper,
                       void         *tria_accessor,
                       const int     structdim,
                       const int     dim,
                       const int     spacedim)
    {
      if constexpr (sd < d)
        AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));

      if constexpr (d > 1)
        if (dim != d)
          return set_vertex_wrapper<d - 1, sd>(
            i, point_wrapper, tria_accessor, structdim, dim, spacedim);
      if (dim != d)
        AssertThrow(false, ExcMessage("Unsupported dimension."));

      if constexpr (sd > 1)
        if (spacedim != sd)
          return set_vertex_wrapper<d, sd - 1>(
            i, point_wrapper, tria_accessor, structdim, dim, spacedim);
      if (spacedim != sd)
        AssertThrow(false, ExcMessage("Unsupported space dimension"));

      if constexpr ((d == 1) && (sd >= 1))
        {
          Assert(structdim <= 1,
                 ExcMessage("structdim must be smaller than dim (1)"));
          if (structdim == 0)
            return set_vertex<0, 1, sd>(i, point_wrapper, tria_accessor);
          else if (structdim == 1)
            return set_vertex<1, 1, sd>(i, point_wrapper, tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 2) && (sd >= 2))
        {
          Assert(structdim <= 2,
                 ExcMessage("structdim must be smaller than dim (2)"));
          if (structdim == 0)
            return set_vertex<0, 2, sd>(i, point_wrapper, tria_accessor);
          else if (structdim == 1)
            return set_vertex<1, 2, sd>(i, point_wrapper, tria_accessor);
          else if (structdim == 2)
            return set_vertex<2, 2, sd>(i, point_wrapper, tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 3) && (sd >= 3))
        {
          Assert(structdim <= 3,
                 ExcMessage("structdim must be smaller than dim (3)"));
          if (structdim == 0)
            return set_vertex<0, 3, sd>(i, point_wrapper, tria_accessor);
          else if (structdim == 1)
            return set_vertex<1, 3, sd>(i, point_wrapper, tria_accessor);
          else if (structdim == 2)
            return set_vertex<2, 3, sd>(i, point_wrapper, tria_accessor);
          else if (structdim == 3)
            return set_vertex<3, 3, sd>(i, point_wrapper, tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
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



    template <int d, int sd>
    PointWrapper
    get_vertex_wrapper(const int   i,
                       const void *tria_accessor,
                       const int   structdim,
                       const int   dim,
                       const int   spacedim)
    {
      if constexpr (sd < d)
        AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));

      if constexpr (d > 1)
        if (dim != d)
          return get_vertex_wrapper<d - 1, sd>(
            i, tria_accessor, structdim, dim, spacedim);
      if (dim != d)
        AssertThrow(false, ExcMessage("Unsupported dimension."));

      if constexpr (sd > 1)
        if (spacedim != sd)
          return get_vertex_wrapper<d, sd - 1>(
            i, tria_accessor, structdim, dim, spacedim);
      if (spacedim != sd)
        AssertThrow(false, ExcMessage("Unsupported space dimension"));

      if constexpr ((d == 1) && (sd >= 1))
        {
          Assert(structdim <= 1,
                 ExcMessage("structdim must be smaller than dim (1)"));
          if (structdim == 0)
            return get_vertex<0, 1, sd>(i, tria_accessor);
          else if (structdim == 1)
            return get_vertex<1, 1, sd>(i, tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 2) && (sd >= 2))
        {
          Assert(structdim <= 2,
                 ExcMessage("structdim must be smaller than dim (2)"));
          if (structdim == 0)
            return get_vertex<0, 2, sd>(i, tria_accessor);
          else if (structdim == 1)
            return get_vertex<1, 2, sd>(i, tria_accessor);
          else if (structdim == 2)
            return get_vertex<2, 2, sd>(i, tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 3) && (sd >= 3))
        {
          Assert(structdim <= 3,
                 ExcMessage("structdim must be smaller than dim (3)"));
          if (structdim == 0)
            return get_vertex<0, 3, sd>(i, tria_accessor);
          else if (structdim == 1)
            return get_vertex<1, 3, sd>(i, tria_accessor);
          else if (structdim == 2)
            return get_vertex<2, 3, sd>(i, tria_accessor);
          else if (structdim == 3)
            return get_vertex<3, 3, sd>(i, tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
    }



    template <int structdim, int dim, int spacedim>
    void
    set_manifold_id(const int manifold_id, void *tria_accessor)
    {
      TriaAccessor<structdim, dim, spacedim> *accessor =
        static_cast<TriaAccessor<structdim, dim, spacedim> *>(tria_accessor);
      accessor->set_manifold_id(manifold_id);
    }



    template <int d, int sd>
    void
    set_manifold_id_wrapper(const int manifold_id,
                            void     *tria_accessor,
                            const int structdim,
                            const int dim,
                            const int spacedim)
    {
      if constexpr (sd < d)
        AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));

      if constexpr (d > 1)
        if (dim != d)
          return set_manifold_id_wrapper<d - 1, sd>(
            manifold_id, tria_accessor, structdim, dim, spacedim);
      if (dim != d)
        AssertThrow(false, ExcMessage("Unsupported dimension."));

      if constexpr (sd > 1)
        if (spacedim != sd)
          return set_manifold_id_wrapper<d, sd - 1>(
            manifold_id, tria_accessor, structdim, dim, spacedim);
      if (spacedim != sd)
        AssertThrow(false, ExcMessage("Unsupported space dimension"));

      if constexpr ((d == 1) && (sd >= 1))
        {
          Assert(structdim <= 1,
                 ExcMessage("structdim must be smaller than dim (1)"));
          if (structdim == 1)
            return set_manifold_id<1, 1, sd>(manifold_id, tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 2) && (sd >= 2))
        {
          Assert(structdim <= 2,
                 ExcMessage("structdim must be smaller than dim (2)"));
          if (structdim == 1)
            return set_manifold_id<1, 2, sd>(manifold_id, tria_accessor);
          else if (structdim == 2)
            return set_manifold_id<2, 2, sd>(manifold_id, tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 3) && (sd >= 3))
        {
          Assert(structdim <= 3,
                 ExcMessage("structdim must be smaller than dim (3)"));
          if (structdim == 1)
            return set_manifold_id<1, 3, sd>(manifold_id, tria_accessor);
          else if (structdim == 2)
            return set_manifold_id<2, 3, sd>(manifold_id, tria_accessor);
          else if (structdim == 3)
            return set_manifold_id<3, 3, sd>(manifold_id, tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
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



    template <int d, int sd>
    int
    get_manifold_id_wrapper(const void *tria_accessor,
                            const int   structdim,
                            const int   dim,
                            const int   spacedim)
    {
      if constexpr (sd < d)
        AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));

      if constexpr (d > 1)
        if (dim != d)
          return get_manifold_id_wrapper<d - 1, sd>(tria_accessor,
                                                    structdim,
                                                    dim,
                                                    spacedim);
      if (dim != d)
        AssertThrow(false, ExcMessage("Unsupported dimension."));

      if constexpr (sd > 1)
        if (spacedim != sd)
          return get_manifold_id_wrapper<d, sd - 1>(tria_accessor,
                                                    structdim,
                                                    dim,
                                                    spacedim);
      if (spacedim != sd)
        AssertThrow(false, ExcMessage("Unsupported space dimension"));

      if constexpr ((d == 1) && (sd >= 1))
        {
          Assert(structdim <= 1,
                 ExcMessage("structdim must be smaller than dim (1)"));
          if (structdim == 1)
            return get_manifold_id<1, 1, sd>(tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 2) && (sd >= 2))
        {
          Assert(structdim <= 2,
                 ExcMessage("structdim must be smaller than dim (2)"));
          if (structdim == 1)
            return get_manifold_id<1, 2, sd>(tria_accessor);
          else if (structdim == 2)
            return get_manifold_id<2, 2, sd>(tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 3) && (sd >= 3))
        {
          Assert(structdim <= 3,
                 ExcMessage("structdim must be smaller than dim (3)"));
          if (structdim == 1)
            return get_manifold_id<1, 3, sd>(tria_accessor);
          else if (structdim == 2)
            return get_manifold_id<2, 3, sd>(tria_accessor);
          else if (structdim == 3)
            return get_manifold_id<3, 3, sd>(tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
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



    template <int d, int sd>
    bool
    at_boundary_wrapper(const void *tria_accessor,
                        const int   structdim,
                        const int   dim,
                        const int   spacedim)
    {
      if constexpr (sd < d)
        AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));

      if constexpr (d > 1)
        if (dim != d)
          return at_boundary_wrapper<d - 1, sd>(tria_accessor,
                                                structdim,
                                                dim,
                                                spacedim);
      if (dim != d)
        AssertThrow(false, ExcMessage("Unsupported dimension."));

      if constexpr (sd > 1)
        if (spacedim != sd)
          return at_boundary_wrapper<d, sd - 1>(tria_accessor,
                                                structdim,
                                                dim,
                                                spacedim);
      if (spacedim != sd)
        AssertThrow(false, ExcMessage("Unsupported space dimension"));

      if constexpr ((d == 1) && (sd >= 1))
        {
          Assert(structdim <= 1,
                 ExcMessage("structdim must be smaller than dim (1)"));
          if (structdim == 1)
            return at_boundary<1, 1, sd>(tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 2) && (sd >= 2))
        {
          Assert(structdim <= 2,
                 ExcMessage("structdim must be smaller than dim (2)"));
          if (structdim == 1)
            return at_boundary<1, 2, sd>(tria_accessor);
          else if (structdim == 2)
            return at_boundary<2, 2, sd>(tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 3) && (sd >= 3))
        {
          Assert(structdim <= 3,
                 ExcMessage("structdim must be smaller than dim (3)"));
          if (structdim == 1)
            return at_boundary<1, 3, sd>(tria_accessor);
          else if (structdim == 2)
            return at_boundary<2, 3, sd>(tria_accessor);
          else if (structdim == 3)
            return at_boundary<3, 3, sd>(tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
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



    template <int d, int sd>
    double
    measure_wrapper(const void *tria_accessor,
                    const int   structdim,
                    const int   dim,
                    const int   spacedim)
    {
      if constexpr (sd < d)
        AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));

      if constexpr (d > 1)
        if (dim != d)
          return measure_wrapper<d - 1, sd>(tria_accessor,
                                            structdim,
                                            dim,
                                            spacedim);
      if (dim != d)
        AssertThrow(false, ExcMessage("Unsupported dimension."));

      if constexpr (sd > 1)
        if (spacedim != sd)
          return measure_wrapper<d, sd - 1>(tria_accessor,
                                            structdim,
                                            dim,
                                            spacedim);
      if (spacedim != sd)
        AssertThrow(false, ExcMessage("Unsupported space dimension"));

      if constexpr ((d == 1) && (sd >= 1))
        {
          Assert(structdim <= 1,
                 ExcMessage("structdim must be smaller than dim (1)"));
          if (structdim == 1)
            return measure<1, 1, sd>(tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 2) && (sd >= 2))
        {
          Assert(structdim <= 2,
                 ExcMessage("structdim must be smaller than dim (2)"));
          if (structdim == 1)
            return measure<1, 2, sd>(tria_accessor);
          else if (structdim == 2)
            return measure<2, 2, sd>(tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 3) && (sd >= 3))
        {
          Assert(structdim <= 3,
                 ExcMessage("structdim must be smaller than dim (3)"));
          if (structdim == 1)
            return measure<1, 3, sd>(tria_accessor);
          else if (structdim == 2)
            return measure<2, 3, sd>(tria_accessor);
          else if (structdim == 3)
            return measure<3, 3, sd>(tria_accessor);
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
    }



    template <int d, int sd>
    void *
    copy_tria_accessor(void     *tria_accessor,
                       const int structdim,
                       const int dim,
                       const int spacedim)
    {
      if constexpr (sd < d)
        AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));

      if constexpr (d > 1)
        if (dim != d)
          return copy_tria_accessor<d - 1, sd>(tria_accessor,
                                               structdim,
                                               dim,
                                               spacedim);
      if (dim != d)
        AssertThrow(false, ExcMessage("Unsupported dimension."));

      if constexpr (sd > 1)
        if (spacedim != sd)
          return copy_tria_accessor<d, sd - 1>(tria_accessor,
                                               structdim,
                                               dim,
                                               spacedim);
      if (spacedim != sd)
        AssertThrow(false, ExcMessage("Unsupported space dimension"));

      if constexpr ((d == 1) && (sd >= 1))
        {
          Assert(structdim <= 1,
                 ExcMessage("structdim must be smaller than dim (1)"));
          if (structdim == 0)
            {
              TriaAccessor<0, 1, sd> *other_accessor =
                static_cast<TriaAccessor<0, 1, sd> *>(tria_accessor);
              return new TriaAccessor<0, 1, sd>(*other_accessor);
            }
          else if (structdim == 1)
            {
              TriaAccessor<1, 1, sd> *other_accessor =
                static_cast<TriaAccessor<1, 1, sd> *>(tria_accessor);
              return new TriaAccessor<1, 1, sd>(*other_accessor);
            }
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 2) && (sd >= 2))
        {
          Assert(structdim <= 2,
                 ExcMessage("structdim must be smaller than dim (2)"));
          if (structdim == 0)
            {
              TriaAccessor<0, 2, sd> *other_accessor =
                static_cast<TriaAccessor<0, 2, sd> *>(tria_accessor);
              return new TriaAccessor<0, 2, sd>(*other_accessor);
            }
          else if (structdim == 1)
            {
              TriaAccessor<1, 2, sd> *other_accessor =
                static_cast<TriaAccessor<1, 2, sd> *>(tria_accessor);
              return new TriaAccessor<1, 2, sd>(*other_accessor);
            }
          else if (structdim == 2)
            {
              TriaAccessor<2, 2, sd> *other_accessor =
                static_cast<TriaAccessor<2, 2, sd> *>(tria_accessor);
              return new TriaAccessor<2, 2, sd>(*other_accessor);
            }
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 3) && (sd >= 3))
        {
          Assert(structdim <= 3,
                 ExcMessage("structdim must be smaller than dim (3)"));
          if (structdim == 0)
            {
              TriaAccessor<0, 3, sd> *other_accessor =
                static_cast<TriaAccessor<0, 3, sd> *>(tria_accessor);
              return new TriaAccessor<0, 3, sd>(*other_accessor);
            }
          else if (structdim == 1)
            {
              TriaAccessor<1, 3, sd> *other_accessor =
                static_cast<TriaAccessor<1, 3, sd> *>(tria_accessor);
              return new TriaAccessor<1, 3, sd>(*other_accessor);
            }
          else if (structdim == 2)
            {
              TriaAccessor<2, 3, sd> *other_accessor =
                static_cast<TriaAccessor<2, 3, sd> *>(tria_accessor);
              return new TriaAccessor<2, 3, sd>(*other_accessor);
            }
          else if (structdim == 3)
            {
              TriaAccessor<3, 3, sd> *other_accessor =
                static_cast<TriaAccessor<3, 3, sd> *>(tria_accessor);
              return new TriaAccessor<3, 3, sd>(*other_accessor);
            }
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
    }



    template <int d, int sd>
    void
    delete_tria_accessor_pointer(void     *tria_accessor,
                                 const int structdim,
                                 const int dim,
                                 const int spacedim)
    {
      if constexpr (sd < d)
        AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));

      if constexpr (d > 1)
        if (dim != d)
          return delete_tria_accessor_pointer<d - 1, sd>(tria_accessor,
                                                         structdim,
                                                         dim,
                                                         spacedim);
      if (dim != d)
        AssertThrow(false, ExcMessage("Unsupported dimension."));

      if constexpr (sd > 1)
        if (spacedim != sd)
          return delete_tria_accessor_pointer<d, sd - 1>(tria_accessor,
                                                         structdim,
                                                         dim,
                                                         spacedim);
      if (spacedim != sd)
        AssertThrow(false, ExcMessage("Unsupported space dimension"));

      if constexpr ((d == 1) && (sd >= 1))
        {
          Assert(structdim <= 1,
                 ExcMessage("structdim must be smaller than dim (1)"));
          if (structdim == 0)
            {
              TriaAccessor<0, 1, sd> *tmp =
                static_cast<TriaAccessor<0, 1, sd> *>(tria_accessor);
              delete tmp;
            }
          else if (structdim == 1)
            {
              TriaAccessor<0, 1, sd> *tmp =
                static_cast<TriaAccessor<0, 1, sd> *>(tria_accessor);
              delete tmp;
            }
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 2) && (sd >= 2))
        {
          Assert(structdim <= 2,
                 ExcMessage("structdim must be smaller than dim (2)"));
          if (structdim == 0)
            {
              TriaAccessor<0, 2, sd> *tmp =
                static_cast<TriaAccessor<0, 2, sd> *>(tria_accessor);
              delete tmp;
            }
          else if (structdim == 1)
            {
              TriaAccessor<1, 2, sd> *tmp =
                static_cast<TriaAccessor<1, 2, sd> *>(tria_accessor);
              delete tmp;
            }
          else if (structdim == 2)
            {
              TriaAccessor<2, 2, sd> *tmp =
                static_cast<TriaAccessor<2, 2, sd> *>(tria_accessor);
              delete tmp;
            }
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
      else if constexpr ((d == 3) && (sd >= 3))
        {
          Assert(structdim <= 3,
                 ExcMessage("structdim must be smaller than dim (3)"));
          if (structdim == 0)
            {
              TriaAccessor<0, 3, sd> *tmp =
                static_cast<TriaAccessor<0, 3, sd> *>(tria_accessor);
              delete tmp;
            }
          else if (structdim == 1)
            {
              TriaAccessor<1, 3, sd> *tmp =
                static_cast<TriaAccessor<1, 3, sd> *>(tria_accessor);
              delete tmp;
            }
          else if (structdim == 2)
            {
              TriaAccessor<2, 3, sd> *tmp =
                static_cast<TriaAccessor<2, 3, sd> *>(tria_accessor);
              delete tmp;
            }
          else if (structdim == 3)
            {
              TriaAccessor<3, 3, sd> *tmp =
                static_cast<TriaAccessor<3, 3, sd> *>(tria_accessor);
              delete tmp;
            }
          else
            AssertThrow(false, ExcMessage("Invalid structdim"));
        }
    }
  } // namespace internal



  TriaAccessorWrapper::TriaAccessorWrapper(const TriaAccessorWrapper &other)
    : structdim(other.structdim)
    , dim(other.dim)
    , spacedim(other.spacedim)
    , tria_accessor(internal::copy_tria_accessor<3, 3>(other.tria_accessor,
                                                       structdim,
                                                       dim,
                                                       spacedim))
  {}

  TriaAccessorWrapper::TriaAccessorWrapper(void     *tria_accessor,
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
        internal::delete_tria_accessor_pointer<3, 3>(tria_accessor,
                                                     structdim,
                                                     dim,
                                                     spacedim);

        dim           = -1;
        spacedim      = -1;
        structdim     = -1;
        tria_accessor = nullptr;
      }
  }



  PointWrapper
  TriaAccessorWrapper::get_barycenter() const
  {
    return internal::get_barycenter_wrapper<3, 3>(tria_accessor,
                                                  structdim,
                                                  dim,
                                                  spacedim);
  }



  PointWrapper
  TriaAccessorWrapper::get_center(const bool respect_manifold,
                                  const bool interpolate_from_surrounding) const
  {
    return internal::get_center_wrapper<3, 3>(respect_manifold,
                                              interpolate_from_surrounding,
                                              tria_accessor,
                                              structdim,
                                              dim,
                                              spacedim);
  }



  void
  TriaAccessorWrapper::set_boundary_id(const int boundary_id)
  {
    internal::set_boundary_id_wrapper<3, 3>(
      boundary_id, tria_accessor, structdim, dim, spacedim);
  }



  void
  TriaAccessorWrapper::set_all_boundary_ids(const int boundary_id)
  {
    internal::set_all_boundary_ids_wrapper<3, 3>(
      boundary_id, tria_accessor, structdim, dim, spacedim);
  }



  int
  TriaAccessorWrapper::get_boundary_id() const
  {
    return internal::get_boundary_id_wrapper<3, 3>(tria_accessor,
                                                   structdim,
                                                   dim,
                                                   spacedim);
  }



  void
  TriaAccessorWrapper::set_vertex(const int i, PointWrapper &point_wrapper)
  {
    AssertThrow(i < static_cast<int>(Utilities::pow(2, dim)),
                ExcVertexDoesNotExist(i, Utilities::pow(2, dim)));

    internal::set_vertex_wrapper<3, 3>(
      i, point_wrapper, tria_accessor, structdim, dim, spacedim);
  }



  PointWrapper
  TriaAccessorWrapper::get_vertex(const int i) const
  {
    AssertThrow(i < static_cast<int>(Utilities::pow(2, dim)),
                ExcVertexDoesNotExist(i, Utilities::pow(2, dim)));

    return internal::get_vertex_wrapper<3, 3>(
      i, tria_accessor, structdim, dim, spacedim);
  }



  void
  TriaAccessorWrapper::set_manifold_id(const int manifold_id)
  {
    internal::set_manifold_id_wrapper<3, 3>(
      manifold_id, tria_accessor, structdim, dim, spacedim);
  }



  int
  TriaAccessorWrapper::get_manifold_id() const
  {
    return internal::get_manifold_id_wrapper<3, 3>(tria_accessor,
                                                   structdim,
                                                   dim,
                                                   spacedim);
  }



  bool
  TriaAccessorWrapper::at_boundary() const
  {
    return internal::at_boundary_wrapper<3, 3>(tria_accessor,
                                               structdim,
                                               dim,
                                               spacedim);
  }



  double
  TriaAccessorWrapper::measure() const
  {
    return internal::measure_wrapper<3, 3>(tria_accessor,
                                           structdim,
                                           dim,
                                           spacedim);
  }

} // namespace python

DEAL_II_NAMESPACE_CLOSE
