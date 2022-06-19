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


#include <dealii-python-bindings/cell_accessor_wrapper.h>
#include <dealii-python-bindings/point_wrapper.h>
#include <dealii-python-bindings/tria_accessor_wrapper.h>
#include <dealii-python-bindings/triangulation_wrapper.h>

DEAL_II_NAMESPACE_OPEN

namespace python
{
  namespace internal
  {
    template <int dim, int spacedim>
    void
    set_refine_flag(const std::string &refinement_case, void *cell_accessor)
    {
      CellAccessor<dim, spacedim> *cell =
        static_cast<CellAccessor<dim, spacedim> *>(cell_accessor);

      std::unique_ptr<RefinementCase<dim>> ref_case;
      bool                                 assigned = false;

      if (refinement_case.compare("isotropic") == 0)
        {
          ref_case.reset(new RefinementCase<dim>(
            RefinementPossibilities<dim>::Possibilities::isotropic_refinement));
          assigned = true;
        }
      else if (refinement_case.compare("no_refinement") == 0)
        {
          ref_case.reset(new RefinementCase<dim>(
            RefinementPossibilities<dim>::Possibilities::no_refinement));
          assigned = true;
        }
      else if (refinement_case.compare("cut_x") == 0)
        {
          ref_case.reset(new RefinementCase<dim>(
            RefinementPossibilities<dim>::Possibilities::cut_x));
          assigned = true;
        }

      if constexpr (dim > 1)
        if (!assigned)
          {
            if (refinement_case.compare("cut_y") == 0)
              {
                ref_case.reset(new RefinementCase<dim>(
                  RefinementPossibilities<dim>::Possibilities::cut_y));
                assigned = true;
              }
            else if (refinement_case.compare("cut_xy") == 0)
              {
                ref_case.reset(new RefinementCase<dim>(
                  RefinementPossibilities<dim>::Possibilities::cut_xy));
                assigned = true;
              }
          }

      if constexpr (dim == 3)
        if (!assigned)
          {
            if (refinement_case.compare("cut_z") == 0)
              {
                ref_case.reset(new RefinementCase<3>(
                  RefinementPossibilities<3>::Possibilities::cut_z));
                assigned = true;
              }
            else if (refinement_case.compare("cut_xz") == 0)
              {
                ref_case.reset(new RefinementCase<3>(
                  RefinementPossibilities<3>::Possibilities::cut_xz));
                assigned = true;
              }
            else if (refinement_case.compare("cut_yz") == 0)
              {
                ref_case.reset(new RefinementCase<3>(
                  RefinementPossibilities<3>::Possibilities::cut_yz));
                assigned = true;
              }
            else if (refinement_case.compare("cut_xyz") == 0)
              {
                ref_case.reset(new RefinementCase<3>(
                  RefinementPossibilities<3>::Possibilities::cut_xyz));
                assigned = true;
              }
          }

      if (!assigned)
        AssertThrow(false, ExcMessage("Unknown refinement possibility."));

      cell->set_refine_flag(*ref_case);
    }


    template <int d, int sd>
    void
    set_refine_flag_wrapper(const std::string &refinement_case,
                            void              *cell_accessor,
                            const int          dim,
                            const int          spacedim)
    {
      if (dim == d && spacedim == sd)
        return set_refine_flag<d, sd>(refinement_case, cell_accessor);

      if constexpr (d == 1)
        {
          if constexpr (sd == 1)
            {
              AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
            }
          else
            return set_refine_flag_wrapper<sd - 1, sd - 1>(refinement_case,
                                                           cell_accessor,
                                                           dim,
                                                           spacedim);
        }
      else
        {
          return set_refine_flag_wrapper<d - 1, sd>(refinement_case,
                                                    cell_accessor,
                                                    dim,
                                                    spacedim);
        }
    }



    template <int dim, int spacedim>
    std::string
    get_refine_flag(const void *cell_accessor)
    {
      const CellAccessor<dim, spacedim> *cell =
        static_cast<const CellAccessor<dim, spacedim> *>(cell_accessor);

      std::string         refine_flag;
      RefinementCase<dim> ref_case = cell->refine_flag_set();
      switch (static_cast<int>(ref_case))
        {
            case (0): {
              refine_flag = "no_refinement";
              break;
            }
            case (1): {
              refine_flag = "cut_x";
              break;
            }
            case (2): {
              refine_flag = "cut_y";
              break;
            }
            case (3): {
              refine_flag = "cut_xy";
              break;
            }
            case (4): {
              refine_flag = "cut_z";
              break;
            }
            case (5): {
              refine_flag = "cut_xz";
              break;
            }
            case (6): {
              refine_flag = "cut_yz";
              break;
            }
            case (7): {
              refine_flag = "cut_xyz";
              break;
            }
            default: {
              AssertThrow(false, ExcMessage("Internal error."));
            }
        }

      return refine_flag;
    }



    template <int d, int sd>
    std::string
    get_refine_flag_wrapper(const void *cell_accessor,
                            const int   dim,
                            const int   spacedim)
    {
      if (dim == d && spacedim == sd)
        return get_refine_flag<d, sd>(cell_accessor);

      if constexpr (d == 1)
        {
          if constexpr (sd == 1)
            {
              AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
            }
          else
            return get_refine_flag_wrapper<sd - 1, sd - 1>(cell_accessor,
                                                           dim,
                                                           spacedim);
        }
      else
        {
          return get_refine_flag_wrapper<d - 1, sd>(cell_accessor,
                                                    dim,
                                                    spacedim);
        }
    }



    template <int dim, int spacedim>
    void
    set_coarsen_flag(const bool coarsen_flag, void *cell_accessor)
    {
      CellAccessor<dim, spacedim> *cell =
        static_cast<CellAccessor<dim, spacedim> *>(cell_accessor);
      if (coarsen_flag)
        cell->set_coarsen_flag();
      else
        cell->clear_coarsen_flag();
    }



    template <int d, int sd>
    void
    set_coarsen_flag_wrapper(const bool coarsen_flag,
                             void      *cell_accessor,
                             const int  dim,
                             const int  spacedim)
    {
      if (dim == d && spacedim == sd)
        return set_coarsen_flag<d, sd>(coarsen_flag, cell_accessor);

      if constexpr (d == 1)
        {
          if constexpr (sd == 1)
            {
              AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
            }
          else
            return set_coarsen_flag_wrapper<sd - 1, sd - 1>(coarsen_flag,
                                                            cell_accessor,
                                                            dim,
                                                            spacedim);
        }
      else
        {
          return set_coarsen_flag_wrapper<d - 1, sd>(coarsen_flag,
                                                     cell_accessor,
                                                     dim,
                                                     spacedim);
        }
    }



    template <int dim, int spacedim>
    bool
    get_coarsen_flag(const void *cell_accessor)
    {
      const CellAccessor<dim, spacedim> *cell =
        static_cast<const CellAccessor<dim, spacedim> *>(cell_accessor);

      return cell->coarsen_flag_set();
    }



    template <int d, int sd>
    bool
    get_coarsen_flag_wrapper(const void *cell_accessor,
                             const int   dim,
                             const int   spacedim)
    {
      if (dim == d && spacedim == sd)
        return get_coarsen_flag<d, sd>(cell_accessor);

      if constexpr (d == 1)
        {
          if constexpr (sd == 1)
            {
              AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
            }
          else
            return get_coarsen_flag_wrapper<sd - 1, sd - 1>(cell_accessor,
                                                            dim,
                                                            spacedim);
        }
      else
        {
          return get_coarsen_flag_wrapper<d - 1, sd>(cell_accessor,
                                                     dim,
                                                     spacedim);
        }
    }



    template <int dim, int spacedim>
    PointWrapper
    get_barycenter(const void *cell_accessor)
    {
      const CellAccessor<dim, spacedim> *cell =
        static_cast<const CellAccessor<dim, spacedim> *>(cell_accessor);
      Point<spacedim> barycenter = cell->barycenter();
      PYSPACE::list   barycenter_list;
      for (int i = 0; i < spacedim; ++i)
        barycenter_list.append(barycenter[i]);

      return PointWrapper(barycenter_list);
    }



    template <int d, int sd>
    PointWrapper
    get_barycenter_wrapper(const void *cell_accessor,
                           const int   dim,
                           const int   spacedim)
    {
      if (dim == d && spacedim == sd)
        return get_barycenter<d, sd>(cell_accessor);

      if constexpr (d == 1)
        {
          if constexpr (sd == 1)
            {
              AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
            }
          else
            return get_barycenter_wrapper<sd - 1, sd - 1>(cell_accessor,
                                                          dim,
                                                          spacedim);
        }
      else
        {
          return get_barycenter_wrapper<d - 1, sd>(cell_accessor,
                                                   dim,
                                                   spacedim);
        }
    }



    template <int dim, int spacedim>
    PointWrapper
    get_center(const bool  respect_manifold,
               const bool  interpolate_from_surrounding,
               const void *cell_accessor)
    {
      const CellAccessor<dim, spacedim> *cell =
        static_cast<const CellAccessor<dim, spacedim> *>(cell_accessor);
      Point<spacedim> center =
        cell->center(respect_manifold, interpolate_from_surrounding);
      PYSPACE::list center_list;
      for (int i = 0; i < spacedim; ++i)
        center_list.append(center[i]);

      return PointWrapper(center_list);
    }


    template <int d, int sd>
    PointWrapper
    get_center_wrapper(const bool  respect_manifold,
                       const bool  interpolate_from_surrounding,
                       const void *cell_accessor,
                       const int   dim,
                       const int   spacedim)
    {
      if (dim == d && spacedim == sd)
        return get_center<d, sd>(respect_manifold,
                                 interpolate_from_surrounding,
                                 cell_accessor);

      if constexpr (d == 1)
        {
          if constexpr (sd == 1)
            {
              AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
            }
          else
            return get_center_wrapper<sd - 1, sd - 1>(
              respect_manifold,
              interpolate_from_surrounding,
              cell_accessor,
              dim,
              spacedim);
        }
      else
        {
          return get_center_wrapper<d - 1, sd>(respect_manifold,
                                               interpolate_from_surrounding,
                                               cell_accessor,
                                               dim,
                                               spacedim);
        }
    }



    template <int dim, int spacedim>
    void
    set_material_id(const int material_id, void *cell_accessor)
    {
      CellAccessor<dim, spacedim> *cell =
        static_cast<CellAccessor<dim, spacedim> *>(cell_accessor);
      cell->set_material_id(material_id);
    }



    template <int d, int sd>
    void
    set_material_id_wrapper(const int material_id,
                            void     *cell_accessor,
                            const int dim,
                            const int spacedim)
    {
      if (dim == d && spacedim == sd)
        return set_material_id<d, sd>(material_id, cell_accessor);

      if constexpr (d == 1)
        {
          if constexpr (sd == 1)
            {
              AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
            }
          else
            return set_material_id_wrapper<sd - 1, sd - 1>(material_id,
                                                           cell_accessor,
                                                           dim,
                                                           spacedim);
        }
      else
        {
          return set_material_id_wrapper<d - 1, sd>(material_id,
                                                    cell_accessor,
                                                    dim,
                                                    spacedim);
        }
    }



    template <int dim, int spacedim>
    int
    get_material_id(const void *cell_accessor)
    {
      const CellAccessor<dim, spacedim> *cell =
        static_cast<const CellAccessor<dim, spacedim> *>(cell_accessor);

      return cell->material_id();
    }



    template <int d, int sd>
    int
    get_material_id_wrapper(const void *cell_accessor,
                            const int   dim,
                            const int   spacedim)
    {
      if (dim == d && spacedim == sd)
        return get_material_id<d, sd>(cell_accessor);

      if constexpr (d == 1)
        {
          if constexpr (sd == 1)
            {
              AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
            }
          else
            return get_material_id_wrapper<sd - 1, sd - 1>(cell_accessor,
                                                           dim,
                                                           spacedim);
        }
      else
        {
          return get_material_id_wrapper<d - 1, sd>(cell_accessor,
                                                    dim,
                                                    spacedim);
        }
    }



    template <int dim, int spacedim>
    void
    set_vertex(const int i, PointWrapper &point_wrapper, void *cell_accessor)
    {
      CellAccessor<dim, spacedim> *cell =
        static_cast<CellAccessor<dim, spacedim> *>(cell_accessor);
      Point<spacedim> *point =
        static_cast<Point<spacedim> *>(point_wrapper.get_point());

      cell->vertex(i) = *point;
    }



    template <int d, int sd>
    void
    set_vertex_wrapper(const int     i,
                       PointWrapper &point_wrapper,
                       void         *cell_accessor,
                       const int     dim,
                       const int     spacedim)
    {
      if (dim == d && spacedim == sd)
        return set_vertex<d, sd>(i, point_wrapper, cell_accessor);

      if constexpr (d == 1)
        {
          if constexpr (sd == 1)
            {
              AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
            }
          else
            return set_vertex_wrapper<sd - 1, sd - 1>(
              i, point_wrapper, cell_accessor, dim, spacedim);
        }
      else
        {
          return set_vertex_wrapper<d - 1, sd>(
            i, point_wrapper, cell_accessor, dim, spacedim);
        }
    }



    template <int dim, int spacedim>
    PointWrapper
    get_vertex(const int i, const void *cell_accessor)
    {
      const CellAccessor<dim, spacedim> *cell =
        static_cast<const CellAccessor<dim, spacedim> *>(cell_accessor);
      Point<spacedim> vertex = cell->vertex(i);

      PYSPACE::list coordinates;
      for (int j = 0; j < spacedim; ++j)
        coordinates.append(vertex[j]);

      return PointWrapper(coordinates);
    }



    template <int d, int sd>
    PointWrapper
    get_vertex_wrapper(const int   i,
                       const void *cell_accessor,
                       const int   dim,
                       const int   spacedim)
    {
      if (dim == d && spacedim == sd)
        return get_vertex<d, sd>(i, cell_accessor);

      if constexpr (d == 1)
        {
          if constexpr (sd == 1)
            {
              AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
            }
          else
            return get_vertex_wrapper<sd - 1, sd - 1>(i,
                                                      cell_accessor,
                                                      dim,
                                                      spacedim);
        }
      else
        {
          return get_vertex_wrapper<d - 1, sd>(i, cell_accessor, dim, spacedim);
        }
    }



    template <int dim, int spacedim>
    void
    set_manifold_id(const int manifold_id, void *cell_accessor)
    {
      CellAccessor<dim, spacedim> *cell =
        static_cast<CellAccessor<dim, spacedim> *>(cell_accessor);
      cell->set_manifold_id(manifold_id);
    }



    template <int d, int sd>
    void
    set_manifold_id_wrapper(const int manifold_id,
                            void     *cell_accessor,
                            const int dim,
                            const int spacedim)
    {
      if (dim == d && spacedim == sd)
        return set_manifold_id<d, sd>(manifold_id, cell_accessor);

      if constexpr (d == 1)
        {
          if constexpr (sd == 1)
            {
              AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
            }
          else
            return set_manifold_id_wrapper<sd - 1, sd - 1>(manifold_id,
                                                           cell_accessor,
                                                           dim,
                                                           spacedim);
        }
      else
        {
          return set_manifold_id_wrapper<d - 1, sd>(manifold_id,
                                                    cell_accessor,
                                                    dim,
                                                    spacedim);
        }
    }



    template <int dim, int spacedim>
    int
    get_manifold_id(const void *cell_accessor)
    {
      const CellAccessor<dim, spacedim> *cell =
        static_cast<const CellAccessor<dim, spacedim> *>(cell_accessor);
      return cell->manifold_id();
    }



    template <int d, int sd>
    int
    get_manifold_id_wrapper(const void *cell_accessor,
                            const int   dim,
                            const int   spacedim)
    {
      if (dim == d && spacedim == sd)
        return get_manifold_id<d, sd>(cell_accessor);

      if constexpr (d == 1)
        {
          if constexpr (sd == 1)
            {
              AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
            }
          else
            return get_manifold_id_wrapper<sd - 1, sd - 1>(cell_accessor,
                                                           dim,
                                                           spacedim);
        }
      else
        {
          return get_manifold_id_wrapper<d - 1, sd>(cell_accessor,
                                                    dim,
                                                    spacedim);
        }
    }



    template <int dim, int spacedim>
    void
    set_all_manifold_ids(const int manifold_id, void *cell_accessor)
    {
      const CellAccessor<dim, spacedim> *cell =
        static_cast<const CellAccessor<dim, spacedim> *>(cell_accessor);
      return cell->set_all_manifold_ids(manifold_id);
    }



    template <int d, int sd>
    void
    set_all_manifold_ids_wrapper(const int manifold_id,
                                 void     *cell_accessor,
                                 const int dim,
                                 const int spacedim)
    {
      if (dim == d && spacedim == sd)
        return set_all_manifold_ids<d, sd>(manifold_id, cell_accessor);

      if constexpr (d == 1)
        {
          if constexpr (sd == 1)
            {
              AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
            }
          else
            return set_all_manifold_ids_wrapper<sd - 1, sd - 1>(manifold_id,
                                                                cell_accessor,
                                                                dim,
                                                                spacedim);
        }
      else
        {
          return set_all_manifold_ids_wrapper<d - 1, sd>(manifold_id,
                                                         cell_accessor,
                                                         dim,
                                                         spacedim);
        }
    }



    template <int dim, int spacedim>
    const CellAccessor<dim, spacedim> *
    neighbor(const int i, const void *cell_accessor)
    {
      const CellAccessor<dim, spacedim> *cell =
        static_cast<const CellAccessor<dim, spacedim> *>(cell_accessor);
      return cell->neighbor(i);
    }



    template <int dim, int spacedim>
    PYSPACE::list
    faces(const void *cell_accessor)
    {
      const CellAccessor<dim, spacedim> *cell =
        static_cast<const CellAccessor<dim, spacedim> *>(cell_accessor);

      PYSPACE::list faces_list;

      auto face_iterators = cell->face_iterators();
      for (auto &it : face_iterators)
        {
          TriaAccessor<dim - 1, dim, spacedim> *face_accessor =
            new TriaAccessor<dim - 1, dim, spacedim>(*it);
          faces_list.append(
            TriaAccessorWrapper(face_accessor, dim - 1, dim, spacedim));
        }

      return faces_list;
    }



    template <int d, int sd>
    PYSPACE::list
    faces_wrapper(const void *cell_accessor, const int dim, const int spacedim)
    {
      if (dim == d && spacedim == sd)
        return faces<d, sd>(cell_accessor);

      if constexpr (d == 1)
        {
          if constexpr (sd == 1)
            {
              AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
            }
          else
            return faces_wrapper<sd - 1, sd - 1>(cell_accessor, dim, spacedim);
        }
      else
        {
          return faces_wrapper<d - 1, sd>(cell_accessor, dim, spacedim);
        }
    }



    template <int dim, int spacedim>
    const CellAccessor<dim, spacedim> *
    cell_cast(const void *cell_accessor)
    {
      return static_cast<const CellAccessor<dim, spacedim> *>(cell_accessor);
    }
  } // namespace internal



  template <int d, int sd>
  void *
  CellAccessorWrapper::cell_accessor_factory1(const CellAccessorWrapper &other,
                                              const int                  dim,
                                              const int spacedim)
  {
    if (d == dim && sd == spacedim)
      {
        CellAccessor<d, sd> *other_cell =
          static_cast<CellAccessor<d, sd> *>(other.cell_accessor);
        return new CellAccessor<d, sd>(*other_cell);
      }

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return cell_accessor_factory1<sd - 1, sd - 1>(other, dim, spacedim);
      }
    else
      {
        return cell_accessor_factory1<d - 1, sd>(other, dim, spacedim);
      }
  }



  template <int d, int sd>
  void *
  CellAccessorWrapper::cell_accessor_factory2(
    TriangulationWrapper &triangulation_wrapper,
    const int             level,
    const int             index,
    const int             dim,
    const int             spacedim)
  {
    if (d == dim && sd == spacedim)
      {
        Triangulation<d, sd> *tmp = static_cast<Triangulation<d, sd> *>(
          triangulation_wrapper.get_triangulation());
        return new CellAccessor<d, sd>(tmp, level, index);
      }

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return cell_accessor_factory2<sd - 1, sd - 1>(
            triangulation_wrapper, level, index, dim, spacedim);
      }
    else
      {
        return cell_accessor_factory2<d - 1, sd>(
          triangulation_wrapper, level, index, dim, spacedim);
      }
  }



  template <int d, int sd>
  void
  CellAccessorWrapper::cell_accessor_destroyer(void     *cell_accessor,
                                               const int dim,
                                               const int spacedim)
  {
    if (d == dim && sd == spacedim)
      {
        CellAccessor<d, sd> *tmp =
          static_cast<CellAccessor<d, sd> *>(cell_accessor);
        delete tmp;
        return;
      }

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return cell_accessor_destroyer<sd - 1, sd - 1>(cell_accessor,
                                                         dim,
                                                         spacedim);
      }
    else
      {
        return cell_accessor_destroyer<d - 1, sd>(cell_accessor, dim, spacedim);
      }
  }



  CellAccessorWrapper::CellAccessorWrapper(const CellAccessorWrapper &other)
    : dim(other.dim)
    , spacedim(other.spacedim)
    , cell_accessor(cell_accessor_factory1<3, 3>(other, dim, spacedim))
  {}



  CellAccessorWrapper::CellAccessorWrapper()
    : dim(-1)
    , spacedim(-1)
    , cell_accessor(nullptr)
  {}



  CellAccessorWrapper::CellAccessorWrapper(
    TriangulationWrapper &triangulation_wrapper,
    const int             level,
    const int             index)
    : dim(triangulation_wrapper.get_dim())
    , spacedim(triangulation_wrapper.get_spacedim())
    , cell_accessor(cell_accessor_factory2<3, 3>(triangulation_wrapper,
                                                 level,
                                                 index,
                                                 dim,
                                                 spacedim))
  {}



  CellAccessorWrapper::~CellAccessorWrapper()
  {
    if (dim != -1)
      {
        cell_accessor_destroyer<3, 3>(cell_accessor, dim, spacedim);
        dim           = -1;
        spacedim      = -1;
        cell_accessor = nullptr;
      }
  }



  void
  CellAccessorWrapper::set_refine_flag(const std::string &refinement_case)
  {
    internal::set_refine_flag_wrapper<3, 3>(refinement_case,
                                            cell_accessor,
                                            dim,
                                            spacedim);
  }



  std::string
  CellAccessorWrapper::get_refine_flag() const
  {
    return internal::get_refine_flag_wrapper<3, 3>(cell_accessor,
                                                   dim,
                                                   spacedim);
  }



  void
  CellAccessorWrapper::set_coarsen_flag(const bool coarsen_flag)
  {
    internal::set_coarsen_flag_wrapper<3, 3>(coarsen_flag,
                                             cell_accessor,
                                             dim,
                                             spacedim);
  }



  bool
  CellAccessorWrapper::get_coarsen_flag() const
  {
    return internal::get_coarsen_flag_wrapper<3, 3>(cell_accessor,
                                                    dim,
                                                    spacedim);
  }



  PointWrapper
  CellAccessorWrapper::get_barycenter() const
  {
    return internal::get_barycenter_wrapper<3, 3>(cell_accessor, dim, spacedim);
  }



  PointWrapper
  CellAccessorWrapper::get_center(const bool respect_manifold,
                                  const bool interpolate_from_surrounding) const
  {
    return internal::get_center_wrapper<3, 3>(respect_manifold,
                                              interpolate_from_surrounding,
                                              cell_accessor,
                                              dim,
                                              spacedim);
  }



  void
  CellAccessorWrapper::set_material_id(const int material_id)
  {
    AssertThrow(static_cast<types::material_id>(material_id) <
                  numbers::invalid_material_id,
                ExcMessage("material_id is too large."));
    return internal::set_material_id_wrapper<3, 3>(material_id,
                                                   cell_accessor,
                                                   dim,
                                                   spacedim);
  }



  int
  CellAccessorWrapper::get_material_id() const
  {
    return internal::get_material_id_wrapper<3, 3>(cell_accessor,
                                                   dim,
                                                   spacedim);
  }



  void
  CellAccessorWrapper::set_vertex(const int i, PointWrapper &point_wrapper)
  {
    AssertThrow(i < static_cast<int>(Utilities::pow(2, dim)),
                ExcVertexDoesNotExist(i, Utilities::pow(2, dim)));
    internal::set_vertex_wrapper<3, 3>(
      i, point_wrapper, cell_accessor, dim, spacedim);
  }



  PointWrapper
  CellAccessorWrapper::get_vertex(const int i) const
  {
    AssertThrow(i < static_cast<int>(Utilities::pow(2, dim)),
                ExcVertexDoesNotExist(i, Utilities::pow(2, dim)));

    return internal::get_vertex_wrapper<3, 3>(i, cell_accessor, dim, spacedim);
  }



  void
  CellAccessorWrapper::set_manifold_id(const int manifold_id)
  {
    internal::set_manifold_id_wrapper<3, 3>(manifold_id,
                                            cell_accessor,
                                            dim,
                                            spacedim);
  }



  int
  CellAccessorWrapper::get_manifold_id() const
  {
    return internal::get_manifold_id_wrapper<3, 3>(cell_accessor,
                                                   dim,
                                                   spacedim);
  }



  void
  CellAccessorWrapper::set_all_manifold_ids(const int manifold_id)
  {
    internal::set_all_manifold_ids_wrapper<3, 3>(manifold_id,
                                                 cell_accessor,
                                                 dim,
                                                 spacedim);
  }



  bool
  CellAccessorWrapper::at_boundary() const
  {
    if ((dim == 1) && (spacedim == 1))
      return internal::cell_cast<1, 1>(cell_accessor)->at_boundary();
    else if ((dim == 1) && (spacedim == 2))
      return internal::cell_cast<1, 2>(cell_accessor)->at_boundary();
    else if ((dim == 1) && (spacedim == 3))
      return internal::cell_cast<1, 3>(cell_accessor)->at_boundary();
    else if ((dim == 2) && (spacedim == 2))
      return internal::cell_cast<2, 2>(cell_accessor)->at_boundary();
    else if ((dim == 2) && (spacedim == 3))
      return internal::cell_cast<2, 3>(cell_accessor)->at_boundary();
    else
      return internal::cell_cast<3, 3>(cell_accessor)->at_boundary();
  }



  bool
  CellAccessorWrapper::has_boundary_lines() const
  {
    if ((dim == 1) && (spacedim == 1))
      return internal::cell_cast<1, 1>(cell_accessor)->has_boundary_lines();
    else if ((dim == 1) && (spacedim == 2))
      return internal::cell_cast<1, 2>(cell_accessor)->has_boundary_lines();
    else if ((dim == 1) && (spacedim == 3))
      return internal::cell_cast<1, 3>(cell_accessor)->has_boundary_lines();
    else if ((dim == 2) && (spacedim == 2))
      return internal::cell_cast<2, 2>(cell_accessor)->has_boundary_lines();
    else if ((dim == 2) && (spacedim == 3))
      return internal::cell_cast<2, 3>(cell_accessor)->has_boundary_lines();
    else
      return internal::cell_cast<3, 3>(cell_accessor)->has_boundary_lines();
  }



  CellAccessorWrapper
  CellAccessorWrapper::neighbor(const int i) const
  {
    AssertThrow(i < (2 * dim), ExcNeighborDoesNotExist(i, 2 * dim));

    if ((dim == 1) && (spacedim == 1))
      return construct_neighbor_wrapper<1, 1>(i);
    else if ((dim == 1) && (spacedim == 2))
      return construct_neighbor_wrapper<1, 2>(i);
    else if ((dim == 1) && (spacedim == 3))
      return construct_neighbor_wrapper<1, 3>(i);
    else if ((dim == 2) && (spacedim == 2))
      return construct_neighbor_wrapper<2, 2>(i);
    else if ((dim == 2) && (spacedim == 3))
      return construct_neighbor_wrapper<2, 3>(i);
    else if ((dim == 3) && (spacedim == 3))
      return construct_neighbor_wrapper<3, 3>(i);
    else
      AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
  }



  PYSPACE::list
  CellAccessorWrapper::faces() const
  {
    return internal::faces_wrapper<3, 3>(cell_accessor, dim, spacedim);
  }



  double
  CellAccessorWrapper::measure() const
  {
    if ((dim == 1) && (spacedim == 1))
      return internal::cell_cast<1, 1>(cell_accessor)->measure();
    else if ((dim == 1) && (spacedim == 2))
      return internal::cell_cast<1, 2>(cell_accessor)->measure();
    else if ((dim == 1) && (spacedim == 3))
      return internal::cell_cast<1, 3>(cell_accessor)->measure();
    else if ((dim == 2) && (spacedim == 2))
      return internal::cell_cast<2, 2>(cell_accessor)->measure();
    else if ((dim == 2) && (spacedim == 3))
      return internal::cell_cast<2, 3>(cell_accessor)->measure();
    else if ((dim == 3) && (spacedim == 3))
      return internal::cell_cast<3, 3>(cell_accessor)->measure();
    else
      AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
  }



  double
  CellAccessorWrapper::diameter() const
  {
    if ((dim == 1) && (spacedim == 1))
      return internal::cell_cast<1, 1>(cell_accessor)->diameter();
    else if ((dim == 1) && (spacedim == 2))
      return internal::cell_cast<1, 2>(cell_accessor)->diameter();
    else if ((dim == 1) && (spacedim == 3))
      return internal::cell_cast<1, 3>(cell_accessor)->diameter();
    else if ((dim == 2) && (spacedim == 2))
      return internal::cell_cast<2, 2>(cell_accessor)->diameter();
    else if ((dim == 2) && (spacedim == 3))
      return internal::cell_cast<2, 3>(cell_accessor)->diameter();
    else if ((dim == 3) && (spacedim == 3))
      return internal::cell_cast<3, 3>(cell_accessor)->diameter();
    else
      AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
  }



  bool
  CellAccessorWrapper::active() const
  {
    if ((dim == 1) && (spacedim == 1))
      return internal::cell_cast<1, 1>(cell_accessor)->is_active();
    else if ((dim == 1) && (spacedim == 2))
      return internal::cell_cast<1, 2>(cell_accessor)->is_active();
    else if ((dim == 1) && (spacedim == 3))
      return internal::cell_cast<1, 3>(cell_accessor)->is_active();
    else if ((dim == 2) && (spacedim == 2))
      return internal::cell_cast<2, 2>(cell_accessor)->is_active();
    else if ((dim == 2) && (spacedim == 3))
      return internal::cell_cast<2, 3>(cell_accessor)->is_active();
    else
      return internal::cell_cast<3, 3>(cell_accessor)->is_active();
  }



  int
  CellAccessorWrapper::level() const
  {
    if ((dim == 1) && (spacedim == 1))
      return internal::cell_cast<1, 1>(cell_accessor)->level();
    else if ((dim == 1) && (spacedim == 2))
      return internal::cell_cast<1, 2>(cell_accessor)->level();
    else if ((dim == 1) && (spacedim == 3))
      return internal::cell_cast<1, 3>(cell_accessor)->level();
    else if ((dim == 2) && (spacedim == 2))
      return internal::cell_cast<2, 2>(cell_accessor)->level();
    else if ((dim == 2) && (spacedim == 3))
      return internal::cell_cast<2, 3>(cell_accessor)->level();
    else
      return internal::cell_cast<3, 3>(cell_accessor)->level();
  }



  int
  CellAccessorWrapper::index() const
  {
    if ((dim == 1) && (spacedim == 1))
      return internal::cell_cast<1, 1>(cell_accessor)->index();
    else if ((dim == 1) && (spacedim == 2))
      return internal::cell_cast<1, 2>(cell_accessor)->index();
    else if ((dim == 1) && (spacedim == 3))
      return internal::cell_cast<1, 3>(cell_accessor)->index();
    else if ((dim == 2) && (spacedim == 2))
      return internal::cell_cast<2, 2>(cell_accessor)->index();
    else if ((dim == 2) && (spacedim == 3))
      return internal::cell_cast<2, 3>(cell_accessor)->index();
    else
      return internal::cell_cast<3, 3>(cell_accessor)->index();
  }



  bool
  CellAccessorWrapper::neighbor_is_coarser(const unsigned int neighbor) const
  {
    if ((dim == 1) && (spacedim == 1))
      return internal::cell_cast<1, 1>(cell_accessor)
        ->neighbor_is_coarser(neighbor);
    else if ((dim == 1) && (spacedim == 2))
      return internal::cell_cast<1, 2>(cell_accessor)
        ->neighbor_is_coarser(neighbor);
    else if ((dim == 1) && (spacedim == 3))
      return internal::cell_cast<1, 3>(cell_accessor)
        ->neighbor_is_coarser(neighbor);
    else if ((dim == 2) && (spacedim == 2))
      return internal::cell_cast<2, 2>(cell_accessor)
        ->neighbor_is_coarser(neighbor);
    else if ((dim == 2) && (spacedim == 3))
      return internal::cell_cast<2, 3>(cell_accessor)
        ->neighbor_is_coarser(neighbor);
    else
      return internal::cell_cast<3, 3>(cell_accessor)
        ->neighbor_is_coarser(neighbor);
  }



  unsigned int
  CellAccessorWrapper::neighbor_of_neighbor(const unsigned int neighbor) const
  {
    if ((dim == 1) && (spacedim == 1))
      return internal::cell_cast<1, 1>(cell_accessor)
        ->neighbor_of_neighbor(neighbor);
    else if ((dim == 1) && (spacedim == 2))
      return internal::cell_cast<1, 2>(cell_accessor)
        ->neighbor_of_neighbor(neighbor);
    else if ((dim == 1) && (spacedim == 3))
      return internal::cell_cast<1, 3>(cell_accessor)
        ->neighbor_of_neighbor(neighbor);
    else if ((dim == 2) && (spacedim == 2))
      return internal::cell_cast<2, 2>(cell_accessor)
        ->neighbor_of_neighbor(neighbor);
    else if ((dim == 2) && (spacedim == 3))
      return internal::cell_cast<2, 3>(cell_accessor)
        ->neighbor_of_neighbor(neighbor);
    else
      return internal::cell_cast<3, 3>(cell_accessor)
        ->neighbor_of_neighbor(neighbor);
  }



  unsigned int
  CellAccessorWrapper::vertex_index(const unsigned int i) const
  {
    if ((dim == 1) && (spacedim == 1))
      return internal::cell_cast<1, 1>(cell_accessor)->vertex_index(i);
    else if ((dim == 1) && (spacedim == 2))
      return internal::cell_cast<1, 2>(cell_accessor)->vertex_index(i);
    else if ((dim == 1) && (spacedim == 3))
      return internal::cell_cast<1, 3>(cell_accessor)->vertex_index(i);
    else if ((dim == 2) && (spacedim == 2))
      return internal::cell_cast<2, 2>(cell_accessor)->vertex_index(i);
    else if ((dim == 2) && (spacedim == 3))
      return internal::cell_cast<2, 3>(cell_accessor)->vertex_index(i);
    else
      return internal::cell_cast<3, 3>(cell_accessor)->vertex_index(i);
  }

} // namespace python

DEAL_II_NAMESPACE_CLOSE
