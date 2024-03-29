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

#include <deal.II/base/types.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <dealii-python-bindings/cell_accessor_wrapper.h>
#include <dealii-python-bindings/triangulation_wrapper.h>

#include <fstream>

DEAL_II_NAMESPACE_OPEN

#ifdef USE_BOOST_PYTHON
#  define PYSIZET int
#endif
#ifdef USE_PYBIND11
#  define PYSIZET pybind11::size_t
#endif



namespace python
{
  namespace internal
  {
    template <int dim, int spacedim>
    void
    create_triangulation(const PYSPACE::list &vertices_list,
                         const PYSPACE::list &cells_vertices,
                         void                *triangulation)
    {
      Triangulation<dim, spacedim> *tria =
        static_cast<Triangulation<dim, spacedim> *>(triangulation);
      tria->clear();

      const size_t                 n_vertices = PYSPACE::len(vertices_list);
      std::vector<Point<spacedim>> vertices(n_vertices);
      for (size_t i = 0; i < n_vertices; ++i)
        {
#ifdef USE_BOOST_PYTHON
          boost::python::list vertex =
            boost::python::extract<boost::python::list>(vertices_list[i]);
          for (int d = 0; d < spacedim; ++d)
            vertices[i][d] = boost::python::extract<double>(vertex[d]);
        }
#endif
#ifdef USE_PYBIND11
      pybind11::list vertex = vertices_list[i];
      for (int d = 0; d < spacedim; ++d)
        vertices[i][d] = pybind11::cast<double>(vertex[d]);
    }
#endif

    const size_t               n_cells = PYSPACE::len(cells_vertices);
    std::vector<CellData<dim>> cell_data(n_cells);
    for (size_t i = 0; i < n_cells; ++i)
      {
#ifdef USE_BOOST_PYTHON
        boost::python::list vertex_indices =
          boost::python::extract<boost::python::list>(cells_vertices[i]);
#endif
#ifdef USE_PYBIND11
        pybind11::list vertex_indices = cells_vertices[i];
#endif

        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
#ifdef USE_BOOST_PYTHON
          cell_data[i].vertices[v] =
            boost::python::extract<unsigned int>(vertex_indices[v]);
#endif
#ifdef USE_PYBIND11
        cell_data[i].vertices[v] =
          pybind11::cast<unsigned int>(vertex_indices[v]);
#endif
      }

    tria->create_triangulation(vertices, cell_data, SubCellData());
  }



  template <int d, int sd>
  void
  create_triangulation_wrapper(const PYSPACE::list &vertices_list,
                               const PYSPACE::list &cells_vertices,
                               void                *triangulation,
                               const int            dim,
                               const int            spacedim)
  {
    if (dim == d && spacedim == sd)
      return create_triangulation<d, sd>(vertices_list,
                                         cells_vertices,
                                         triangulation);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return create_triangulation_wrapper<sd - 1, sd - 1>(
            vertices_list, cells_vertices, triangulation, dim, spacedim);
      }
    else
      {
        return create_triangulation_wrapper<d - 1, sd>(
          vertices_list, cells_vertices, triangulation, dim, spacedim);
      }
  }


  template <int dim, int spacedim>
  void
  generate_hyper_cube(const double left,
                      const double right,
                      const bool   colorize,
                      void        *triangulation)
  {
    Triangulation<dim, spacedim> *tria =
      static_cast<Triangulation<dim, spacedim> *>(triangulation);
    tria->clear();
    GridGenerator::hyper_cube(*tria, left, right, colorize);
  }



  template <int d, int sd>
  void
  generate_hyper_cube_wrapper(const double left,
                              const double right,
                              const bool   colorize,
                              void        *triangulation,
                              const int    dim,
                              const int    spacedim)
  {
    if (dim == d && spacedim == sd)
      return generate_hyper_cube<d, sd>(left, right, colorize, triangulation);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return generate_hyper_cube_wrapper<sd - 1, sd - 1>(
            left, right, colorize, triangulation, dim, spacedim);
      }
    else
      {
        return generate_hyper_cube_wrapper<d - 1, sd>(
          left, right, colorize, triangulation, dim, spacedim);
      }
  }



  template <int dim>
  void
  generate_simplex(std::vector<PointWrapper> &wrapped_points,
                   void                      *triangulation)
  {
    // Cast the PointWrapper objects to Point<dim>
    std::vector<Point<dim>> points(dim + 1);
    for (int i = 0; i < dim + 1; ++i)
      points[i] = *(static_cast<Point<dim> *>((wrapped_points[i]).get_point()));

    Triangulation<dim> *tria = static_cast<Triangulation<dim> *>(triangulation);
    tria->clear();
    GridGenerator::simplex(*tria, points);
  }



  template <int dim, int spacedim>
  void
  generate_subdivided_hyper_cube(const unsigned int repetitions,
                                 const double       left,
                                 const double       right,
                                 void              *triangulation)
  {
    Triangulation<dim, spacedim> *tria =
      static_cast<Triangulation<dim, spacedim> *>(triangulation);
    tria->clear();
    GridGenerator::subdivided_hyper_cube(*tria, repetitions, left, right);
  }



  template <int d, int sd>
  void
  generate_subdivided_hyper_cube_wrapper(const unsigned int repetitions,
                                         const double       left,
                                         const double       right,
                                         void              *triangulation,
                                         const int          dim,
                                         const int          spacedim)
  {
    if (dim == d && spacedim == sd)
      return generate_subdivided_hyper_cube<d, sd>(repetitions,
                                                   left,
                                                   right,
                                                   triangulation);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return generate_subdivided_hyper_cube_wrapper<sd - 1, sd - 1>(
            repetitions, left, right, triangulation, dim, spacedim);
      }
    else
      {
        return generate_subdivided_hyper_cube_wrapper<d - 1, sd>(
          repetitions, left, right, triangulation, dim, spacedim);
      }
  }



  template <int dim, int spacedim>
  void
  generate_hyper_rectangle(PointWrapper &p1,
                           PointWrapper &p2,
                           const bool    colorize,
                           void         *triangulation)
  {
    AssertThrow(
      p1.get_dim() == dim,
      ExcMessage(
        "Dimension of p1 is not the same as the dimension of the Triangulation."));
    AssertThrow(
      p2.get_dim() == dim,
      ExcMessage(
        "Dimension of p2 is not the same as the dimension of the Triangulation."));
    // Cast the PointWrapper object to Point<dim>
    Point<dim> point_1 = *(static_cast<Point<dim> *>(p1.get_point()));
    Point<dim> point_2 = *(static_cast<Point<dim> *>(p2.get_point()));

    Triangulation<dim, spacedim> *tria =
      static_cast<Triangulation<dim, spacedim> *>(triangulation);
    tria->clear();
    GridGenerator::hyper_rectangle(*tria, point_1, point_2, colorize);
  }



  template <int d, int sd>
  void
  generate_hyper_rectangle_wrapper(PointWrapper &p1,
                                   PointWrapper &p2,
                                   const bool    colorize,
                                   void         *triangulation,
                                   const int     dim,
                                   const int     spacedim)
  {
    if (dim == d && spacedim == sd)
      return generate_hyper_rectangle<d, sd>(p1, p2, colorize, triangulation);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return generate_hyper_rectangle_wrapper<sd - 1, sd - 1>(
            p1, p2, colorize, triangulation, dim, spacedim);
      }
    else
      {
        return generate_hyper_rectangle_wrapper<d - 1, sd>(
          p1, p2, colorize, triangulation, dim, spacedim);
      }
  }



  template <int dim>
  void
  generate_hyper_cube_with_cylindrical_hole(const double       inner_radius,
                                            const double       outer_radius,
                                            const double       L,
                                            const unsigned int repetitions,
                                            const bool         colorize,
                                            void              *triangulation)
  {
    Triangulation<dim> *tria = static_cast<Triangulation<dim> *>(triangulation);
    tria->clear();
    GridGenerator::hyper_cube_with_cylindrical_hole(
      *tria, inner_radius, outer_radius, L, repetitions, colorize);
  }



  template <int dim, int spacedim>
  void
  generate_subdivided_hyper_rectangle(
    const std::vector<unsigned int> &repetitions,
    PointWrapper                    &p1,
    PointWrapper                    &p2,
    const bool                       colorize,
    void                            *triangulation)
  {
    AssertThrow(
      p1.get_dim() == dim,
      ExcMessage(
        "Dimension of p1 is not the same as the dimension of the Triangulation."));
    AssertThrow(
      p2.get_dim() == dim,
      ExcMessage(
        "Dimension of p2 is not the same as the dimension of the Triangulation."));
    // Cast the PointWrapper object to Point<dim>
    Point<dim> point_1 = *(static_cast<Point<dim> *>(p1.get_point()));
    Point<dim> point_2 = *(static_cast<Point<dim> *>(p2.get_point()));

    Triangulation<dim, spacedim> *tria =
      static_cast<Triangulation<dim, spacedim> *>(triangulation);
    tria->clear();
    GridGenerator::subdivided_hyper_rectangle(
      *tria, repetitions, point_1, point_2, colorize);
  }



  template <int d, int sd>
  void
  generate_subdivided_hyper_rectangle_wrapper(
    const std::vector<unsigned int> &repetitions,
    PointWrapper                    &p1,
    PointWrapper                    &p2,
    const bool                       colorize,
    void                            *triangulation,
    const int                        dim,
    const int                        spacedim)
  {
    if (dim == d && spacedim == sd)
      return generate_subdivided_hyper_rectangle<d, sd>(
        repetitions, p1, p2, colorize, triangulation);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return generate_subdivided_hyper_rectangle_wrapper<sd - 1, sd - 1>(
            repetitions, p1, p2, colorize, triangulation, dim, spacedim);
      }
    else
      {
        return generate_subdivided_hyper_rectangle_wrapper<d - 1, sd>(
          repetitions, p1, p2, colorize, triangulation, dim, spacedim);
      }
  }



  template <int dim>
  void
  generate_subdivided_steps_hyper_rectangle(
    const std::vector<std::vector<double>> &step_sizes,
    PointWrapper                           &p1,
    PointWrapper                           &p2,
    const bool                              colorize,
    void                                   *triangulation)
  {
    AssertThrow(
      p1.get_dim() == dim,
      ExcMessage(
        "Dimension of p1 is not the same as the dimension of the Triangulation."));
    AssertThrow(
      p2.get_dim() == dim,
      ExcMessage(
        "Dimension of p2 is not the same as the dimension of the Triangulation."));
    // Cast the PointWrapper object to Point<dim>
    Point<dim> point_1 = *(static_cast<Point<dim> *>(p1.get_point()));
    Point<dim> point_2 = *(static_cast<Point<dim> *>(p2.get_point()));

    Triangulation<dim> *tria = static_cast<Triangulation<dim> *>(triangulation);
    tria->clear();
    GridGenerator::subdivided_hyper_rectangle(
      *tria, step_sizes, point_1, point_2, colorize);
  }



  template <int dim>
  void
  generate_subdivided_material_hyper_rectangle(
    const std::vector<std::vector<double>> &spacing,
    PointWrapper                           &p,
    const Table<dim, types::material_id>   &material_ids,
    const bool                              colorize,
    void                                   *triangulation)
  {
    AssertThrow(
      p.get_dim() == dim,
      ExcMessage(
        "Dimension of p is not the same as the dimension of the Triangulation."));
    // Cast the PointWrapper object to Point<dim>
    Point<dim>          point = *(static_cast<Point<dim> *>(p.get_point()));
    Triangulation<dim> *tria = static_cast<Triangulation<dim> *>(triangulation);
    tria->clear();
    GridGenerator::subdivided_hyper_rectangle(
      *tria, spacing, point, material_ids, colorize);
  }



  template <int dim, int spacedim>
  void
  generate_cheese(const std::vector<unsigned int> &holes, void *triangulation)
  {
    Triangulation<dim, spacedim> *tria =
      static_cast<Triangulation<dim, spacedim> *>(triangulation);
    tria->clear();
    GridGenerator::cheese(*tria, holes);
  }



  template <int d, int sd>
  void
  generate_cheese_wrapper(const std::vector<unsigned int> &holes,
                          void                            *triangulation,
                          const int                        dim,
                          const int                        spacedim)
  {
    if (dim == d && spacedim == sd)
      return generate_cheese<d, sd>(holes, triangulation);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return generate_cheese_wrapper<sd - 1, sd - 1>(holes,
                                                         triangulation,
                                                         dim,
                                                         spacedim);
      }
    else
      {
        return generate_cheese_wrapper<d - 1, sd>(holes,
                                                  triangulation,
                                                  dim,
                                                  spacedim);
      }
  }



  template <int dim>
  void
  generate_general_cell(std::vector<PointWrapper> &wrapped_points,
                        const bool                 colorize,
                        void                      *triangulation)
  {
    // Cast the PointWrapper objects to Point<dim>
    const unsigned int      size = wrapped_points.size();
    std::vector<Point<dim>> points(size);
    for (unsigned int i = 0; i < size; ++i)
      points[i] = *(static_cast<Point<dim> *>((wrapped_points[i]).get_point()));

    Triangulation<dim> *tria = static_cast<Triangulation<dim> *>(triangulation);
    tria->clear();
    GridGenerator::general_cell(*tria, points, colorize);
  }



  template <int dim>
  void
  generate_parallelogram(std::vector<PointWrapper> &wrapped_points,
                         const bool                 colorize,
                         void                      *triangulation)
  {
    // Cast the PointWrapper objects to Point<dim>
    Point<dim> points[dim];
    for (unsigned int i = 0; i < dim; ++i)
      points[i] = *(static_cast<Point<dim> *>((wrapped_points[i]).get_point()));

    Triangulation<dim> *tria = static_cast<Triangulation<dim> *>(triangulation);
    tria->clear();
    GridGenerator::parallelogram(*tria, points, colorize);
  }



  template <int dim>
  void
  generate_parallelepiped(std::vector<PointWrapper> &wrapped_points,
                          const bool                 colorize,
                          void                      *triangulation)
  {
    // Cast the PointWrapper objects to Point<dim>
    Point<dim> points[dim];
    for (unsigned int i = 0; i < dim; ++i)
      points[i] = *(static_cast<Point<dim> *>((wrapped_points[i]).get_point()));

    Triangulation<dim> *tria = static_cast<Triangulation<dim> *>(triangulation);
    tria->clear();
    GridGenerator::parallelepiped(*tria, points, colorize);
  }



  template <int dim>
  void
  generate_fixed_subdivided_parallelepiped(
    unsigned int               n_subdivisions,
    std::vector<PointWrapper> &wrapped_points,
    const bool                 colorize,
    void                      *triangulation)
  {
    // Cast the PointWrapper objects to Point<dim>
    Point<dim> points[dim];
    for (unsigned int i = 0; i < dim; ++i)
      points[i] = *(static_cast<Point<dim> *>((wrapped_points[i]).get_point()));

    Triangulation<dim> *tria = static_cast<Triangulation<dim> *>(triangulation);
    tria->clear();
    GridGenerator::subdivided_parallelepiped(*tria,
                                             n_subdivisions,
                                             points,
                                             colorize);
  }



  template <int dim>
  void
  generate_varying_subdivided_parallelepiped(
    std::vector<unsigned int> &n_subdivisions,
    std::vector<PointWrapper> &wrapped_points,
    const bool                 colorize,
    void                      *triangulation)
  {
    // Cast the PointWrapper objects to Point<dim>
    Point<dim>   points[dim];
    unsigned int subdivisions[dim];
    for (unsigned int i = 0; i < dim; ++i)
      {
        points[i] =
          *(static_cast<Point<dim> *>((wrapped_points[i]).get_point()));
        subdivisions[i] = n_subdivisions[i];
      }

    Triangulation<dim> *tria = static_cast<Triangulation<dim> *>(triangulation);
    tria->clear();
    GridGenerator::subdivided_parallelepiped(*tria,
                                             subdivisions,
                                             points,
                                             colorize);
  }



  template <int dim>
  void
  generate_enclosed_hyper_cube(const double left,
                               const double right,
                               const double thickness,
                               const bool   colorize,
                               void        *triangulation)
  {
    Triangulation<dim> *tria = static_cast<Triangulation<dim> *>(triangulation);
    tria->clear();
    GridGenerator::enclosed_hyper_cube(*tria, left, right, thickness, colorize);
  }



  template <int dim>
  void
  generate_hyper_ball(PointWrapper &center,
                      const double  radius,
                      void         *triangulation)
  {
    // Cast the PointWrapper object to Point<dim>
    Point<dim> center_point = *(static_cast<Point<dim> *>(center.get_point()));

    Triangulation<dim> *tria = static_cast<Triangulation<dim> *>(triangulation);
    tria->clear();
    GridGenerator::hyper_ball(*tria, center_point, radius);
  }



  template <int dim, int spacedim>
  void
  generate_hyper_sphere(PointWrapper &center,
                        const double  radius,
                        void         *triangulation)
  {
    // Cast the PointWrapper object to Point<dim>
    Point<spacedim> center_point =
      *(static_cast<Point<spacedim> *>(center.get_point()));

    Triangulation<dim, spacedim> *tria =
      static_cast<Triangulation<dim, spacedim> *>(triangulation);
    tria->clear();
    GridGenerator::hyper_sphere(*tria, center_point, radius);
  }


  template <int dim>
  void
  generate_hyper_shell(PointWrapper  &center,
                       const double   inner_radius,
                       const double   outer_radius,
                       const unsigned n_cells,
                       bool           colorize,
                       void          *triangulation)
  {
    // Cast the PointWrapper object to Point<dim>
    Point<dim> center_point = *(static_cast<Point<dim> *>(center.get_point()));

    Triangulation<dim> *tria = static_cast<Triangulation<dim> *>(triangulation);
    tria->clear();
    GridGenerator::hyper_shell(
      *tria, center_point, inner_radius, outer_radius, n_cells, colorize);
  }


  template <int dim>
  void
  generate_quarter_hyper_ball(PointWrapper &center,
                              const double  radius,
                              void         *triangulation)
  {
    // Cast the PointWrapper object to Point<dim>
    Point<dim> center_point = *(static_cast<Point<dim> *>(center.get_point()));

    Triangulation<dim> *tria = static_cast<Triangulation<dim> *>(triangulation);
    tria->clear();
    GridGenerator::quarter_hyper_ball(*tria, center_point, radius);
  }



  template <int dim>
  void
  generate_half_hyper_ball(PointWrapper &center,
                           const double  radius,
                           void         *triangulation)
  {
    // Cast the PointWrapper object to Point<dim>
    Point<dim> center_point = *(static_cast<Point<dim> *>(center.get_point()));

    Triangulation<dim> *tria = static_cast<Triangulation<dim> *>(triangulation);
    tria->clear();
    GridGenerator::half_hyper_ball(*tria, center_point, radius);
  }



  template <int dim, int spacedim>
  void
  scale(const double scaling_factor, void *triangulation)
  {
    Triangulation<dim, spacedim> *tria =
      static_cast<Triangulation<dim, spacedim> *>(triangulation);
    GridTools::scale(scaling_factor, *tria);
  }



  template <int d, int sd>
  void
  scale_wrapper(const double scaling_factor,
                void        *triangulation,
                const int    dim,
                const int    spacedim)
  {
    if (dim == d && spacedim == sd)
      return scale<d, sd>(scaling_factor, triangulation);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return scale_wrapper<sd - 1, sd - 1>(scaling_factor,
                                               triangulation,
                                               dim,
                                               spacedim);
      }
    else
      {
        return scale_wrapper<d - 1, sd>(scaling_factor,
                                        triangulation,
                                        dim,
                                        spacedim);
      }
  }



  template <int dim, int spacedim>
  void
  shift(PYSPACE::list &shift_list, void *triangulation)
  {
    // Extract the shift vector from the python list
    Tensor<1, spacedim> shift_vector;
    for (int i = 0; i < spacedim; ++i)
#ifdef USE_BOOST_PYTHON
      shift_vector[i] = boost::python::extract<double>(shift_list[i]);
#endif
#ifdef USE_PYBIND11
    shift_vector[i] = pybind11::cast<double>(shift_list[i]);
#endif

    Triangulation<dim, spacedim> *tria =
      static_cast<Triangulation<dim, spacedim> *>(triangulation);
    GridTools::shift(shift_vector, *tria);
  }



  template <int d, int sd>
  void
  shift_wrapper(PYSPACE::list &shift_list,
                void          *triangulation,
                const int      dim,
                const int      spacedim)
  {
    if (dim == d && spacedim == sd)
      return shift<d, sd>(shift_list, triangulation);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return shift_wrapper<sd - 1, sd - 1>(shift_list,
                                               triangulation,
                                               dim,
                                               spacedim);
      }
    else
      {
        return shift_wrapper<d - 1, sd>(shift_list,
                                        triangulation,
                                        dim,
                                        spacedim);
      }
  }



  template <int dim, int spacedim>
  void
  merge_triangulations(TriangulationWrapper &triangulation_1,
                       TriangulationWrapper &triangulation_2,
                       void                 *triangulation)
  {
    Triangulation<dim, spacedim> *tria =
      static_cast<Triangulation<dim, spacedim> *>(triangulation);
    tria->clear();
    Triangulation<dim, spacedim> *tria_1 =
      static_cast<Triangulation<dim, spacedim> *>(
        triangulation_1.get_triangulation());
    Triangulation<dim, spacedim> *tria_2 =
      static_cast<Triangulation<dim, spacedim> *>(
        triangulation_2.get_triangulation());
    GridGenerator::merge_triangulations(*tria_1, *tria_2, *tria);
    // We need to reassign tria to triangulation because tria was cleared
    // inside merge_triangulations.
    triangulation = tria;
  }



  template <int d, int sd>
  void
  merge_triangulations_wrapper(TriangulationWrapper &triangulation_1,
                               TriangulationWrapper &triangulation_2,
                               void                 *triangulation,
                               const int             dim,
                               const int             spacedim)
  {
    if (dim == d && spacedim == sd)
      return merge_triangulations<d, sd>(triangulation_1,
                                         triangulation_2,
                                         triangulation);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return merge_triangulations_wrapper<sd - 1, sd - 1>(
            triangulation_1, triangulation_2, triangulation, dim, spacedim);
      }
    else
      {
        return merge_triangulations_wrapper<d - 1, sd>(
          triangulation_1, triangulation_2, triangulation, dim, spacedim);
      }
  }



  template <int dim, int spacedim_1, int spacedim_2>
  void
  flatten_triangulation(void *triangulation, TriangulationWrapper &tria_out)
  {
    Triangulation<dim, spacedim_1> *tria =
      static_cast<Triangulation<dim, spacedim_1> *>(triangulation);
    Triangulation<dim, spacedim_2> *tria_2 =
      static_cast<Triangulation<dim, spacedim_2> *>(
        tria_out.get_triangulation());
    GridGenerator::flatten_triangulation(*tria, *tria_2);
  }



  template <int dim, int spacedim>
  void
  distort_random(const double factor,
                 const bool   keep_boundary,
                 void        *triangulation)
  {
    Triangulation<dim, spacedim> *tria =
      static_cast<Triangulation<dim, spacedim> *>(triangulation);

    GridTools::distort_random(factor, *tria, keep_boundary);
  }



  template <int d, int sd>
  void
  distort_random_wrapper(const double factor,
                         const bool   keep_boundary,
                         void        *triangulation,
                         const int    dim,
                         const int    spacedim)
  {
    if (dim == d && spacedim == sd)
      return distort_random<d, sd>(factor, keep_boundary, triangulation);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return distort_random_wrapper<sd - 1, sd - 1>(
            factor, keep_boundary, triangulation, dim, spacedim);
      }
    else
      {
        return distort_random_wrapper<d - 1, sd>(
          factor, keep_boundary, triangulation, dim, spacedim);
      }
  }



  template <int dim, int spacedim>
  void
  transform(PYSPACE::object &transformation, void *triangulation)
  {
    Triangulation<dim, spacedim> *tria =
      static_cast<Triangulation<dim, spacedim> *>(triangulation);

    auto func = [&transformation](const Point<spacedim> &p_in) {
      PYSPACE::list p_list_in;

      for (int d = 0; d < spacedim; ++d)
        p_list_in.append(p_in[d]);
#ifdef USE_BOOST_PYTHON
      boost::python::list p_list_out =
        boost::python::extract<boost::python::list>(transformation(p_list_in));
#endif
#ifdef USE_PYBIND11
      pybind11::list p_list_out = transformation(p_list_in);
#endif

      Point<spacedim> p_out;

#ifdef USE_BOOST_PYTHON
      for (int d = 0; d < spacedim; ++d)
        p_out[d] = boost::python::extract<double>(p_list_out[d]);
#endif
#ifdef USE_PYBIND11
      for (int d = 0; d < spacedim; ++d)
        p_out[d] = pybind11::cast<double>(p_list_out[d]);
#endif
      return p_out;
    };

    GridTools::transform(func, *tria);
  }



  template <int d, int sd>
  void
  transform_wrapper(PYSPACE::object &transformation,
                    void            *triangulation,
                    const int        dim,
                    const int        spacedim)
  {
    if (dim == d && spacedim == sd)
      return transform<d, sd>(transformation, triangulation);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return transform_wrapper<sd - 1, sd - 1>(transformation,
                                                   triangulation,
                                                   dim,
                                                   spacedim);
      }
    else
      {
        return transform_wrapper<d - 1, sd>(transformation,
                                            triangulation,
                                            dim,
                                            spacedim);
      }
  }



  template <int dim, int spacedim>
  std::pair<int, int>
  find_active_cell_around_point(PointWrapper           &p,
                                MappingQGenericWrapper &mapping_wrapper,
                                void                   *triangulation)
  {
    Triangulation<dim, spacedim> *tria =
      static_cast<Triangulation<dim, spacedim> *>(triangulation);

    Point<spacedim> point = *(static_cast<Point<spacedim> *>(p.get_point()));

    if (mapping_wrapper.get_mapping() != nullptr)
      {
        const MappingQGeneric<dim, spacedim> *mapping =
          static_cast<const MappingQGeneric<dim, spacedim> *>(
            mapping_wrapper.get_mapping());

        auto cell_pair =
          GridTools::find_active_cell_around_point(*mapping, *tria, point);
        return std::make_pair(cell_pair.first->level(),
                              cell_pair.first->index());
      }
    else
      {
        auto cell = GridTools::find_active_cell_around_point(*tria, point);
        return std::make_pair(cell->level(), cell->index());
      }
  }



  template <int d, int sd>
  std::pair<int, int>
  find_active_cell_around_point_wrapper(PointWrapper           &p,
                                        MappingQGenericWrapper &mapping_wrapper,
                                        void                   *triangulation,
                                        const int               dim,
                                        const int               spacedim)
  {
    if (dim == d && spacedim == sd)
      return find_active_cell_around_point<d, sd>(p,
                                                  mapping_wrapper,
                                                  triangulation);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return find_active_cell_around_point_wrapper<sd - 1, sd - 1>(
            p, mapping_wrapper, triangulation, dim, spacedim);
      }
    else
      {
        return find_active_cell_around_point_wrapper<d - 1, sd>(
          p, mapping_wrapper, triangulation, dim, spacedim);
      }
  }



  template <int dim, int spacedim>
  PYSPACE::list
  compute_aspect_ratio_of_cells(
    const MappingQGenericWrapper &mapping_wrapper,
    const QuadratureWrapper      &quadrature_wrapper,
    const TriangulationWrapper   &triangulation_wrapper)
  {
    const Triangulation<dim, spacedim> *tria =
      static_cast<const Triangulation<dim, spacedim> *>(
        triangulation_wrapper.get_triangulation());

    const Quadrature<dim> *quad =
      static_cast<const Quadrature<dim> *>(quadrature_wrapper.get_quadrature());

    const MappingQGeneric<dim, spacedim> *mapping =
      static_cast<const MappingQGeneric<dim, spacedim> *>(
        mapping_wrapper.get_mapping());

    auto aspect_ratios =
      GridTools::compute_aspect_ratio_of_cells(*mapping, *tria, *quad);

    PYSPACE::list ratios;
    for (size_t i = 0; i < aspect_ratios.size(); ++i)
      ratios.append(aspect_ratios[i]);

    return ratios;
  }



  template <int dim, int spacedim>
  PYSPACE::list
  find_cells_adjacent_to_vertex(const unsigned int    vertex_index,
                                TriangulationWrapper &triangulation_wrapper)
  {
    Triangulation<dim, spacedim> *tria =
      static_cast<Triangulation<dim, spacedim> *>(
        triangulation_wrapper.get_triangulation());

    std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
      adjacent_cells =
        GridTools::find_cells_adjacent_to_vertex(*tria, vertex_index);

    PYSPACE::list cells;
    for (auto &cell : adjacent_cells)
      cells.append(CellAccessorWrapper(triangulation_wrapper,
                                       cell->level(),
                                       cell->index()));

    return cells;
  }



  template <int d, int sd>
  PYSPACE::list
  find_cells_adjacent_to_vertex_wrapper(
    const unsigned int    vertex_index,
    TriangulationWrapper &triangulation_wrapper,
    const int             dim,
    const int             spacedim)
  {
    if (dim == d && spacedim == sd)
      return find_cells_adjacent_to_vertex<d, sd>(vertex_index,
                                                  triangulation_wrapper);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return find_cells_adjacent_to_vertex_wrapper<sd - 1, sd - 1>(
            vertex_index, triangulation_wrapper, dim, spacedim);
      }
    else
      {
        return find_cells_adjacent_to_vertex_wrapper<d - 1, sd>(
          vertex_index, triangulation_wrapper, dim, spacedim);
      }
  }



  template <int dim, int spacedim>
  PYSPACE::list
  active_cells(TriangulationWrapper &triangulation_wrapper)
  {
    Triangulation<dim, spacedim> *tria =
      static_cast<Triangulation<dim, spacedim> *>(
        triangulation_wrapper.get_triangulation());
    PYSPACE::list cells;
    for (auto &cell : tria->active_cell_iterators())
      cells.append(CellAccessorWrapper(triangulation_wrapper,
                                       cell->level(),
                                       cell->index()));

    return cells;
  }



  template <int d, int sd>
  PYSPACE::list
  active_cells_wrapper(TriangulationWrapper &triangulation_wrapper,
                       const int             dim,
                       const int             spacedim)
  {
    if (dim == d && spacedim == sd)
      return active_cells<d, sd>(triangulation_wrapper);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return active_cells_wrapper<sd - 1, sd - 1>(triangulation_wrapper,
                                                      dim,
                                                      spacedim);
      }
    else
      {
        return active_cells_wrapper<d - 1, sd>(triangulation_wrapper,
                                               dim,
                                               spacedim);
      }
  }



  template <int dim, int spacedim>
  double
  maximal_cell_diameter(const void *triangulation)
  {
    const Triangulation<dim, spacedim> *tria =
      static_cast<const Triangulation<dim, spacedim> *>(triangulation);
    return GridTools::maximal_cell_diameter(*tria);
  }



  template <int d, int sd>
  double
  maximal_cell_diameter_wrapper(const void *triangulation,
                                const int   dim,
                                const int   spacedim)
  {
    if (dim == d && spacedim == sd)
      return maximal_cell_diameter<d, sd>(triangulation);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return maximal_cell_diameter_wrapper<sd - 1, sd - 1>(triangulation,
                                                               dim,
                                                               spacedim);
      }
    else
      {
        return maximal_cell_diameter_wrapper<d - 1, sd>(triangulation,
                                                        dim,
                                                        spacedim);
      }
  }



  template <int dim, int spacedim>
  double
  minimal_cell_diameter(const void *triangulation)
  {
    const Triangulation<dim, spacedim> *tria =
      static_cast<const Triangulation<dim, spacedim> *>(triangulation);
    return GridTools::minimal_cell_diameter(*tria);
  }



  template <int d, int sd>
  double
  minimal_cell_diameter_wrapper(const void *triangulation,
                                const int   dim,
                                const int   spacedim)
  {
    if (dim == d && spacedim == sd)
      return minimal_cell_diameter<d, sd>(triangulation);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return minimal_cell_diameter_wrapper<sd - 1, sd - 1>(triangulation,
                                                               dim,
                                                               spacedim);
      }
    else
      {
        return minimal_cell_diameter_wrapper<d - 1, sd>(triangulation,
                                                        dim,
                                                        spacedim);
      }
  }



  template <int dim, int spacedim>
  void
  write(const std::string &filename,
        const std::string &format,
        const void        *triangulation)
  {
    const Triangulation<dim, spacedim> *tria =
      static_cast<const Triangulation<dim, spacedim> *>(triangulation);

    GridOut::OutputFormat output_format;
    if (format.compare("dx") == 0)
      output_format = GridOut::OutputFormat::dx;
    else if (format.compare("gnuplot") == 0)
      output_format = GridOut::OutputFormat::gnuplot;
    else if (format.compare("eps") == 0)
      output_format = GridOut::OutputFormat::eps;
    else if (format.compare("ucd") == 0)
      output_format = GridOut::OutputFormat::ucd;
    else if (format.compare("xfig") == 0)
      output_format = GridOut::OutputFormat::xfig;
    else if (format.compare("msh") == 0)
      output_format = GridOut::OutputFormat::msh;
    else if (format.compare("svg") == 0)
      output_format = GridOut::OutputFormat::svg;
    else if (format.compare("mathgl") == 0)
      output_format = GridOut::OutputFormat::mathgl;
    else if (format.compare("vtk") == 0)
      output_format = GridOut::OutputFormat::vtk;
    else if (format.compare("vtu") == 0)
      output_format = GridOut::OutputFormat::vtu;
    else
      output_format = GridOut::OutputFormat::none;

    GridOut       mesh_writer;
    std::ofstream ofs(filename);
    mesh_writer.write(*tria, ofs, output_format);
    ofs.close();
  }



  template <int d, int sd>
  void
  write_wrapper(const std::string &filename,
                const std::string &format,
                const void        *triangulation,
                const int          dim,
                const int          spacedim)
  {
    if (dim == d && spacedim == sd)
      return write<d, sd>(filename, format, triangulation);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return write_wrapper<sd - 1, sd - 1>(
            filename, format, triangulation, dim, spacedim);
      }
    else
      {
        return write_wrapper<d - 1, sd>(
          filename, format, triangulation, dim, spacedim);
      }
  }



  template <int dim, int spacedim>
  void
  read(const std::string &filename,
       const std::string &format,
       void              *triangulation)
  {
    Triangulation<dim, spacedim> *tria =
      static_cast<Triangulation<dim, spacedim> *>(triangulation);

    tria->clear();

    typename GridIn<dim, spacedim>::Format input_format =
      GridIn<dim, spacedim>::Format::Default;
    if (format.compare("msh") == 0)
      input_format = GridIn<dim, spacedim>::Format::msh;
    else if (format.compare("vtk") == 0)
      input_format = GridIn<dim, spacedim>::Format::vtk;
    else
      Assert(false,
             ExcMessage("Cannot read triangulation of the given format."));

    GridIn<dim, spacedim> mesh_reader;
    mesh_reader.attach_triangulation(*tria);
    std::ifstream ifs(filename);
    AssertThrow(ifs, ExcIO());
    mesh_reader.read(ifs, input_format);
    ifs.close();
  }



  template <int d, int sd>
  void
  read_wrapper(const std::string &filename,
               const std::string &format,
               void              *triangulation,
               const int          dim,
               const int          spacedim)
  {
    if (dim == d && spacedim == sd)
      return read<d, sd>(filename, format, triangulation);

    if constexpr (d == 1)
      {
        if constexpr (sd == 1)
          {
            AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
          }
        else
          return read_wrapper<sd - 1, sd - 1>(
            filename, format, triangulation, dim, spacedim);
      }
    else
      {
        return read_wrapper<d - 1, sd>(
          filename, format, triangulation, dim, spacedim);
      }
  }



  template <int d, int sd>
  void
  delete_triangulation_pointer(void     *triangulation,
                               const int dim,
                               const int spacedim)
  {
    if (dim == d && spacedim == sd)
      {
        Triangulation<d, sd> *tmp =
          static_cast<Triangulation<d, sd> *>(triangulation);
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
          delete_triangulation_pointer<sd - 1, sd - 1>(triangulation,
                                                       dim,
                                                       spacedim);
      }
    else
      {
        delete_triangulation_pointer<d - 1, sd>(triangulation, dim, spacedim);
      }
  }
} // namespace internal



TriangulationWrapper::TriangulationWrapper(const std::string &dimension)
{
  if ((dimension.compare("1D") == 0) || (dimension.compare("1d") == 0))
    setup("1D", "1D");
  else if ((dimension.compare("2D") == 0) || (dimension.compare("2d") == 0))
    setup("2D", "2D");
  else if ((dimension.compare("3D") == 0) || (dimension.compare("3d") == 0))
    setup("3D", "3D");
  else
    AssertThrow(false, ExcMessage("Dimension needs to be 1D, 2D or 3D"));
}



TriangulationWrapper::TriangulationWrapper(const std::string &dimension,
                                           const std::string &spacedimension)
{
  setup(dimension, spacedimension);
}



TriangulationWrapper::~TriangulationWrapper()
{
  if (triangulation != nullptr)
    internal::delete_triangulation_pointer<3, 3>(triangulation, dim, spacedim);
  dim = -1;
}



unsigned int
TriangulationWrapper::n_active_cells() const
{
  if ((dim == 1) && (spacedim == 1))
    {
      return (*static_cast<Triangulation<1, 1> *>(triangulation))
        .n_active_cells();
    }
  else if ((dim == 1) && (spacedim == 2))
    {
      return (*static_cast<Triangulation<1, 2> *>(triangulation))
        .n_active_cells();
    }
  else if ((dim == 1) && (spacedim == 3))
    {
      return (*static_cast<Triangulation<1, 3> *>(triangulation))
        .n_active_cells();
    }
  else if ((dim == 2) && (spacedim == 2))
    return (*static_cast<Triangulation<2, 2> *>(triangulation))
      .n_active_cells();
  else if ((dim == 2) && (spacedim == 3))
    return (*static_cast<Triangulation<2, 3> *>(triangulation))
      .n_active_cells();
  else if ((dim == 3) && (spacedim == 3))
    return (*static_cast<Triangulation<3, 3> *>(triangulation))
      .n_active_cells();
  else
    AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
}



void
TriangulationWrapper::create_triangulation(const PYSPACE::list &vertices,
                                           const PYSPACE::list &cells_vertices)
{
  internal::create_triangulation_wrapper<3, 3>(
    vertices, cells_vertices, triangulation, dim, spacedim);
}



void
TriangulationWrapper::generate_hyper_cube(const double left,
                                          const double right,
                                          const bool   colorize)
{
  internal::generate_hyper_cube_wrapper<3, 3>(
    left, right, colorize, triangulation, dim, spacedim);
}



void
TriangulationWrapper::generate_simplex(PYSPACE::list &vertices)
{
  AssertThrow(PYSPACE::len(vertices) == (PYSIZET)(dim + 1),
              ExcMessage("The number of vertices should be equal to dim+1."));
  AssertThrow(
    dim == spacedim,
    ExcMessage("This function is only implemented for dim equal to spacedim."));
  // Extract the PointWrapper object from the python list
  std::vector<PointWrapper> wrapped_points(dim + 1);
  for (int i = 0; i < dim + 1; ++i)
    {
#ifdef USE_BOOST_PYTHON
      wrapped_points[i] = boost::python::extract<PointWrapper>(vertices[i]);
#endif
#ifdef USE_PYBIND11
      wrapped_points[i] = pybind11::cast<PointWrapper>(vertices[i]);
#endif
      AssertThrow(wrapped_points[i].get_dim() == dim,
                  ExcMessage("Point of wrong dimension."));
    }

  if (dim == 1)
    internal::generate_simplex<1>(wrapped_points, triangulation);
  else if (dim == 2)
    internal::generate_simplex<2>(wrapped_points, triangulation);
  else if (dim == 3)
    internal::generate_simplex<3>(wrapped_points, triangulation);
  else
    AssertThrow(false, ExcMessage("Unsupported dimension."));
}



void
TriangulationWrapper::generate_subdivided_hyper_cube(
  const unsigned int repetitions,
  const double       left,
  const double       right)
{
  internal::generate_subdivided_hyper_cube_wrapper<3, 3>(
    repetitions, left, right, triangulation, dim, spacedim);
}



void
TriangulationWrapper::generate_hyper_rectangle(PointWrapper &p1,
                                               PointWrapper &p2,
                                               const bool    colorize)
{
  internal::generate_hyper_rectangle_wrapper<3, 3>(
    p1, p2, colorize, triangulation, dim, spacedim);
}



void
TriangulationWrapper::generate_subdivided_hyper_rectangle(
  PYSPACE::list &repetition_list,
  PointWrapper  &p1,
  PointWrapper  &p2,
  const bool     colorize)
{
  AssertThrow(
    PYSPACE::len(repetition_list) == (PYSIZET)dim,
    ExcMessage(
      "The list of repetitions must have the same length as the number of dimension."));

  // Extract the repetitions from the python list
  std::vector<unsigned int> repetitions(dim);
  for (int i = 0; i < dim; ++i)
#ifdef USE_BOOST_PYTHON
    repetitions[i] = boost::python::extract<unsigned int>(repetition_list[i]);
#endif
#ifdef USE_PYBIND11
  repetitions[i] = pybind11::cast<unsigned int>(repetition_list[i]);
#endif
  internal::generate_subdivided_hyper_rectangle_wrapper<3, 3>(
    repetitions, p1, p2, colorize, triangulation, dim, spacedim);
}



void
TriangulationWrapper::generate_subdivided_steps_hyper_rectangle(
  PYSPACE::list &step_sizes_list,
  PointWrapper  &p1,
  PointWrapper  &p2,
  const bool     colorize)
{
  AssertThrow(
    spacedim == dim,
    ExcMessage("This function is only implemented for dim equal to spacedim."));
  AssertThrow(
    PYSPACE::len(step_sizes_list) == (PYSIZET)dim,
    ExcMessage(
      "The list of step_sizes must have the same length as the number of dimension."));

  // Extract the step sizes from the python list
  std::vector<std::vector<double>> step_sizes(dim);
  for (int i = 0; i < dim; ++i)
    {
      step_sizes[i].resize(PYSPACE::len(step_sizes_list[i]));
      for (unsigned int j = 0; j < step_sizes[i].size(); ++j)
#ifdef USE_BOOST_PYTHON
        step_sizes[i][j] =
          boost::python::extract<double>(step_sizes_list[i][j]);
#endif
#ifdef USE_PYBIND11
      step_sizes[i][j] = pybind11::cast<double>(
        pybind11::cast<pybind11::list>(step_sizes_list[i])[j]);
#endif
    }

  if (dim == 1)
    internal::generate_subdivided_steps_hyper_rectangle<1>(
      step_sizes, p1, p2, colorize, triangulation);
  else if (dim == 2)
    internal::generate_subdivided_steps_hyper_rectangle<2>(
      step_sizes, p1, p2, colorize, triangulation);
  else if (dim == 3)
    internal::generate_subdivided_steps_hyper_rectangle<3>(
      step_sizes, p1, p2, colorize, triangulation);
  else
    AssertThrow(false, ExcMessage("Unsupported dimension."));
}



void
TriangulationWrapper::generate_subdivided_material_hyper_rectangle(
  PYSPACE::list &spacing_list,
  PointWrapper  &p,
  PYSPACE::list &material_id_list,
  const bool     colorize)
{
  AssertThrow(
    spacedim == dim,
    ExcMessage("This function is only implemented for dim equal to spacedim."));
  AssertThrow(
    PYSPACE::len(spacing_list) == (PYSIZET)dim,
    ExcMessage(
      "The list of spacing must have the same length as the number of dimension."));

  // Extract the spacing and the material ID from the python list
  std::vector<std::vector<double>> spacing(dim);
  for (int i = 0; i < dim; ++i)
    {
      spacing[i].resize(PYSPACE::len(spacing_list[i]));
      for (unsigned int j = 0; j < spacing[i].size(); ++j)
#ifdef USE_BOOST_PYTHON
        spacing[i][j] = boost::python::extract<double>(spacing_list[i][j]);
#endif
#ifdef USE_PYBIND11
      spacing[i][j] = pybind11::cast<double>(
        pybind11::cast<pybind11::list>(spacing_list[i])[j]);
#endif
    }
  if (dim == 1)
    {
      const unsigned int           index_0 = PYSPACE::len(material_id_list);
      Table<1, types::material_id> material_ids(index_0);
      for (unsigned int i = 0; i < index_0; ++i)
      // We cannot use extract<types::material_id> because boost will
      // throw an exception if we try to extract -1
#ifdef USE_BOOST_PYTHON
        material_ids[i] = boost::python::extract<int>(material_id_list[i]);
#endif
#ifdef USE_PYBIND11
      material_ids[i] = pybind11::cast<int>(material_id_list[i]);
#endif
      internal::generate_subdivided_material_hyper_rectangle<1>(
        spacing, p, material_ids, colorize, triangulation);
    }
  else if (dim == 2)
    {
      const unsigned int           index_0 = PYSPACE::len(material_id_list);
      const unsigned int           index_1 = PYSPACE::len(material_id_list[0]);
      Table<2, types::material_id> material_ids(index_0, index_1);
      for (unsigned int i = 0; i < index_0; ++i)
        for (unsigned int j = 0; j < index_1; ++j)
#ifdef USE_BOOST_PYTHON
          material_ids[i][j] =
            boost::python::extract<int>(material_id_list[i][j]);
#endif
#ifdef USE_PYBIND11
      material_ids[i][j] = pybind11::cast<int>(
        pybind11::cast<pybind11::list>(material_id_list[i])[j]);
#endif
      internal::generate_subdivided_material_hyper_rectangle<2>(
        spacing, p, material_ids, colorize, triangulation);
    }
  else if (dim == 3)
    {
      const unsigned int index_0 = PYSPACE::len(material_id_list);
      const unsigned int index_1 = PYSPACE::len(material_id_list[0]);
      const unsigned int index_2 = PYSPACE::len(material_id_list[0][0]);
      Table<3, types::material_id> material_ids(index_0, index_1, index_2);
      for (unsigned int i = 0; i < index_0; ++i)
        for (unsigned int j = 0; j < index_1; ++j)
          for (unsigned int k = 0; k < index_2; ++k)
#ifdef USE_BOOST_PYTHON
            material_ids[i][j][k] =
              boost::python::extract<int>(material_id_list[i][j][k]);
#endif
#ifdef USE_PYBIND11
      material_ids[i][j][k] =
        pybind11::cast<int>(pybind11::cast<pybind11::list>(
          pybind11::cast<pybind11::list>(material_id_list[i])[j])[k]);
#endif
      internal::generate_subdivided_material_hyper_rectangle<3>(
        spacing, p, material_ids, colorize, triangulation);
    }
  else
    AssertThrow(false, ExcMessage("Unsupported dimension."));
}



void
TriangulationWrapper::generate_cheese(PYSPACE::list &holes_list)
{
  const unsigned int        size = PYSPACE::len(holes_list);
  std::vector<unsigned int> holes(size);
  for (unsigned int i = 0; i < size; ++i)
#ifdef USE_BOOST_PYTHON
    holes[i] = boost::python::extract<unsigned int>(holes_list[i]);
#endif
#ifdef USE_PYBIND11
  holes[i] = pybind11::cast<unsigned int>(holes_list[i]);
#endif

  internal::generate_cheese_wrapper<3, 3>(holes, triangulation, dim, spacedim);
}



void
TriangulationWrapper::generate_general_cell(PYSPACE::list &vertices,
                                            const bool     colorize)
{
  AssertThrow(
    spacedim == dim,
    ExcMessage("This function is only implemented for dim equal to spacedim."));
  // Extract the PointWrapper object from the python list
  const int size = PYSPACE::len(vertices);
  AssertThrow(size > 0, ExcMessage("The vertices list is empty."));
  std::vector<PointWrapper> wrapped_points(size);
  for (int i = 0; i < size; ++i)
#ifdef USE_BOOST_PYTHON
    wrapped_points[i] = boost::python::extract<PointWrapper>(vertices[i]);
#endif
#ifdef USE_PYBIND11
  wrapped_points[i] = pybind11::cast<PointWrapper>(vertices[i]);
#endif
  if (dim == 1)
    internal::generate_general_cell<1>(wrapped_points, colorize, triangulation);
  else if (dim == 2)
    internal::generate_general_cell<2>(wrapped_points, colorize, triangulation);
  else if (dim == 3)
    internal::generate_general_cell<3>(wrapped_points, colorize, triangulation);
  else
    AssertThrow(false, ExcMessage("Unsupported dimension."));
}



void
TriangulationWrapper::generate_parallelogram(PYSPACE::list &corners,
                                             const bool     colorize)
{
  AssertThrow(
    spacedim == dim,
    ExcMessage("This function is only implemented for dim equal to spacedim."));
  // Extract the PointWrapper object from the python list
  AssertThrow(
    PYSPACE::len(corners) == (PYSIZET)dim,
    ExcMessage(
      "The list of corners must have the same length as the number of dimension."));
  std::vector<PointWrapper> wrapped_points(dim);
  for (int i = 0; i < dim; ++i)
#ifdef USE_BOOST_PYTHON
    wrapped_points[i] = boost::python::extract<PointWrapper>(corners[i]);
#endif
#ifdef USE_PYBIND11
  wrapped_points[i] = pybind11::cast<PointWrapper>(corners[i]);
#endif
  if (dim == 1)
    internal::generate_parallelogram<1>(wrapped_points,
                                        colorize,
                                        triangulation);
  else if (dim == 2)
    internal::generate_parallelogram<2>(wrapped_points,
                                        colorize,
                                        triangulation);
  else if (dim == 3)
    internal::generate_parallelogram<3>(wrapped_points,
                                        colorize,
                                        triangulation);
  else
    AssertThrow(false, ExcMessage("Unsupported dimension."));
}



void
TriangulationWrapper::generate_parallelepiped(PYSPACE::list &corners,
                                              const bool     colorize)
{
  AssertThrow(
    spacedim == dim,
    ExcMessage("This function is only implemented for dim equal to spacedim."));
  // Extract the PointWrapper object from the python list
  AssertThrow(
    PYSPACE::len(corners) == (PYSIZET)dim,
    ExcMessage(
      "The list of corners must have the same length as the number of dimension."));
  std::vector<PointWrapper> wrapped_points(dim);
  for (int i = 0; i < dim; ++i)
#ifdef USE_BOOST_PYTHON
    wrapped_points[i] = boost::python::extract<PointWrapper>(corners[i]);
#endif
#ifdef USE_PYBIND11
  wrapped_points[i] = pybind11::cast<PointWrapper>(corners[i]);
#endif
  if (dim == 1)
    internal::generate_parallelepiped<1>(wrapped_points,
                                         colorize,
                                         triangulation);
  else if (dim == 2)
    internal::generate_parallelepiped<2>(wrapped_points,
                                         colorize,
                                         triangulation);
  else if (dim == 3)
    internal::generate_parallelepiped<3>(wrapped_points,
                                         colorize,
                                         triangulation);
  else
    AssertThrow(false, ExcMessage("Unsupported dimension."));
}



void
TriangulationWrapper::generate_fixed_subdivided_parallelepiped(
  const unsigned int n_subdivisions,
  PYSPACE::list     &corners,
  const bool         colorize)
{
  AssertThrow(
    spacedim == dim,
    ExcMessage("This function is only implemented for dim equal to spacedim."));
  // Extract the PointWrapper object from the python list
  AssertThrow(
    PYSPACE::len(corners) == (PYSIZET)dim,
    ExcMessage(
      "The list of corners must have the same length as the number of dimension."));
  std::vector<PointWrapper> wrapped_points(dim);
  for (int i = 0; i < dim; ++i)
#ifdef USE_BOOST_PYTHON
    wrapped_points[i] = boost::python::extract<PointWrapper>(corners[i]);
#endif
#ifdef USE_PYBIND11
  wrapped_points[i] = pybind11::cast<PointWrapper>(corners[i]);
#endif
  if (dim == 1)
    {
      internal::generate_fixed_subdivided_parallelepiped<1>(n_subdivisions,
                                                            wrapped_points,
                                                            colorize,
                                                            triangulation);
    }
  else if (dim == 2)
    internal::generate_fixed_subdivided_parallelepiped<2>(n_subdivisions,
                                                          wrapped_points,
                                                          colorize,
                                                          triangulation);
  else if (dim == 3)
    internal::generate_fixed_subdivided_parallelepiped<3>(n_subdivisions,
                                                          wrapped_points,
                                                          colorize,
                                                          triangulation);
  else
    AssertThrow(false, ExcMessage("Unsupported dimension."));
}



void
TriangulationWrapper::generate_varying_subdivided_parallelepiped(
  PYSPACE::list &n_subdivisions,
  PYSPACE::list &corners,
  const bool     colorize)
{
  AssertThrow(
    spacedim == dim,
    ExcMessage("This function is only implemented for dim equal to spacedim."));
  // Extract the subdivisions from the python list
  AssertThrow(
    PYSPACE::len(n_subdivisions) == (PYSIZET)dim,
    ExcMessage(
      "The list of subdivisions must have the same length as the number of dimension."));
  std::vector<unsigned int> subdivisions(dim);
  for (int i = 0; i < dim; ++i)
#ifdef USE_BOOST_PYTHON
    subdivisions[i] = boost::python::extract<unsigned int>(n_subdivisions[i]);
#endif
#ifdef USE_PYBIND11
  subdivisions[i] = pybind11::cast<unsigned int>(n_subdivisions[i]);
#endif
  // Extract the PointWrapper object from the python list
  AssertThrow(
    PYSPACE::len(corners) == (PYSIZET)dim,
    ExcMessage(
      "The list of corners must have the same length as the number of dimension."));
  std::vector<PointWrapper> wrapped_points(dim);
  for (int i = 0; i < dim; ++i)
#ifdef USE_BOOST_PYTHON
    wrapped_points[i] = boost::python::extract<PointWrapper>(corners[i]);
#endif
#ifdef USE_PYBIND11
  wrapped_points[i] = pybind11::cast<PointWrapper>(corners[i]);
#endif
  if (dim == 1)
    internal::generate_varying_subdivided_parallelepiped<1>(subdivisions,
                                                            wrapped_points,
                                                            colorize,
                                                            triangulation);
  else if (dim == 2)
    internal::generate_varying_subdivided_parallelepiped<2>(subdivisions,
                                                            wrapped_points,
                                                            colorize,
                                                            triangulation);
  else if (dim == 3)
    internal::generate_varying_subdivided_parallelepiped<3>(subdivisions,
                                                            wrapped_points,
                                                            colorize,
                                                            triangulation);
  else
    AssertThrow(false, ExcMessage("Unsupported dimension."));
}



void
TriangulationWrapper::generate_enclosed_hyper_cube(const double left,
                                                   const double right,
                                                   const double thickness,
                                                   const bool   colorize)
{
  AssertThrow(
    spacedim == dim,
    ExcMessage("This function is only implemented for dim equal to spacedim."));

  if (dim == 1)
    internal::generate_enclosed_hyper_cube<1>(
      left, right, thickness, colorize, triangulation);
  else if (dim == 2)
    internal::generate_enclosed_hyper_cube<2>(
      left, right, thickness, colorize, triangulation);
  else if (dim == 3)
    internal::generate_enclosed_hyper_cube<3>(
      left, right, thickness, colorize, triangulation);
  else
    AssertThrow(false, ExcMessage("Unsupported dimension."));
}



void
TriangulationWrapper::generate_hyper_ball(PointWrapper &center,
                                          const double  radius)
{
  AssertThrow(
    dim == spacedim,
    ExcMessage("This function is only implemented for dim equal to spacedim."));
  if (dim == 1)
    internal::generate_hyper_ball<1>(center, radius, triangulation);
  else if (dim == 2)
    internal::generate_hyper_ball<2>(center, radius, triangulation);
  else if (dim == 3)
    internal::generate_hyper_ball<3>(center, radius, triangulation);
  else
    AssertThrow(false, ExcMessage("Unsupported dimension."));
}



void
TriangulationWrapper::generate_hyper_shell(PointWrapper  &center,
                                           const double   inner_radius,
                                           const double   outer_radius,
                                           const unsigned n_cells,
                                           bool           colorize)
{
  AssertThrow(
    dim == spacedim,
    ExcMessage("This function is only implemented for dim equal to spacedim."));
  if (dim == 1)
    internal::generate_hyper_shell<1>(
      center, inner_radius, outer_radius, n_cells, colorize, triangulation);
  else if (dim == 2)
    internal::generate_hyper_shell<2>(
      center, inner_radius, outer_radius, n_cells, colorize, triangulation);
  else if (dim == 3)
    internal::generate_hyper_shell<3>(
      center, inner_radius, outer_radius, n_cells, colorize, triangulation);
  else
    AssertThrow(false, ExcMessage("Unsupported dimension."));
}



void
TriangulationWrapper::generate_hyper_sphere(PointWrapper &center,
                                            const double  radius)
{
  AssertThrow(
    spacedim == dim + 1,
    ExcMessage(
      "This function is only implemented for spacedim equal to dim+1."));
  if (dim == 1)
    internal::generate_hyper_sphere<1, 2>(center, radius, triangulation);
  else if (dim == 2)
    internal::generate_hyper_sphere<2, 3>(center, radius, triangulation);
  else
    AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
}



void
TriangulationWrapper::generate_quarter_hyper_ball(PointWrapper &center,
                                                  const double  radius)
{
  AssertThrow(
    dim == spacedim,
    ExcMessage("This function is only implemented for dim equal to spacedim."));
  if (dim == 1)
    internal::generate_quarter_hyper_ball<1>(center, radius, triangulation);
  else if (dim == 2)
    internal::generate_quarter_hyper_ball<2>(center, radius, triangulation);
  else if (dim == 3)
    internal::generate_quarter_hyper_ball<3>(center, radius, triangulation);
  else
    AssertThrow(false, ExcMessage("Unsupported dimension."));
}


void
TriangulationWrapper::generate_half_hyper_ball(PointWrapper &center,
                                               const double  radius)
{
  AssertThrow(
    dim == spacedim,
    ExcMessage("This function is only implemented for dim equal to spacedim."));
  if (dim == 1)
    internal::generate_half_hyper_ball<1>(center, radius, triangulation);
  else if (dim == 2)
    internal::generate_half_hyper_ball<2>(center, radius, triangulation);
  else if (dim == 3)
    internal::generate_half_hyper_ball<3>(center, radius, triangulation);
  else
    AssertThrow(false, ExcMessage("Unsupported dimension."));
}



void
TriangulationWrapper::generate_hyper_cube_with_cylindrical_hole(
  const double       inner_radius,
  const double       outer_radius,
  const double       L,
  const unsigned int repetitions,
  const bool         colorize)
{
  AssertThrow(
    dim == spacedim,
    ExcMessage("This function is only implemented for dim equal to spacedim."));

  switch (dim)
    {
      case 1:
        internal::generate_hyper_cube_with_cylindrical_hole<1>(
          inner_radius, outer_radius, L, repetitions, colorize, triangulation);
        break;
      case 2:
        internal::generate_hyper_cube_with_cylindrical_hole<2>(
          inner_radius, outer_radius, L, repetitions, colorize, triangulation);
        break;
      case 3:
        internal::generate_hyper_cube_with_cylindrical_hole<3>(
          inner_radius, outer_radius, L, repetitions, colorize, triangulation);
        break;
      default:
        AssertThrow(false, ExcMessage("Unsupported dimension."));
    }
}



void
TriangulationWrapper::shift(PYSPACE::list &shift_list)
{
  AssertThrow(PYSPACE::len(shift_list) == (PYSIZET)spacedim,
              ExcMessage("Size of the shift vector is not equal to spacedim."));

  internal::shift_wrapper<3, 3>(shift_list, triangulation, dim, spacedim);
}



void
TriangulationWrapper::scale(const double scaling_factor)
{
  internal::scale_wrapper<3, 3>(scaling_factor, triangulation, dim, spacedim);
}



void
TriangulationWrapper::merge_triangulations(
  TriangulationWrapper &triangulation_1,
  TriangulationWrapper &triangulation_2)
{
  AssertThrow(
    triangulation_1.get_dim() == triangulation_2.get_dim(),
    ExcMessage(
      "Triangulation_1 and Triangulation_2 should have the same dimension."));
  AssertThrow(
    dim == triangulation_2.get_dim(),
    ExcMessage(
      "Triangulation and Triangulation_2 should have the same dimension."));
  AssertThrow(
    triangulation_1.get_spacedim() == triangulation_2.get_spacedim(),
    ExcMessage(
      "Triangulation_1 and Triangulation_2 should have the same space dimension."));
  AssertThrow(
    spacedim == triangulation_2.get_spacedim(),
    ExcMessage(
      "Triangulation and Triangulation_2 should have the same space dimension."));

  internal::merge_triangulations_wrapper<3, 3>(
    triangulation_1, triangulation_2, triangulation, dim, spacedim);
}



void
TriangulationWrapper::flatten_triangulation(TriangulationWrapper &tria_out)
{
  AssertThrow(
    dim == tria_out.get_dim(),
    ExcMessage(
      "The Triangulation and tria_out should have the same dimension."));

  // Why this check???
  AssertThrow(spacedim >= tria_out.get_spacedim(),
              ExcMessage(
                "The Triangulation should have a spacedim greater or equal "
                "to the spacedim of tria_out."));

  int spacedim_out = tria_out.get_spacedim();
  switch (dim)
    {
      case 1:
        switch (spacedim)
          {
            case 1:
              if (spacedim_out == 1)
                internal::flatten_triangulation<1, 1, 1>(triangulation,
                                                         tria_out);
              else if (spacedim_out == 2)
                internal::flatten_triangulation<1, 1, 2>(triangulation,
                                                         tria_out);
              else if (spacedim_out == 3)
                internal::flatten_triangulation<1, 1, 3>(triangulation,
                                                         tria_out);
              else
                AssertThrow(false,
                            ExcMessage("Unsupported spacedim for tria_out."));
              break;
            case 2:
              if (spacedim_out == 1)
                internal::flatten_triangulation<1, 2, 1>(triangulation,
                                                         tria_out);
              else if (spacedim_out == 2)
                internal::flatten_triangulation<1, 2, 2>(triangulation,
                                                         tria_out);
              else if (spacedim_out == 3)
                internal::flatten_triangulation<1, 2, 3>(triangulation,
                                                         tria_out);
              else
                AssertThrow(false,
                            ExcMessage("Unsupported spacedim for tria_out."));
              break;
            case 3:
              if (spacedim_out == 1)
                internal::flatten_triangulation<1, 3, 1>(triangulation,
                                                         tria_out);
              else if (spacedim_out == 2)
                internal::flatten_triangulation<1, 3, 2>(triangulation,
                                                         tria_out);
              else if (spacedim_out == 3)
                internal::flatten_triangulation<1, 3, 3>(triangulation,
                                                         tria_out);
              else
                AssertThrow(false,
                            ExcMessage("Unsupported spacedim for tria_out."));
              break;
            default:
              AssertThrow(false, ExcMessage("Invalid spacedim"));
          }
        break;
        case 2: {
          switch (spacedim)
            {
              case 2:
                if (spacedim_out == 2)
                  internal::flatten_triangulation<2, 2, 2>(triangulation,
                                                           tria_out);
                else if (spacedim_out == 3)
                  internal::flatten_triangulation<2, 2, 3>(triangulation,
                                                           tria_out);
                else
                  AssertThrow(false,
                              ExcMessage("Invalid spacedim for tria_out"));
                break;
              case 3:
                if (spacedim_out == 2)
                  internal::flatten_triangulation<2, 3, 2>(triangulation,
                                                           tria_out);
                else if (spacedim_out == 3)
                  internal::flatten_triangulation<2, 3, 3>(triangulation,
                                                           tria_out);
                else
                  AssertThrow(false,
                              ExcMessage("Invalid spacedim for tria_out"));
                break;
              default:
                AssertThrow(false, ExcMessage("Invalid spacedim"));
            }
        }
        break;
        case 3: {
          AssertThrow(spacedim == 3, ExcMessage("Invalid spacedim"));
          AssertThrow(spacedim_out == 3,
                      ExcMessage("Invalid spacedim for tria_out"));
          internal::flatten_triangulation<3, 3, 3>(triangulation, tria_out);
          break;
        }
      default:
        AssertThrow(false, ExcMessage("Unsupported dimension."));
    }
}



void
TriangulationWrapper::extrude_triangulation(
  const unsigned int    n_slices,
  const double          height,
  TriangulationWrapper &triangulation_out)
{
  AssertThrow((dim == 2) && (spacedim == 2),
              ExcMessage(
                "Extrude can only be applied to the dim and spacedim two."));
  AssertThrow((triangulation_out.get_dim() == 3) &&
                (triangulation_out.get_spacedim() == 3),
              ExcMessage(
                "The output Triangulation must be of dimension three"));


  Triangulation<2, 2> *tria = static_cast<Triangulation<2, 2> *>(triangulation);
  Triangulation<3, 3> *tria_out =
    static_cast<Triangulation<3, 3> *>(triangulation_out.get_triangulation());
  GridGenerator::extrude_triangulation(*tria, n_slices, height, *tria_out);
}



void
TriangulationWrapper::distort_random(const double factor,
                                     const bool   keep_boundary)
{
  internal::distort_random_wrapper<3, 3>(
    factor, keep_boundary, triangulation, dim, spacedim);
}



void
TriangulationWrapper::transform(PYSPACE::object &transformation)
{
  internal::transform_wrapper<3, 3>(transformation,
                                    triangulation,
                                    dim,
                                    spacedim);
}



CellAccessorWrapper
TriangulationWrapper::find_active_cell_around_point(
  PointWrapper          &p,
  MappingQGenericWrapper mapping)
{
  std::pair<int, int> level_index_pair =
    internal::find_active_cell_around_point_wrapper<3, 3>(
      p, mapping, triangulation, dim, spacedim);

  return CellAccessorWrapper(*this,
                             level_index_pair.first,
                             level_index_pair.second);
}


#ifdef USE_PYBIND11
CellAccessorWrapper
TriangulationWrapper::find_active_cell_around_point_wrapper(
  PointWrapper                         &p,
  std::optional<MappingQGenericWrapper> mapping)
{
  if (mapping.has_value())
    return this->find_active_cell_around_point(p, mapping.value());
  else
    return this->find_active_cell_around_point(p, MappingQGenericWrapper());
}
#endif



PYSPACE::list
TriangulationWrapper::find_cells_adjacent_to_vertex(
  const unsigned int vertex_index)
{
  return internal::find_cells_adjacent_to_vertex_wrapper<3, 3>(vertex_index,
                                                               *this,
                                                               dim,
                                                               spacedim);
}



PYSPACE::list
TriangulationWrapper::compute_aspect_ratio_of_cells(
  const MappingQGenericWrapper &mapping,
  const QuadratureWrapper      &quadrature)
{
  if ((dim == 1) && (spacedim == 1))
    return internal::compute_aspect_ratio_of_cells<1, 1>(mapping,
                                                         quadrature,
                                                         *this);
  else if ((dim == 2) && (spacedim == 2))
    return internal::compute_aspect_ratio_of_cells<2, 2>(mapping,
                                                         quadrature,
                                                         *this);
  else if ((dim == 3) && (spacedim == 3))
    return internal::compute_aspect_ratio_of_cells<3, 3>(mapping,
                                                         quadrature,
                                                         *this);
  else
    AssertThrow(
      false, ExcMessage("Thie combination of dim-spacedim is not supported."));
}



void
TriangulationWrapper::refine_global(const unsigned int n)
{
  if ((dim == 1) && (spacedim == 1))
    {
      Triangulation<1, 1> *tria =
        static_cast<Triangulation<1, 1> *>(triangulation);
      tria->refine_global(n);
    }
  else if ((dim == 1) && (spacedim == 2))
    {
      Triangulation<1, 2> *tria =
        static_cast<Triangulation<1, 2> *>(triangulation);
      tria->refine_global(n);
    }
  else if ((dim == 1) && (spacedim == 3))
    {
      Triangulation<1, 3> *tria =
        static_cast<Triangulation<1, 3> *>(triangulation);
      tria->refine_global(n);
    }
  else if ((dim == 2) && (spacedim == 2))
    {
      Triangulation<2, 2> *tria =
        static_cast<Triangulation<2, 2> *>(triangulation);
      tria->refine_global(n);
    }
  else if ((dim == 2) && (spacedim == 3))
    {
      Triangulation<2, 3> *tria =
        static_cast<Triangulation<2, 3> *>(triangulation);
      tria->refine_global(n);
    }
  else
    {
      Triangulation<3, 3> *tria =
        static_cast<Triangulation<3, 3> *>(triangulation);
      tria->refine_global(n);
    }
}



void
TriangulationWrapper::execute_coarsening_and_refinement()
{
  if ((dim == 1) && (spacedim == 1))
    {
      Triangulation<1, 1> *tria =
        static_cast<Triangulation<1, 1> *>(triangulation);
      tria->execute_coarsening_and_refinement();
    }
  else if ((dim == 1) && (spacedim == 2))
    {
      Triangulation<1, 2> *tria =
        static_cast<Triangulation<1, 2> *>(triangulation);
      tria->execute_coarsening_and_refinement();
    }
  else if ((dim == 1) && (spacedim == 3))
    {
      Triangulation<1, 3> *tria =
        static_cast<Triangulation<1, 3> *>(triangulation);
      tria->execute_coarsening_and_refinement();
    }
  else if ((dim == 2) && (spacedim == 2))
    {
      Triangulation<2, 2> *tria =
        static_cast<Triangulation<2, 2> *>(triangulation);
      tria->execute_coarsening_and_refinement();
    }
  else if ((dim == 2) && (spacedim == 3))
    {
      Triangulation<2, 3> *tria =
        static_cast<Triangulation<2, 3> *>(triangulation);
      tria->execute_coarsening_and_refinement();
    }
  else
    {
      Triangulation<3, 3> *tria =
        static_cast<Triangulation<3, 3> *>(triangulation);
      tria->execute_coarsening_and_refinement();
    }
}



double
TriangulationWrapper::minimal_cell_diameter() const
{
  return internal::minimal_cell_diameter_wrapper<3, 3>(triangulation,
                                                       dim,
                                                       spacedim);
}



double
TriangulationWrapper::maximal_cell_diameter() const
{
  return internal::maximal_cell_diameter_wrapper<3, 3>(triangulation,
                                                       dim,
                                                       spacedim);
}



void
TriangulationWrapper::write(const std::string &filename,
                            const std::string  format) const
{
  internal::write_wrapper<3, 3>(filename, format, triangulation, dim, spacedim);
}



void
TriangulationWrapper::read(const std::string &filename,
                           const std::string  format) const
{
  internal::read_wrapper<3, 3>(filename, format, triangulation, dim, spacedim);
}



void
TriangulationWrapper::save(const std::string &filename) const
{
  std::ofstream                   ofs(filename);
  boost::archive::binary_oarchive oa(ofs);

  if ((dim == 1) && (spacedim == 1))
    {
      Triangulation<1, 1> *tria =
        static_cast<Triangulation<1, 1> *>(triangulation);

      tria->save(oa, 0);
    }
  else if ((dim == 1) && (spacedim == 2))
    {
      Triangulation<1, 2> *tria =
        static_cast<Triangulation<1, 2> *>(triangulation);

      tria->save(oa, 0);
    }
  else if ((dim == 1) && (spacedim == 3))
    {
      Triangulation<1, 3> *tria =
        static_cast<Triangulation<1, 3> *>(triangulation);

      tria->save(oa, 0);
    }
  else if ((dim == 2) && (spacedim == 2))
    {
      Triangulation<2, 2> *tria =
        static_cast<Triangulation<2, 2> *>(triangulation);

      tria->save(oa, 0);
    }
  else if ((dim == 2) && (spacedim == 3))
    {
      {
        Triangulation<2, 3> *tria =
          static_cast<Triangulation<2, 3> *>(triangulation);

        tria->save(oa, 0);
      }
    }
  else if ((dim == 3) && (spacedim == 3))
    {
      Triangulation<3, 3> *tria =
        static_cast<Triangulation<3, 3> *>(triangulation);

      tria->save(oa, 0);
    }
  else
    AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
}



void
TriangulationWrapper::load(const std::string &filename)
{
  std::ifstream                   ifs(filename);
  boost::archive::binary_iarchive ia(ifs);

  if ((dim == 1) && (spacedim == 1))
    {
      Triangulation<1, 1> *tria =
        static_cast<Triangulation<1, 1> *>(triangulation);

      tria->load(ia, 0);
    }
  else if ((dim == 1) && (spacedim == 2))
    {
      Triangulation<1, 2> *tria =
        static_cast<Triangulation<1, 2> *>(triangulation);

      tria->load(ia, 0);
    }
  else if ((dim == 1) && (spacedim == 3))
    {
      Triangulation<1, 3> *tria =
        static_cast<Triangulation<1, 3> *>(triangulation);

      tria->load(ia, 0);
    }
  else if ((dim == 2) && (spacedim == 2))
    {
      Triangulation<2, 2> *tria =
        static_cast<Triangulation<2, 2> *>(triangulation);

      tria->load(ia, 0);
    }
  else if ((dim == 2) && (spacedim == 3))
    {
      Triangulation<2, 3> *tria =
        static_cast<Triangulation<2, 3> *>(triangulation);

      tria->load(ia, 0);
    }
  else if ((dim == 3) && (spacedim == 3))
    {
      Triangulation<3> *tria =
        static_cast<Triangulation<3, 3> *>(triangulation);

      tria->load(ia, 0);
    }
  else
    AssertThrow(false, ExcMessage("Wrong dim-spacedim combination."));
}



PYSPACE::list
TriangulationWrapper::active_cells()
{
  return internal::active_cells_wrapper<3, 3>(*this, dim, spacedim);
}



void
TriangulationWrapper::set_manifold(const int number, ManifoldWrapper &manifold)
{
  AssertThrow(
    dim == manifold.get_dim(),
    ExcMessage(
      "The Triangulation and Manifold should have the same dimension."));
  AssertThrow(
    spacedim == manifold.get_spacedim(),
    ExcMessage(
      "The Triangulation and Manifold should have the same space dimension."));

  if ((dim == 1) && (spacedim == 1))
    {
      Triangulation<1, 1> *tria =
        static_cast<Triangulation<1, 1> *>(triangulation);
      Manifold<1, 1> *m =
        static_cast<Manifold<1, 1> *>(manifold.get_manifold());
      tria->set_manifold(number, *m);
    }
  else if ((dim == 1) && (spacedim == 2))
    {
      Triangulation<1, 2> *tria =
        static_cast<Triangulation<1, 2> *>(triangulation);
      Manifold<1, 2> *m =
        static_cast<Manifold<1, 2> *>(manifold.get_manifold());
      tria->set_manifold(number, *m);
    }
  else if ((dim == 1) && (spacedim == 3))
    {
      Triangulation<1, 3> *tria =
        static_cast<Triangulation<1, 3> *>(triangulation);
      Manifold<1, 3> *m =
        static_cast<Manifold<1, 3> *>(manifold.get_manifold());
      tria->set_manifold(number, *m);
    }
  else if ((dim == 2) && (spacedim == 2))
    {
      Triangulation<2, 2> *tria =
        static_cast<Triangulation<2, 2> *>(triangulation);
      Manifold<2, 2> *m =
        static_cast<Manifold<2, 2> *>(manifold.get_manifold());
      tria->set_manifold(number, *m);
    }
  else if ((dim == 2) && (spacedim == 3))
    {
      Triangulation<2, 3> *tria =
        static_cast<Triangulation<2, 3> *>(triangulation);
      Manifold<2, 3> *m =
        static_cast<Manifold<2, 3> *>(manifold.get_manifold());
      tria->set_manifold(number, *m);
    }
  else
    {
      Triangulation<3, 3> *tria =
        static_cast<Triangulation<3, 3> *>(triangulation);
      Manifold<3, 3> *m =
        static_cast<Manifold<3, 3> *>(manifold.get_manifold());
      tria->set_manifold(number, *m);
    }
}



void
TriangulationWrapper::reset_manifold(const int number)
{
  if ((dim == 1) && (spacedim == 1))
    {
      Triangulation<1, 1> *tria =
        static_cast<Triangulation<1, 1> *>(triangulation);
      tria->reset_manifold(number);
    }
  else if ((dim == 1) && (spacedim == 2))
    {
      Triangulation<1, 2> *tria =
        static_cast<Triangulation<1, 2> *>(triangulation);
      tria->reset_manifold(number);
    }
  else if ((dim == 1) && (spacedim == 3))
    {
      Triangulation<1, 3> *tria =
        static_cast<Triangulation<1, 3> *>(triangulation);
      tria->reset_manifold(number);
    }
  else if ((dim == 2) && (spacedim == 2))
    {
      Triangulation<2, 2> *tria =
        static_cast<Triangulation<2, 2> *>(triangulation);
      tria->reset_manifold(number);
    }
  else if ((dim == 2) && (spacedim == 3))
    {
      Triangulation<2, 3> *tria =
        static_cast<Triangulation<2, 3> *>(triangulation);
      tria->reset_manifold(number);
    }
  else
    {
      Triangulation<3, 3> *tria =
        static_cast<Triangulation<3, 3> *>(triangulation);
      tria->reset_manifold(number);
    }
}



void
TriangulationWrapper::setup(const std::string &dimension,
                            const std::string &spacedimension)
{
  if ((dimension.compare("1D") == 0) || (dimension.compare("1d") == 0))
    {
      dim = 1;

      if ((spacedimension.compare("1D") == 0) ||
          (spacedimension.compare("1d") == 0))
        {
          spacedim      = 1;
          triangulation = new Triangulation<1, 1>();
        }
      else if ((spacedimension.compare("2D") == 0) ||
               (spacedimension.compare("2d") == 0))
        {
          spacedim      = 2;
          triangulation = new Triangulation<1, 2>();
        }
      else if ((spacedimension.compare("3D") == 0) ||
               (spacedimension.compare("3d") == 0))
        {
          spacedim      = 3;
          triangulation = new Triangulation<1, 3>();
        }
      else
        AssertThrow(false,
                    ExcMessage("Spacedimension needs to be 1D, 2D or 3D."));
    }
  else if ((dimension.compare("2D") == 0) || (dimension.compare("2d") == 0))
    {
      dim = 2;

      if ((spacedimension.compare("2D") == 0) ||
          (spacedimension.compare("2d") == 0))
        {
          spacedim      = 2;
          triangulation = new Triangulation<2, 2>();
        }
      else if ((spacedimension.compare("3D") == 0) ||
               (spacedimension.compare("3d") == 0))
        {
          spacedim      = 3;
          triangulation = new Triangulation<2, 3>();
        }
      else
        AssertThrow(false, ExcMessage("Spacedimension needs to be 2D or 3D."));
    }
  else if ((dimension.compare("3D") == 0) || (dimension.compare("3d") == 0))
    {
      if ((spacedimension.compare("3D") != 0) &&
          (spacedimension.compare("3d") != 0))
        AssertThrow(false, ExcMessage("Spacedimension needs to be 3D."));
      dim           = 3;
      spacedim      = 3;
      triangulation = new Triangulation<3, 3>();
    }
  else
    AssertThrow(false, ExcMessage("Dimension needs to be 2D or 3D."));
}
} // namespace python

DEAL_II_NAMESPACE_CLOSE
