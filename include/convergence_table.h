#pragma once

#include <deal.II/base/parsed_convergence_table.h>

namespace Ddhdg
{
  class ConvergenceTable : public dealii::ParsedConvergenceTable
  {
  public:
    explicit ConvergenceTable()
      : dealii::ParsedConvergenceTable(
          {"E", "E", "V", "W", "W", "n"},
          {{},
           {dealii::VectorTools::H1_norm, dealii::VectorTools::L2_norm},
           {},
           {dealii::VectorTools::H1_norm, dealii::VectorTools::L2_norm}})
    {}
  };
}