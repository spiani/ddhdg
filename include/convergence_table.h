#pragma once

#include <deal.II/base/parsed_convergence_table.h>

namespace Ddhdg
{
  DeclExceptionMsg(InvalidDimension, "Only allowed dimensions are 1, 2 and 3.");

  class ConvergenceTable : public dealii::ParsedConvergenceTable
  {
  public:
    explicit ConvergenceTable(const unsigned int dim)
      : dealii::ParsedConvergenceTable(
          generate_components(dim),
          {{},
           {dealii::VectorTools::H1_norm, dealii::VectorTools::L2_norm},
           {},
           {dealii::VectorTools::H1_norm, dealii::VectorTools::L2_norm},
           {},
           {dealii::VectorTools::H1_norm, dealii::VectorTools::L2_norm}})
    {}

    static std::vector<std::string>
    generate_components(const unsigned int dim)
    {
      switch (dim)
        {
          case 1:
            return {"E", "V", "Wn", "n", "Wp", "p"};
          case 2:
            return {"E", "E", "V", "Wn", "Wn", "n", "Wp", "Wp", "p"};
          case 3:
            return {
              "E", "E", "E", "V", "Wn", "Wn", "Wn", "n", "Wp", "Wp", "Wp", "p"};
          default:
            AssertThrow(false, InvalidDimension());
        }
    }
  };
} // namespace Ddhdg
