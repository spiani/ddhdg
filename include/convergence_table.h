#pragma once

#include <deal.II/base/parsed_convergence_table.h>

namespace Ddhdg {
    DeclExceptionMsg(InvalidDimension, "Only allowed dimensions are 1, 2 and 3.");

    class ConvergenceTable : public dealii::ParsedConvergenceTable {
    public:
        explicit ConvergenceTable(const unsigned int dim)
                : dealii::ParsedConvergenceTable(
                generate_components(dim),
                {{},
                 {dealii::VectorTools::H1_norm, dealii::VectorTools::L2_norm},
                 {},
                 {dealii::VectorTools::H1_norm, dealii::VectorTools::L2_norm}}) {}

    private:
        static std::vector<std::string> generate_components(const unsigned int dim) {
            switch (dim) {
                case 1:
                    return {"E", "V", "W", "n"};
                case 2:
                    return {"E", "E", "V", "W", "W", "n"};
                case 3:
                    return {"E", "E", "E", "V", "W", "W", "W", "n"};
                default:
                    AssertThrow(false, InvalidDimension());
            }
        }
    };
}
