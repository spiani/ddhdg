#pragma once

#include <deal.II/base/function.h>

template<int dim>
class FunctionByComponents : public dealii::Function<dim> {
public:
    FunctionByComponents(int n_of_components,
            const std::map<unsigned int, const std::shared_ptr<const dealii::Function<dim>>> components)
            :
            dealii::Function<dim>(n_of_components)
            , component_map(components) { }

    double value(const dealii::Point<dim>& p, const unsigned int component = 0) const override
    {
        auto component_function = component_map.find(component);
        if (component_function!=component_map.end()) {
            return component_function->second->value(p, 0.);
        }
        else {
            return 0.;
        }
    }

    dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p, const unsigned int component = 0) const override
    {
        auto component_function = component_map.find(component);
        if (component_function!=component_map.end()) {
            return component_function->second->gradient(p, 0.);
        }
        else {
            dealii::Tensor<1, dim> zeros;
            for (int i = 0; i<dim; i++)
                zeros[i] = 0.;
            return zeros;
        }
    }

private:
    const std::map<unsigned int, const std::shared_ptr<const dealii::Function<dim>>> component_map;
};
