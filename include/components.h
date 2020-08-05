#pragma once

#include "deal.II/base/exceptions.h"

#include <set>

namespace Ddhdg
{
  DeclExceptionMsg(InvalidComponent, "Invalid component specified");
  DeclExceptionMsg(InvalidDisplacement, "Invalid displacement specified");

  enum Component
  {
    V     = 0,
    n     = 1,
    p     = 2,
    phi_n = 3,
    phi_p = 4
  };

  // This is useful for iterating over the previous enumerator
  inline std::set<Component>
  all_primary_components()
  {
    std::set<Component> component_set{V, n, p};
    return component_set;
  }

  inline std::string
  get_component_name(const Component c)
  {
    switch (c)
      {
        case V:
          return "electric_potential";
        case n:
          return "electron_density";
        case p:
          return "hole_density";
        case phi_n:
          return "electron_quasi_fermi_potential";
        case phi_p:
          return "hole_quasi_fermi_potential";
        default:
          Assert(false, InvalidComponent());
          break;
      }
    return "InvalidComponent";
  }

  inline std::string
  get_component_short_name(const Component c)
  {
    switch (c)
      {
        case V:
          return "V";
        case n:
          return "n";
        case p:
          return "p";
        case phi_n:
          return "phi_n";
        case phi_p:
          return "phi_p";
        default:
          Assert(false, InvalidComponent());
          break;
      }
    return "InvalidComponent";
  }

  inline unsigned int
  get_component_index(const Component c, const std::set<Component> &components)
  {
    unsigned int i = 0;
    for (const Component current_cmp : components)
      {
        if (current_cmp == c)
          return i;
        i++;
      }
    Assert(false, InvalidComponent());
    return dealii::numbers::invalid_unsigned_int;
  }

  inline unsigned int
  get_component_index(const Component c)
  {
    return get_component_index(c, all_primary_components());
  }

  enum Displacement
  {
    E,
    Wn,
    Wp
  };

  inline std::string
  get_displacement_name(const Displacement d)
  {
    switch (d)
      {
        case E:
          return "electric_field";
        case Wn:
          return "electron_displacement";
        case Wp:
          return "hole_displacement";
        default:
          Assert(false, InvalidDisplacement());
          break;
      }
    return "InvalidDisplacement";
  }

  inline std::string
  get_displacement_short_name(const Displacement d)
  {
    switch (d)
      {
        case E:
          return "E";
        case Wn:
          return "Wn";
        case Wp:
          return "Wp";
        default:
          Assert(false, InvalidDisplacement());
          break;
      }
    return "InvalidDisplacement";
  }

  constexpr Displacement
  component2displacement(Component c)
  {
    switch (c)
      {
        case V:
          return E;
        case n:
          return Wn;
        case p:
          return Wp;
        default:
          Assert(false, InvalidComponent());
          break;
      }
    return E;
  }

  constexpr Component
  displacement2component(Displacement f)
  {
    switch (f)
      {
        case E:
          return V;
        case Wn:
          return n;
        case Wp:
          return p;
        default:
          Assert(false, InvalidDisplacement());
          break;
      }
    return V;
  }

  inline std::string
  get_displacement_name(const Component c)
  {
    return get_displacement_name(component2displacement(c));
  }

  inline std::string
  get_displacement_short_name(const Component c)
  {
    return get_displacement_short_name(component2displacement(c));
  }

} // namespace Ddhdg
