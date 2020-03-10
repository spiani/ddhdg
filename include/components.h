#pragma once

namespace Ddhdg
{
  DeclExceptionMsg(UnknownComponent, "Invalid component specified");
  DeclExceptionMsg(UnknownDisplacement, "Invalid displacement specified");

  enum Component
  {
    V = 0,
    n = 1,
    p = 2,
  };

  // This is useful for iterating over the previous enumerator
  inline std::set<Component>
  all_components()
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
        default:
          Assert(false, UnknownComponent());
          break;
      }
    return "UnknownComponent";
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
        default:
          Assert(false, UnknownComponent());
          break;
      }
    return "UnknownComponent";
  }

  inline unsigned int
  get_component_number(const Component c)
  {
    unsigned int i = 0;
    for (const Component current_cmp : all_components())
      {
        if (current_cmp == c)
          return i;
        i++;
      }
    Assert(false, UnknownComponent());
    return 9999;
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
          Assert(false, UnknownDisplacement());
          break;
      }
    return "UnknownDisplacement";
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
          Assert(false, UnknownDisplacement());
          break;
      }
    return "UnknownDisplacement";
  }

  inline Displacement
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
          Assert(false, UnknownComponent());
          break;
      }
    return E;
  }

  inline Component
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
          Assert(false, UnknownDisplacement());
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
