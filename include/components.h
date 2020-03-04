#pragma once

namespace Ddhdg
{
  DeclExceptionMsg(UnknownComponent, "Invalid component specified");
  DeclExceptionMsg(UnknownDisplacement, "Invalid displacement specified");

  enum Component
  {
    V,
    n,
    p
  };

  // This is useful for iterating over the previous enumerator
  static const Component AllComponents[] = {V, n, p};

  enum Displacement
  {
    E,
    Wn,
    Wp
  };

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
} // namespace Ddhdg
