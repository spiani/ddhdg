#pragma once

namespace Ddhdg
{
  DeclExceptionMsg(UnknownComponent, "Invalid component specified");
  DeclExceptionMsg(UnknownDisplacement, "Invalid displacement specified");

  enum Component
  {
    V,
    n
  };

  // This is useful for iterating over the previous enumerator
  static const Component AllComponents[] = {V, n};

  enum Displacement
  {
    E,
    W
  };

  inline Displacement
  component2displacement(Component c)
  {
    switch (c)
      {
        case V:
          return E;
        case n:
          return W;
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
        case W:
          return n;
        default:
          Assert(false, UnknownDisplacement());
          break;
      }
    return V;
  }
} // namespace Ddhdg
