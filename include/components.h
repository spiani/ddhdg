#pragma once

namespace Ddhdg
{
  DeclExceptionMsg(UnknownComponent, "Invalid component specified");

  enum Component
  {
    V,
    n
  };

  // This is useful for iterating over the previous enumerator
  static const Component AllComponents[] = { V, n };
}