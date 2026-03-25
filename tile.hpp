#pragma once

#include "memory.hpp"

namespace four_stage {

template <Mem M, Arch A, int R, int C>
struct tile {
  static constexpr Mem mem = M;
  static constexpr Arch arch = A;
  static constexpr int rows = R;
  static constexpr int cols = C;
};

// Convenience aliases (like TK's rt, st, gl)
template <int R, int C>
using reg_tile = tile<Mem::Reg, Arch::Hopper, R, C>;
template <int R, int C>
using shared_tile = tile<Mem::Shared, Arch::Hopper, R, C>;
template <int R, int C>
using global_tile = tile<Mem::Global, Arch::Hopper, R, C>;

}  // namespace four_stage
