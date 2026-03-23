#pragma once

#include "tile.hpp"

namespace four_stage {

// Stub: swizzle/layout for a tile
template <Mem M, Arch A>
struct swizzle {
  static constexpr bool use_swizzle = (M == Mem::Shared && A == Arch::Hopper);
  static constexpr int swizzle_bytes = 128;
};

}  // namespace four_stage
