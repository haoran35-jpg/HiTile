#pragma once

#include "tile.hpp"

namespace four_stage {

// assign tile work to thread hierarchy (block/warp/thread).
// Specialize per (Arch, SrcMem, DstMem) later.
template <Mem SrcMem, Mem DstMem, Arch A, int R, int C>
struct assign_to_thread {
  static constexpr int block_threads = 128;
  static constexpr int warps_per_block = 4;
  static constexpr int warp_size = 32;
};

}  // namespace four_stage
