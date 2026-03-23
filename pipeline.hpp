#pragma once

#include "tile.hpp"

namespace four_stage {

// Pipeline depth + K schedule aliases
template <Arch A, int Stages = 2>
struct pipeline {
  static constexpr int stages = Stages;
  using k_schedule = k_pipe_schedule<Stages>;
  using k_cursor = k_pipe_cursor<Stages>;
};

}  // namespace four_stage
