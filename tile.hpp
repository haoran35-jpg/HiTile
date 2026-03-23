#pragma once

/**
 * Single header: K-centric tile abstraction + (Src, Dst) specializations.
 *
 * 1) K forward pass (core): k_pipe_schedule / k_pipe_cursor — which K-tile to load, which ping-pong buffer to write.
 * 2) Tile shape: tile<Mem, Arch, R, C>.
 * 3) Different src/target: tile_algorithm<Src, Dst, Arch> main template + full specializations.
 * 4) Combination: tiled_transfer, tiled_transfer_k (K + memory layer transfer together).
 */

namespace four_stage {

// =============================================================================
// Memory / arch tags
// =============================================================================

enum struct Mem { Reg, Shared, Global };
enum struct Arch { Hopper };  // Ampere, etc. later

// =============================================================================
// K forward pass (tile abstraction core: step forward along K)
// =============================================================================

/**
 * At global step s:
 *   - which K-tile to load,
 *   - which ping-pong buffer to write,
 *   - which buffer / K-tile for compute (MMA), including warmup.
 */
template <int Stages>
struct k_pipe_schedule {
  static_assert(Stages >= 2, "pipeline needs at least 2 stages for overlap");

  static constexpr int load_buffer(int step) noexcept { return step % Stages; }
  static constexpr int k_tile_load(int step) noexcept { return step; }
  static constexpr int k_tile_compute(int step) noexcept {
    return step > 0 ? step - 1 : -1;
  }
  static constexpr int compute_buffer(int step) noexcept {
    return step > 0 ? (step - 1) % Stages : -1;
  }
  static constexpr bool has_load(int step, int num_k) noexcept {
    return step >= 0 && step < num_k;
  }
  static constexpr bool has_compute(int step) noexcept { return step > 0; }
  static constexpr int steps_for_k_chain(int num_k) noexcept {
    return num_k + (Stages - 1);
  }
};

template <int Stages>
struct k_pipe_cursor {
  static_assert(Stages >= 2);

  int step = 0;

  constexpr int load_buffer() const noexcept { return step % Stages; }
  constexpr int k_tile_load() const noexcept { return step; }
  constexpr int k_tile_compute() const noexcept {
    return step > 0 ? step - 1 : -1;
  }
  constexpr int compute_buffer() const noexcept {
    return step > 0 ? (step - 1) % Stages : -1;
  }
  constexpr void advance() noexcept { ++step; }
  constexpr void reset() noexcept { step = 0; }
};

// =============================================================================
// Tile shape (per memory level)
// =============================================================================

template <Mem M, Arch A, int R, int C>
struct tile {
  static constexpr Mem mem = M;
  static constexpr Arch arch = A;
  static constexpr int rows = R;
  static constexpr int cols = C;
};

template <int R, int C>
using reg_tile = tile<Mem::Reg, Arch::Hopper, R, C>;
template <int R, int C>
using shared_tile = tile<Mem::Shared, Arch::Hopper, R, C>;
template <int R, int C>
using global_tile = tile<Mem::Global, Arch::Hopper, R, C>;

// =============================================================================
// (Src -> Dst) transfer: abstract core + per-pair specializations
// =============================================================================

enum class TransferEngine : int {
  None = 0,
  TMA_Load,
  TMA_Store,
  SharedToReg,
  RegToShared,
  RegToReg,
  Staged,
  GlobalToGlobal,
  SharedToShared,
};

template <Mem Src, Mem Dst, Arch A>
struct tile_algorithm {
  static constexpr bool supported = false;
  static constexpr bool valid = false;
  static constexpr TransferEngine engine = TransferEngine::None;
  static constexpr int min_align_bytes = 0;
  static constexpr bool direct = false;
};

template <>
struct tile_algorithm<Mem::Global, Mem::Shared, Arch::Hopper> {
  static constexpr bool supported = true;
  static constexpr bool valid = true;
  static constexpr TransferEngine engine = TransferEngine::TMA_Load;
  static constexpr int min_align_bytes = 16;
  static constexpr bool direct = true;
};

template <>
struct tile_algorithm<Mem::Shared, Mem::Global, Arch::Hopper> {
  static constexpr bool supported = true;
  static constexpr bool valid = true;
  static constexpr TransferEngine engine = TransferEngine::TMA_Store;
  static constexpr int min_align_bytes = 16;
  static constexpr bool direct = true;
};

template <>
struct tile_algorithm<Mem::Shared, Mem::Reg, Arch::Hopper> {
  static constexpr bool supported = true;
  static constexpr bool valid = true;
  static constexpr TransferEngine engine = TransferEngine::SharedToReg;
  static constexpr int min_align_bytes = 16;
  static constexpr bool direct = true;
};

template <>
struct tile_algorithm<Mem::Reg, Mem::Shared, Arch::Hopper> {
  static constexpr bool supported = true;
  static constexpr bool valid = true;
  static constexpr TransferEngine engine = TransferEngine::RegToShared;
  static constexpr int min_align_bytes = 16;
  static constexpr bool direct = true;
};

template <>
struct tile_algorithm<Mem::Reg, Mem::Reg, Arch::Hopper> {
  static constexpr bool supported = true;
  static constexpr bool valid = true;
  static constexpr TransferEngine engine = TransferEngine::RegToReg;
  static constexpr int min_align_bytes = 4;
  static constexpr bool direct = true;
};

template <>
struct tile_algorithm<Mem::Global, Mem::Reg, Arch::Hopper> {
  static constexpr bool supported = true;
  static constexpr bool valid = true;
  static constexpr TransferEngine engine = TransferEngine::Staged;
  static constexpr int min_align_bytes = 16;
  static constexpr bool direct = false;
};

template <>
struct tile_algorithm<Mem::Reg, Mem::Global, Arch::Hopper> {
  static constexpr bool supported = true;
  static constexpr bool valid = true;
  static constexpr TransferEngine engine = TransferEngine::Staged;
  static constexpr int min_align_bytes = 16;
  static constexpr bool direct = false;
};

template <>
struct tile_algorithm<Mem::Global, Mem::Global, Arch::Hopper> {
  static constexpr bool supported = true;
  static constexpr bool valid = true;
  static constexpr TransferEngine engine = TransferEngine::GlobalToGlobal;
  static constexpr int min_align_bytes = 16;
  static constexpr bool direct = true;
};

template <>
struct tile_algorithm<Mem::Shared, Mem::Shared, Arch::Hopper> {
  static constexpr bool supported = true;
  static constexpr bool valid = true;
  static constexpr TransferEngine engine = TransferEngine::SharedToShared;
  static constexpr int min_align_bytes = 4;
  static constexpr bool direct = true;
};

template <Mem Src, Mem Dst, Arch A>
using tile_transfer_traits = tile_algorithm<Src, Dst, A>;

// =============================================================================
// Bundles
// =============================================================================

template <Mem Src, Mem Dst, Arch A, int R, int C>
struct tiled_transfer {
  using src_tile = tile<Src, A, R, C>;
  using dst_tile = tile<Dst, A, R, C>;
  using algorithm = tile_algorithm<Src, Dst, A>;
  using traits = algorithm;
};

/** K forward pass + (Src->Dst) transfer algorithm (GEMM main loop). */
template <Mem Src, Mem Dst, Arch A, int Stages, int R, int C>
struct tiled_transfer_k {
  static_assert(Stages >= 2);

  using transfer = tiled_transfer<Src, Dst, A, R, C>;
  using algorithm = typename transfer::algorithm;
  using src_tile = typename transfer::src_tile;
  using dst_tile = typename transfer::dst_tile;
  using k_schedule = k_pipe_schedule<Stages>;
  using k_cursor = k_pipe_cursor<Stages>;
};

template <Mem Src, Mem Dst, Arch A>
constexpr bool transfer_is_valid() {
  return tile_algorithm<Src, Dst, A>::supported;
}

template <Mem Src, Mem Dst, Arch A>
constexpr bool transfer_is_direct() {
  return tile_algorithm<Src, Dst, A>::direct;
}

template <Mem Src, Mem Dst, Arch A, typename F>
constexpr void tile_dispatch(F&& f) {
  if constexpr (tile_algorithm<Src, Dst, A>::supported) {
    f(tile_algorithm<Src, Dst, A>{});
  }
}

}  // namespace four_stage
