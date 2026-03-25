#pragma once

/**
 * @file cute_mapping.hpp
 * @brief four_stage::Mem / Arch / (R,C) → CuTe `Shape` / `Layout` rules and factories
 *
 * dependencies: CUTLASS headers ( `#include <cute/...>` ). Only including this header will not automatically pull CUDA; normal use in device code.
 *
 */

#include "tile.hpp"

#ifdef __CUDACC__
#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/pointer.hpp>
#include <cute/tensor.hpp>
#endif

#include <type_traits>

namespace four_stage {

// =============================================================================
// Mem / Arch — logical layer → CuTe "engine" semantics
// =============================================================================

/**
 * | four_stage | CuTe / convention |
 * |------------|----------------|
 * | Mem::Global | global memory tensor (gmem `Tensor`) |
 * | Mem::Shared | shared memory tensor (smem) |
 * | Mem::Reg    | register / thread-private (rmem) |
 *
 * | Arch        | description |
 * |-------------|-------------|
 * | Arch::Hopper | SM90+; later bound TMA / WGMMA etc. atoms in tile_algorithm |
 */

template <Mem M>
struct cute_mem_tag {};  ///< placeholder: subsequent `as_cute_tensor` can combine this tag to select Engine

template <>
struct cute_mem_tag<Mem::Global> {};
template <>
struct cute_mem_tag<Mem::Shared> {};
template <>
struct cute_mem_tag<Mem::Reg> {};

template <Arch A>
struct cute_arch_tag {};  ///< placeholder: subsequent binding of `TiledMMA` / TMA capabilities

template <>
struct cute_arch_tag<Arch::Hopper> {};

// =============================================================================
// (R, C) → cute::Shape — consistent with 2D tile<Mem, Arch, R, C>
// =============================================================================

#ifdef __CUDACC__

/**
 * static Shape for 2D tile: `(R, C)`, consistent with `tile<*, *, R, C>::rows/cols`.
 * CuTe dimension order convention is **(row, col)**, consistent with natural indexing of row-major layout.
 */
template <int R, int C>
using cute_tile_shape = decltype(cute::make_shape(cute::Int<R>{}, cute::Int<C>{}));

template <int R, int C>
constexpr cute_tile_shape<R, C> make_cute_tile_shape() noexcept {
  return cute::make_shape(cute::Int<R>{}, cute::Int<C>{});
}

// =============================================================================
// Layout factory — single data source: subsequent tiled_transfer / swizzle will be replaced or combined here
// =============================================================================

/** Row-major: row-continuous, `(r,c)` linear index = `r * C + c` (C is column width). */
template <int R, int C>
constexpr auto make_cute_layout_row_major() noexcept {
  return cute::make_layout(
      cute::make_shape(cute::Int<R>{}, cute::Int<C>{}),
      cute::make_stride(cute::Int<C>{}, cute::Int<1>{}));
}

/** Column-major: column-continuous, `(r,c)` linear index = `r + c * R`. */
template <int R, int C>
constexpr auto make_cute_layout_col_major() noexcept {
  return cute::make_layout(
      cute::make_shape(cute::Int<R>{}, cute::Int<C>{}),
      cute::make_stride(cute::Int<1>{}, cute::Int<R>{}));
}

/**
 * default layout strategy (stage 0.2 fixed convention, to avoid implicit ambiguity):
 * - Global: row-major (consistent with C row-major, majority of examples)
 * - Shared: row-major base layout; **swizzle / composed layout** in swizzle stage covered by `ComposedLayout`
 * - Reg: logically still use `(R,C)` Shape; physical register layout determined by MMA, here only providing Shape/stride reference consistent with tile
 */
template <Mem M, Arch A, int R, int C>
constexpr auto make_cute_tile_layout_default() noexcept {
  static_cast<void>(A);
  static_cast<void>(M);
  // Global / Shared / Reg: currently unified row-major base layout; Reg constrained by TiledMMA on MMA path
  return make_cute_layout_row_major<R, C>();
}

// =============================================================================
// convenient wrapper aligned with tile<> type alias
// =============================================================================

template <Mem M, Arch A, int R, int C>
struct cute_tile_traits {
  using four_stage_tile = tile<M, A, R, C>;
  using shape_type = cute_tile_shape<R, C>;

  static constexpr int shape_elems = R * C;

  static constexpr shape_type shape() noexcept { return make_cute_tile_shape<R, C>(); }

  static constexpr auto layout_default() noexcept {
    return make_cute_tile_layout_default<M, A, R, C>();
  }
};

// =============================================================================
// 0.3 tile -> CuTe tensor view (gmem/smem)
// =============================================================================
/**
 * as_cute_tensor:
 *   - Input: raw pointer + tile dims (R,C) via template params
 *   - Output: CuTe non-owning Tensor view, ready for `cute::copy`.
 *
 * Currently implemented for `Mem::Global` and `Mem::Shared` only.
 * `Mem::Reg` fragment tensor is expected to be introduced later via MMA/CuTe fragment logic.
 */
template <typename Element, Mem M, Arch A, int R, int C>
constexpr auto as_cute_tensor(Element* ptr) noexcept {
  static_assert(M != Mem::Reg, "as_cute_tensor for Mem::Reg is not implemented yet");
  using T = std::remove_const_t<Element>;
  auto layout = cute_tile_traits<M, A, R, C>::layout_default();

  if constexpr (M == Mem::Global) {
    return cute::make_tensor(cute::make_gmem_ptr<T>(ptr), layout);
  } else if constexpr (M == Mem::Shared) {
    return cute::make_tensor(cute::make_smem_ptr<T>(ptr), layout);
  }

  static_assert(M == Mem::Global || M == Mem::Shared, "unsupported Mem for as_cute_tensor");
}

template <typename Element, Mem M, Arch A, int R, int C>
constexpr auto as_cute_tensor(Element const* ptr) noexcept {
  return as_cute_tensor<std::remove_const_t<Element>, M, A, R, C>(
      const_cast<std::remove_const_t<Element>*>(ptr));
}

template <typename Element, Mem M, Arch A, int R, int C>
using cute_tensor_t = decltype(as_cute_tensor<Element, M, A, R, C>(static_cast<Element*>(nullptr)));

#else  // !__CUDACC__ : host-only stub

// Minimal host-only stubs so `#include <four_stage/cute_mapping.hpp>` works without CUDA.
template <int R, int C>
struct cute_tile_shape {
  static constexpr int rows = R;
  static constexpr int cols = C;
  static constexpr int size = R * C;
};

template <int R, int C>
constexpr cute_tile_shape<R, C> make_cute_tile_shape() noexcept {
  return {};
}

template <int R, int C>
struct cute_layout_dummy {};

template <int R, int C>
constexpr auto make_cute_layout_row_major() noexcept {
  return cute_layout_dummy<R, C>{};
}

template <int R, int C>
constexpr auto make_cute_layout_col_major() noexcept {
  return cute_layout_dummy<R, C>{};
}

template <Mem M, Arch A, int R, int C>
constexpr auto make_cute_tile_layout_default() noexcept {
  (void)M;
  (void)A;
  return cute_layout_dummy<R, C>{};
}

template <Mem M, Arch A, int R, int C>
struct cute_tile_traits {
  using four_stage_tile = tile<M, A, R, C>;
  using shape_type = cute_tile_shape<R, C>;

  static constexpr int shape_elems = R * C;

  static constexpr shape_type shape() noexcept { return make_cute_tile_shape<R, C>(); }

  static constexpr auto layout_default() noexcept {
    return make_cute_tile_layout_default<M, A, R, C>();
  }
};

struct cute_tensor_stub {};

template <typename Element, Mem M, Arch A, int R, int C>
constexpr cute_tensor_stub as_cute_tensor(Element*) noexcept {
  (void)sizeof(Element);
  (void)M;
  (void)A;
  (void)R;
  (void)C;
  static_assert(sizeof(Element) == 0, "as_cute_tensor requires CUDA compilation (__CUDACC__).");
  return {};
}

template <typename Element, Mem M, Arch A, int R, int C>
constexpr cute_tensor_stub as_cute_tensor(Element const* ptr) noexcept {
  return as_cute_tensor<Element, M, A, R, C>(const_cast<Element*>(ptr));
}

template <typename Element, Mem M, Arch A, int R, int C>
using cute_tensor_t = cute_tensor_stub;

#endif  // __CUDACC__

}  // namespace four_stage
