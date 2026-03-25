#pragma once

#include "tile.hpp"
#include "cute_mapping.hpp"

namespace four_stage {

#ifdef __CUDACC__
#include <cute/algorithm/copy.hpp>
// AutoCopyAsync tag (CuTe chooses cp.async/universal copy based on Src/Dst engines + element sizes)
#include <cute/arch/copy.hpp>
#endif

// -----------------------------------------------------------------------------
// A1: first copy path (Global -> Shared, Hopper): use CuTe `cute::copy`.
// -----------------------------------------------------------------------------
/**
 * copy_tile:
 *   - Src/Dst are raw pointers (only at the API boundary)
 *   - Internally converts to CuTe Tensor views via `as_cute_tensor(...)`
 *   - Then performs `cute::copy` (for now using AutoCopyAsync; TMA-specific Copy_Atom comes in later)
 */
template <Mem SrcMem, Mem DstMem, Arch A, int R, int C, typename Element>
inline void copy_tile(Element const* src, Element* dst) {
#ifndef __CUDACC__
  (void)src;
  (void)dst;
  // Only error out when the template is instantiated (i.e., someone calls it on host).
  static_assert(sizeof(Element) == 0, "copy_tile is CUDA-only (requires __CUDACC__).");
#else
  static_assert(tile_algorithm<SrcMem, DstMem, A>::supported,
                "Unsupported (SrcMem,DstMem,Arch) transfer pair");
  static_assert(tile_algorithm<SrcMem, DstMem, A>::valid,
                "Invalid transfer pair (must be specialized in tile_algorithm)");

  auto src_tensor = as_cute_tensor<Element, SrcMem, A, R, C>(const_cast<Element*>(src));
  auto dst_tensor = as_cute_tensor<Element, DstMem, A, R, C>(dst);

  // First A1 landing path:
  //   - TMA_Load: Global -> Shared
  // For now we use AutoCopyAsync; later we will select explicit TMA/cp.async Copy_Atom
  // based on `tile_algorithm::engine` and architecture.
  if constexpr (tile_algorithm<SrcMem, DstMem, A>::engine == TransferEngine::TMA_Load) {
    cute::copy(cute::AutoCopyAsync{}, src_tensor, dst_tensor);
  } else {
    static_assert(tile_algorithm<SrcMem, DstMem, A>::engine == TransferEngine::TMA_Load,
                  "A1: only TransferEngine::TMA_Load is implemented in copy_tile (first path).");
  }
#endif
}

}  // namespace four_stage

