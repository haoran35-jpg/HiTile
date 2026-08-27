//! Abstract syntax of HiTile.
//!
//! ## Algebra layers (all pure / rewrite-safe)
//!
//! Stage-1 e-graph rewrites jointly over:
//!
//! | Layer | Ops | Role |
//! |-------|-----|------|
//! | **Loop algebra** | `map`, `reduce`, `split`, `join`, `zip` (+ `generate`) | iteration, tiling/blocking, fusion/fission, reduction |
//! | **Tile algebra** | `matmul`, `permute`, `reshape`, `slice`, `concat`, `broadcast`, `exp`/`add`/`mul`/`div`/`max` | math & shape transforms on tiles |
//!
//! Submodules: [`kinds`], [`shapes`], [`types`], [`ast`].

pub mod kinds;
pub mod shapes;
pub mod types;
pub mod ast;
