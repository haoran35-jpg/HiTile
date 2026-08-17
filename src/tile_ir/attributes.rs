//! Static (compile-time) attributes attached to operations.
//! Attribute values attached to ops. Differ from SSA values and their type info changed by op.
//! Example:
//! x = reshape(a, shape=[4,8]),shape=[4,8] is attribute value, x is SSA value.
//! x : Tile<f32, [4,8]> here is type info changed by op reshape.
use std::fmt;

use crate::syntax::shapes::{NatExpr, ShapeExpr};
use crate::syntax::types::DType;
use crate::syntax::ast::ScalarConst;
use crate::tile_ir::types::{Layout, MemorySpace};

/// The well-known algebraic reductions (a generic combiner is expressed via a
/// region instead).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionKind {
    Add,
    Mul,
    Max,
    Min,
    Generic,
}

/// A per-dimension slice: `offset : size : step`.
#[derive(Clone, Debug, PartialEq)]
pub struct SliceDim {
    pub offset: NatExpr,
    pub size: NatExpr,
    pub step: NatExpr,
}

/// Slice bounds for every dimension of a tile.
#[derive(Clone, Debug, PartialEq)]
pub struct SliceSpec {
    pub dims: Vec<SliceDim>,
}

impl SliceSpec {
    pub fn result_shape(&self) -> ShapeExpr {
        ShapeExpr::lit(self.dims.iter().map(|d| d.size.clone()))
    }
}

/// A static attribute value.
#[derive(Clone, Debug, PartialEq)]
pub enum Attribute {
    Shape(ShapeExpr),
    Axis(u64),
    Permutation(Vec<u64>),
    Slice(SliceSpec),
    Layout(Layout),
    MemorySpace(MemorySpace),
    /// Block tile sizes, e.g. `[BM, BN, BK]` for `matmul_block`.
    Tile(Vec<u64>),
    Reduction(ReductionKind),
    Scalar(ScalarConst),
    Bool(bool),
    DType(DType),
}

impl fmt::Display for ReductionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ReductionKind::Add => "add",
            ReductionKind::Mul => "mul",
            ReductionKind::Max => "max",
            ReductionKind::Min => "min",
            ReductionKind::Generic => "generic",
        })
    }
}

impl fmt::Display for SliceDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.offset, self.size, self.step)
    }
}

impl fmt::Display for SliceSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

pub fn scalar_const_str(c: &ScalarConst) -> String {
    match c {
        ScalarConst::F32(x) => format!("f32 {x}"),
        ScalarConst::I32(x) => format!("i32 {x}"),
        ScalarConst::Bool(b) => format!("bool {b}"),
        ScalarConst::Index(n) => format!("index {n}"),
    }
}
