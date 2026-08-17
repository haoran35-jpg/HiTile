//! Semantic Tile IR types
//! A tile is a *logical* immutable value carrying static attributes:
//! Tile<dtype, shape, layout, storage>


use std::fmt;

use crate::syntax::shapes::ShapeExpr;
use crate::syntax::types::DType;
use crate::typecheck::normalization::{normalize_shape, shape_equal};

/// Physical layout interpretation of a tile's logical indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    RowMajor,
    ColMajor,
}

/// On-chip / off-chip memory space of a *materialized* tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemorySpace {
    Global,
    Shared,
    Register,
}

/// Where a tile lives. `Virtual` tiles are pure dataflow values that have not
/// been committed to any concrete storage yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    Virtual,
    Materialized(MemorySpace),
}

/// A tile type Tile<dtype, shape, layout, storage>
#[derive(Clone, Debug, PartialEq)]
pub struct TileType {
    pub dtype: DType,
    pub shape: ShapeExpr,
    pub layout: Layout,
    pub storage: Storage,
}

impl TileType {
    pub fn logical(dtype: DType, shape: ShapeExpr) -> TileType {
        TileType {
            dtype,
            shape,
            layout: Layout::RowMajor,
            storage: Storage::Virtual,
        }
    }
}

/// Semantic Tile IR types
#[derive(Clone, Debug, PartialEq)]
pub enum IrType {
    Unit,
    Bool,
    Index,
    Scalar(DType),
    Shape(ShapeExpr),
    Tile(TileType),
    Tuple(Vec<IrType>),
}

impl IrType {
    pub fn tile(dtype: DType, shape: ShapeExpr) -> IrType {
        IrType::Tile(TileType::logical(dtype, shape))
    }

    /// Semantic equivalence: shapes are compared up to normalization

    pub fn equiv(&self, other: &IrType) -> bool {
        match (self, other) {
            (IrType::Unit, IrType::Unit)
            | (IrType::Bool, IrType::Bool)
            | (IrType::Index, IrType::Index) => true,
            (IrType::Scalar(a), IrType::Scalar(b)) => a == b,
            (IrType::Shape(a), IrType::Shape(b)) => shape_equal(a, b),
            (IrType::Tile(a), IrType::Tile(b)) => {
                a.dtype == b.dtype
                    && a.layout == b.layout
                    && a.storage == b.storage
                    && shape_equal(&a.shape, &b.shape)
            }
            (IrType::Tuple(a), IrType::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.equiv(y))
            }
            _ => false,
        }
    }

    pub fn normalized(&self) -> IrType {
        match self {
            IrType::Shape(s) => IrType::Shape(normalize_shape(s).to_shape()),
            IrType::Tile(t) => IrType::Tile(TileType {
                dtype: t.dtype,
                shape: normalize_shape(&t.shape).to_shape(),
                layout: t.layout,
                storage: t.storage,
            }),
            IrType::Tuple(ts) => IrType::Tuple(ts.iter().map(|t| t.normalized()).collect()),
            other => other.clone(),
        }
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Layout::RowMajor => "row_major",
            Layout::ColMajor => "col_major",
        })
    }
}

impl fmt::Display for MemorySpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            MemorySpace::Global => "global",
            MemorySpace::Shared => "shared",
            MemorySpace::Register => "register",
        })
    }
}

impl fmt::Display for Storage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Storage::Virtual => f.write_str("virtual"),
            Storage::Materialized(m) => write!(f, "{m}"),
        }
    }
}

impl fmt::Display for IrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrType::Unit => f.write_str("unit"),
            IrType::Bool => f.write_str("bool"),
            IrType::Index => f.write_str("index"),
            IrType::Scalar(d) => write!(f, "scalar<{d}>"),
            IrType::Shape(s) => write!(f, "shape<{s}>"),
            IrType::Tile(t) => write!(
                f,
                "tile<{}, {}, {}, {}>",
                t.dtype, t.shape, t.layout, t.storage
            ),
            IrType::Tuple(ts) => {
                write!(f, "tuple<")?;
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{t}")?;
                }
                write!(f, ">")
            }
        }
    }
}
