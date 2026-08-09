//! Ordinary types (kind `Type`) plus data types (kind `DType`).
//!
//! ```text
//! τ ::= Unit | Bool | Index
//!     | Scalar<d> | NatVal<n> | ShapeVal<S> | Tile<d,S>
//!     | τ × τ | τ → τ
//!     | ∀α:κ. τ | ∃α:κ. τ
//!     | α                 (a Type-kinded variable)
//! ```

use std::fmt;

use crate::syntax::kinds::Kind;
use crate::syntax::shapes::{NatExpr, ShapeExpr};

/// Concrete element/data types (values of kind `DType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    I32,
    Bool,
    /// Integer index element type (used for e.g. `Tile<index, S>`).
    Index,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DType::F32 => "f32",
            DType::I32 => "i32",
            DType::Bool => "bool",
            DType::Index => "index",
        };
        f.write_str(s)
    }
}

/// A `DType`-kinded expression: either a concrete dtype or a variable `d`.
#[derive(Clone, Debug, PartialEq)]
pub enum DTypeExpr {
    Const(DType),
    Var(String),
}

impl DTypeExpr {
    pub fn f32() -> Self {
        DTypeExpr::Const(DType::F32)
    }
    pub fn var(name: impl Into<String>) -> Self {
        DTypeExpr::Var(name.into())
    }
}

impl fmt::Display for DTypeExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DTypeExpr::Const(d) => write!(f, "{d}"),
            DTypeExpr::Var(x) => write!(f, "{x}"),
        }
    }
}

/// HiTile types (kind `Type`).
#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    Unit,
    Bool,
    /// A single multi-dimensional index into a tile.
    Index,
    /// `Scalar<d>`.
    Scalar(DTypeExpr),
    /// `NatVal<n>` — a runtime natural number carrying its value in the type.
    NatVal(NatExpr),
    /// `ShapeVal<S>` — a runtime shape carrying its value in the type.
    ShapeVal(ShapeExpr),
    /// `Tile<d, S>`.
    Tile(DTypeExpr, ShapeExpr),
    /// Product type `τ1 × τ2`.
    Prod(Box<Ty>, Box<Ty>),
    /// Function type `τ1 → τ2`.
    Arrow(Box<Ty>, Box<Ty>),
    /// Universal type `∀α:κ. τ`.
    Forall(String, Kind, Box<Ty>),
    /// Existential type `∃α:κ. τ`.
    Exists(String, Kind, Box<Ty>),
    /// A `Type`-kinded type variable `α`.
    Var(String),
}

impl Ty {
    pub fn arrow(a: Ty, b: Ty) -> Ty {
        Ty::Arrow(Box::new(a), Box::new(b))
    }
    pub fn prod(a: Ty, b: Ty) -> Ty {
        Ty::Prod(Box::new(a), Box::new(b))
    }
    pub fn forall(a: impl Into<String>, k: Kind, body: Ty) -> Ty {
        Ty::Forall(a.into(), k, Box::new(body))
    }
    pub fn exists(a: impl Into<String>, k: Kind, body: Ty) -> Ty {
        Ty::Exists(a.into(), k, Box::new(body))
    }
    pub fn tile(d: DTypeExpr, s: ShapeExpr) -> Ty {
        Ty::Tile(d, s)
    }
    pub fn scalar(d: DTypeExpr) -> Ty {
        Ty::Scalar(d)
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Unit => write!(f, "Unit"),
            Ty::Bool => write!(f, "Bool"),
            Ty::Index => write!(f, "Index"),
            Ty::Scalar(d) => write!(f, "Scalar<{d}>"),
            Ty::NatVal(n) => write!(f, "NatVal<{n}>"),
            Ty::ShapeVal(s) => write!(f, "ShapeVal<{s}>"),
            Ty::Tile(d, s) => write!(f, "Tile<{d}, {s}>"),
            Ty::Prod(a, b) => write!(f, "({a} × {b})"),
            Ty::Arrow(a, b) => write!(f, "({a} → {b})"),
            Ty::Forall(a, k, t) => write!(f, "∀{a}:{k}. {t}"),
            Ty::Exists(a, k, t) => write!(f, "∃{a}:{k}. {t}"),
            Ty::Var(x) => write!(f, "{x}"),
        }
    }
}
