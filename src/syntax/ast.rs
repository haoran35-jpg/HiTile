//! Expression AST.
//!
//! ```text
//! e ::= x | c
//!     | fun (x:τ) => e | e e
//!     | let x = e in e | if e then e else e
//!     | (e, e) | fst e | snd e
//!     | shape[n0,...] | rank e | dim e i | concatShape e e | shapeOf e
//!     | generate s (fun i => e)
//!     | map f x | zipWith f x y | broadcast x s
//!     | reshape x s | transpose x p | reduce f z x a | matmul a b
//!     | Λα:κ. e | e [arg]                 (type abstraction / application)
//!     | pack[∃α:κ.τ] arg with e | unpack (α, x) = e in e
//!     | prim(op, args...)                 (scalar builtins)
//!     | tile{d, shape, data}              (runtime tile literal / value)
//!     | idx[i0,...]                        (runtime multi-index value)
//! ```
//!
//! Note: a `Parser` is intended to eventually target this `Expr`; the tests
//! build `Expr` values directly.

use crate::syntax::kinds::Kind;
use crate::syntax::shapes::{NatExpr, ShapeExpr};
use crate::syntax::types::{DType, DTypeExpr, Ty};

/// A concrete scalar constant.
#[derive(Clone, Debug, PartialEq)]
pub enum ScalarConst {
    F32(f32),
    I32(i64),
    Bool(bool),
    Index(u64),
}

impl ScalarConst {
    pub fn dtype(&self) -> DType {
        match self {
            ScalarConst::F32(_) => DType::F32,
            ScalarConst::I32(_) => DType::I32,
            ScalarConst::Bool(_) => DType::Bool,
            ScalarConst::Index(_) => DType::Index,
        }
    }
}

/// Primitive (built-in) scalar operators that act directly on scalar values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

/// A type-level argument, used to instantiate `∀` and to introduce `∃`.
/// The variant must match the kind of the binder being instantiated.
#[derive(Clone, Debug, PartialEq)]
pub enum TyArg {
    Type(Ty),
    DType(DTypeExpr),
    Nat(NatExpr),
    Shape(ShapeExpr),
}

/// HiTile expressions.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    // --- core lambda calculus -------------------------------------------
    Var(String),
    UnitLit,
    BoolLit(bool),
    /// A scalar constant literal (`Scalar<d>`).
    ScalarLit(ScalarConst),
    /// A `Nat` runtime value (`NatVal<n>`).
    NatLit(u64),
    Fun(String, Ty, Box<Expr>),
    App(Box<Expr>, Box<Expr>),
    Let(String, Box<Expr>, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),

    // --- products -------------------------------------------------------
    Pair(Box<Expr>, Box<Expr>),
    Fst(Box<Expr>),
    Snd(Box<Expr>),

    // --- shapes ---------------------------------------------------------
    /// `shape[n0, ..., n_{k-1}]` where each entry is a `Nat` expression.
    ShapeLit(Vec<NatExpr>),
    Rank(Box<Expr>),
    /// `dim e i`.
    Dim(Box<Expr>, u64),
    ConcatShape(Box<Expr>, Box<Expr>),
    ShapeOf(Box<Expr>),

    // --- tile primitives ------------------------------------------------
    /// `generate s (fun i => body)`.
    Generate {
        shape: Box<Expr>,
        index: String,
        body: Box<Expr>,
    },
    Map(Box<Expr>, Box<Expr>),
    ZipWith(Box<Expr>, Box<Expr>, Box<Expr>),
    Broadcast(Box<Expr>, Box<Expr>),
    Reshape(Box<Expr>, Box<Expr>),
    Transpose(Box<Expr>, Vec<u64>),
    /// `slice x [(offset,size,step), ...]` — one triple per dimension.
    /// Result shape is `[size₀, size₁, ...]`.
    Slice {
        x: Box<Expr>,
        dims: Vec<(NatExpr, NatExpr, NatExpr)>,
    },
    /// `reduce f z x axis`.
    Reduce {
        f: Box<Expr>,
        z: Box<Expr>,
        x: Box<Expr>,
        axis: u64,
    },
    Matmul(Box<Expr>, Box<Expr>),

    // --- polymorphism / existentials ------------------------------------
    /// Type abstraction `Λα:κ. e` (introduces `∀`).
    TyAbs(String, Kind, Box<Expr>),
    /// Type application `e [arg]` (eliminates `∀`).
    TyApp(Box<Expr>, TyArg),
    /// `pack[∃α:κ.τ] witness with e`.
    Pack {
        witness: TyArg,
        body: Box<Expr>,
        ty: Ty, // the existential type being introduced
    },
    /// `unpack (α, x) = e1 in e2`.
    Unpack {
        tyvar: String,
        kind: Kind,
        valvar: String,
        packed: Box<Expr>,
        body: Box<Expr>,
    },

    // --- scalar builtins ------------------------------------------------
    Prim(PrimOp, Vec<Expr>),

    // --- runtime value forms --------------------------------------------
    /// A fully-evaluated tile value. Invariant: `data.len() == product(shape)`.
    TileLit {
        dtype: DType,
        shape: Vec<u64>,
        data: Vec<ScalarConst>,
    },
    /// A multi-index value of type `Index`.
    IndexLit(Vec<u64>),
}

impl Expr {
    pub fn app(f: Expr, x: Expr) -> Expr {
        Expr::App(Box::new(f), Box::new(x))
    }
    pub fn fun(x: impl Into<String>, t: Ty, body: Expr) -> Expr {
        Expr::Fun(x.into(), t, Box::new(body))
    }
    pub fn let_(x: impl Into<String>, e1: Expr, e2: Expr) -> Expr {
        Expr::Let(x.into(), Box::new(e1), Box::new(e2))
    }
    pub fn if_(c: Expr, t: Expr, e: Expr) -> Expr {
        Expr::If(Box::new(c), Box::new(t), Box::new(e))
    }
    pub fn pair(a: Expr, b: Expr) -> Expr {
        Expr::Pair(Box::new(a), Box::new(b))
    }
    pub fn fst(e: Expr) -> Expr {
        Expr::Fst(Box::new(e))
    }
    pub fn snd(e: Expr) -> Expr {
        Expr::Snd(Box::new(e))
    }
    pub fn shape_of(e: Expr) -> Expr {
        Expr::ShapeOf(Box::new(e))
    }
    pub fn generate(shape: Expr, index: impl Into<String>, body: Expr) -> Expr {
        Expr::Generate {
            shape: Box::new(shape),
            index: index.into(),
            body: Box::new(body),
        }
    }
    pub fn map(f: Expr, x: Expr) -> Expr {
        Expr::Map(Box::new(f), Box::new(x))
    }
    pub fn zip_with(f: Expr, x: Expr, y: Expr) -> Expr {
        Expr::ZipWith(Box::new(f), Box::new(x), Box::new(y))
    }
    pub fn broadcast(x: Expr, s: Expr) -> Expr {
        Expr::Broadcast(Box::new(x), Box::new(s))
    }
    pub fn reshape(x: Expr, s: Expr) -> Expr {
        Expr::Reshape(Box::new(x), Box::new(s))
    }
    pub fn transpose(x: Expr, p: Vec<u64>) -> Expr {
        Expr::Transpose(Box::new(x), p)
    }
    /// `slice(x, [(off,size,step), ...])`.
    pub fn slice(x: Expr, dims: Vec<(NatExpr, NatExpr, NatExpr)>) -> Expr {
        Expr::Slice {
            x: Box::new(x),
            dims,
        }
    }
    pub fn reduce(f: Expr, z: Expr, x: Expr, axis: u64) -> Expr {
        Expr::Reduce {
            f: Box::new(f),
            z: Box::new(z),
            x: Box::new(x),
            axis,
        }
    }
    pub fn matmul(a: Expr, b: Expr) -> Expr {
        Expr::Matmul(Box::new(a), Box::new(b))
    }
    pub fn ty_abs(a: impl Into<String>, k: Kind, e: Expr) -> Expr {
        Expr::TyAbs(a.into(), k, Box::new(e))
    }
    pub fn ty_app(e: Expr, arg: TyArg) -> Expr {
        Expr::TyApp(Box::new(e), arg)
    }
    pub fn f32(x: f32) -> Expr {
        Expr::ScalarLit(ScalarConst::F32(x))
    }
    pub fn i32(x: i64) -> Expr {
        Expr::ScalarLit(ScalarConst::I32(x))
    }

    /// Convenience: a binary scalar function `fun (p : Scalar<d>×Scalar<d>) => prim(op, fst p, snd p)`
    /// suitable to pass to `zipWith` / `reduce`.
    pub fn binop_fn(op: PrimOp, d: DType) -> Expr {
        let sd = Ty::Scalar(DTypeExpr::Const(d));
        Expr::fun(
            "__p",
            Ty::prod(sd.clone(), sd),
            Expr::Prim(
                op,
                vec![
                    Expr::fst(Expr::Var("__p".into())),
                    Expr::snd(Expr::Var("__p".into())),
                ],
            ),
        )
    }

    /// Whether this expression is a runtime *value* (fully evaluated).
    pub fn is_value(&self) -> bool {
        match self {
            Expr::UnitLit
            | Expr::BoolLit(_)
            | Expr::ScalarLit(_)
            | Expr::NatLit(_)
            | Expr::ShapeLit(_)
            | Expr::TileLit { .. }
            | Expr::IndexLit(_)
            | Expr::Fun(..)
            | Expr::TyAbs(..) => true,
            Expr::Pair(a, b) => a.is_value() && b.is_value(),
            Expr::Pack { body, .. } => body.is_value(),
            _ => false,
        }
    }
}
