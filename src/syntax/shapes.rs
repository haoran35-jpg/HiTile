//! Type-level shape language.
//!
//! ```text
//! NatExpr   = n | α | n + n | n * n | ceilDiv(n,n)
//!           | product(S) | rank(S) | dim(S, i)
//! ShapeExpr = [] | n :: S | α | concat(S1, S2)
//!           | removeDim(S, a) | permute(S, p) | broadcastShape(S1, S2)
//! ```
//!
//! Only the *symbolic* structure lives here. The simplification rules
//! (`M + 0 → M`, `concat([],S) → S`, `product([M,N]) → M*N`, ...) live in
//! `typecheck::normalization`.

use std::fmt;

/// Natural-number-valued type-level expression (kind `Nat`).
#[derive(Clone, Debug, PartialEq)]
pub enum NatExpr {
    /// A concrete natural number literal.
    Const(u64),
    /// A `Nat`-kinded type variable, e.g. `M`.
    Var(String),
    /// `a + b`.
    Add(Box<NatExpr>, Box<NatExpr>),
    /// `a * b`.
    Mul(Box<NatExpr>, Box<NatExpr>),
    /// `ceilDiv(a, b) = ceil(a / b)`.
    CeilDiv(Box<NatExpr>, Box<NatExpr>),
    /// `product(S)` — the product of all dimensions of `S`.
    Product(Box<ShapeExpr>),
    /// `rank(S)` — the number of dimensions of `S`.
    Rank(Box<ShapeExpr>),
    /// `dim(S, i)` — the `i`-th dimension of `S`.
    Dim(Box<ShapeExpr>, u64),
}

/// Shape-valued type-level expression (kind `Shape`).
#[derive(Clone, Debug, PartialEq)]
pub enum ShapeExpr {
    /// The empty shape `[]`.
    Nil,
    /// Cons: `n :: S`.
    Cons(NatExpr, Box<ShapeExpr>),
    /// A `Shape`-kinded type variable, e.g. `S`.
    Var(String),
    /// `concat(S1, S2)`.
    Concat(Box<ShapeExpr>, Box<ShapeExpr>),
    /// `removeDim(S, axis)`.
    Remove(Box<ShapeExpr>, u64),
    /// `permute(S, p)`.
    Permute(Box<ShapeExpr>, Vec<u64>),
    /// `broadcastShape(S1, S2)`.
    Broadcast(Box<ShapeExpr>, Box<ShapeExpr>),
}

impl NatExpr {
    pub fn add(a: NatExpr, b: NatExpr) -> NatExpr {
        NatExpr::Add(Box::new(a), Box::new(b))
    }
    pub fn mul(a: NatExpr, b: NatExpr) -> NatExpr {
        NatExpr::Mul(Box::new(a), Box::new(b))
    }
    pub fn ceil_div(a: NatExpr, b: NatExpr) -> NatExpr {
        NatExpr::CeilDiv(Box::new(a), Box::new(b))
    }
    pub fn var(name: impl Into<String>) -> NatExpr {
        NatExpr::Var(name.into())
    }
}

impl ShapeExpr {
    /// Build a shape from a list of dimension expressions: `[n0, n1, ...]`.
    pub fn lit(dims: impl IntoIterator<Item = NatExpr>) -> ShapeExpr {
        let mut acc = ShapeExpr::Nil;
        let dims: Vec<NatExpr> = dims.into_iter().collect();
        for d in dims.into_iter().rev() {
            acc = ShapeExpr::Cons(d, Box::new(acc));
        }
        acc
    }
    pub fn var(name: impl Into<String>) -> ShapeExpr {
        ShapeExpr::Var(name.into())
    }
}

// ---------------------------------------------------------------------------
// Symbolic shape operations. These merely *construct* the corresponding
// symbolic node; `normalization` decides them when the operands are concrete.
// ---------------------------------------------------------------------------

/// `rank(S)`.
pub fn rank(s: ShapeExpr) -> NatExpr {
    NatExpr::Rank(Box::new(s))
}

/// `product(S)`.
pub fn product(s: ShapeExpr) -> NatExpr {
    NatExpr::Product(Box::new(s))
}

/// `removeDim(S, a)`.
pub fn remove_dim(s: ShapeExpr, a: u64) -> ShapeExpr {
    ShapeExpr::Remove(Box::new(s), a)
}

/// `permute(S, p)`.
pub fn permute(s: ShapeExpr, p: Vec<u64>) -> ShapeExpr {
    ShapeExpr::Permute(Box::new(s), p)
}

/// `broadcastShape(S1, S2)`.
pub fn broadcast_shape(a: ShapeExpr, b: ShapeExpr) -> ShapeExpr {
    ShapeExpr::Broadcast(Box::new(a), Box::new(b))
}

impl fmt::Display for NatExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NatExpr::Const(n) => write!(f, "{n}"),
            NatExpr::Var(x) => write!(f, "{x}"),
            NatExpr::Add(a, b) => write!(f, "({a} + {b})"),
            NatExpr::Mul(a, b) => write!(f, "({a} * {b})"),
            NatExpr::CeilDiv(a, b) => write!(f, "ceilDiv({a}, {b})"),
            NatExpr::Product(s) => write!(f, "product({s})"),
            NatExpr::Rank(s) => write!(f, "rank({s})"),
            NatExpr::Dim(s, i) => write!(f, "dim({s}, {i})"),
        }
    }
}

impl fmt::Display for ShapeExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeExpr::Var(x) => return write!(f, "{x}"),
            ShapeExpr::Concat(a, b) => return write!(f, "concat({a}, {b})"),
            ShapeExpr::Remove(s, a) => return write!(f, "removeDim({s}, {a})"),
            ShapeExpr::Permute(s, p) => return write!(f, "permute({s}, {p:?})"),
            ShapeExpr::Broadcast(a, b) => return write!(f, "broadcastShape({a}, {b})"),
            _ => {}
        }
        // Nil / Cons chain: render as a list if it is a proper list.
        let mut dims = Vec::new();
        let mut cur = self;
        loop {
            match cur {
                ShapeExpr::Nil => {
                    write!(f, "[")?;
                    for (i, d) in dims.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{d}")?;
                    }
                    return write!(f, "]");
                }
                ShapeExpr::Cons(n, rest) => {
                    dims.push(n.clone());
                    cur = rest;
                }
                other => {
                    // Improper tail (a variable etc.): fall back.
                    write!(f, "(")?;
                    for d in &dims {
                        write!(f, "{d} :: ")?;
                    }
                    return write!(f, "{other})");
                }
            }
        }
    }
}
