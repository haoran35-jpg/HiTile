//! Kinds classify type-level expressions: they are "the type of a type".
//!
//! ```text
//! κ ::= Type | DType | Nat | Shape
//! ```
//!
//! A type-level variable always carries the kind it was bound at, e.g.
//! `M : Nat`, `S : Shape`, `d : DType`. Kind checking (see
//! `typecheck::kind_checker`) uses these to reject nonsense like
//! `Tile<16, f32>` where a `Nat` sits where a `DType` is expected.

use std::fmt;

/// The four kinds of HiTile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Ordinary types inhabited by runtime values (e.g. `Tile<f32,[M,N]>`).
    Type,
    /// Element/data types such as `f32`, `i32`, `bool`, `index`.
    DType,
    /// Natural-number type-level values (dimensions, ranks, ...).
    Nat,
    /// Shapes: sequences of `Nat`s.
    Shape,
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Kind::Type => "Type",
            Kind::DType => "DType",
            Kind::Nat => "Nat",
            Kind::Shape => "Shape",
        };
        f.write_str(s)
    }
}
