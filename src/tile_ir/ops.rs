//! Compute ops (exactly the StructInfo transfer set):
//! `Matmul`, `MatmulBlock`, `Materialize`, `Permute`, `Concat`, `Broadcast`,
//! `Slice`, `Reshape`.
//! Structural: `Constant`, `Return`, `Yield`.

use crate::syntax::shapes::{self, NatExpr, ShapeExpr};
use crate::typecheck::constraints::{Constraint, Phi};
use crate::typecheck::normalization::{normalize_shape, shape_equal};

use crate::tile_ir::attributes::Attribute;
use crate::tile_ir::region::Region;
use crate::tile_ir::types::{IrType, MemorySpace, Storage, TileType};
use crate::tile_ir::value::{Value, ValueId};

#[derive(Clone, Debug, PartialEq)]
pub enum OpKind {
    /// A compile-time constant (scalar/bool/shape), value in attributes.
    Constant,
    Matmul,
    MatmulBlock,
    Materialize,
    Permute,
    Concat,
    Broadcast,
    Slice,
    Reshape,
    /// Function terminator.
    Return,
    /// Region terminator (kept for nested regions / future use).
    Yield,
}

impl OpKind {
    pub fn mnemonic(&self) -> String {
        match self {
            OpKind::Constant => "tile.constant".into(),
            OpKind::Matmul => "tile.matmul".into(),
            OpKind::MatmulBlock => "tile.matmul_block".into(),
            OpKind::Materialize => "tile.materialize".into(),
            OpKind::Permute => "tile.permute".into(),
            OpKind::Concat => "tile.concat".into(),
            OpKind::Broadcast => "tile.broadcast".into(),
            OpKind::Slice => "tile.slice".into(),
            OpKind::Reshape => "tile.reshape".into(),
            OpKind::Return => "tile.return".into(),
            OpKind::Yield => "tile.yield".into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    Pure,
    Read,
    Write,
}

pub fn effect_of(kind: &OpKind) -> Effect {
    match kind {
        OpKind::Materialize => Effect::Write,
        _ => Effect::Pure,
    }
}


#[derive(Clone, Debug, PartialEq)]
pub struct Operation {
    pub kind: OpKind,
    pub operands: Vec<ValueId>,
    pub results: Vec<Value>,
    pub attributes: Vec<Attribute>,
    pub regions: Vec<Region>,
}

impl Operation {
    pub fn result_types(&self) -> Vec<IrType> {
        self.results.iter().map(|v| v.ty.clone()).collect()
    }

    pub fn shape_attr(&self) -> Option<&ShapeExpr> {
        self.attributes.iter().find_map(|a| match a {
            Attribute::Shape(s) => Some(s),
            _ => None,
        })
    }
    pub fn axis_attr(&self) -> Option<u64> {
        self.attributes.iter().find_map(|a| match a {
            Attribute::Axis(x) => Some(*x),
            _ => None,
        })
    }
    pub fn perm_attr(&self) -> Option<&Vec<u64>> {
        self.attributes.iter().find_map(|a| match a {
            Attribute::Permutation(p) => Some(p),
            _ => None,
        })
    }
    pub fn slice_attr(&self) -> Option<&crate::tile_ir::attributes::SliceSpec> {
        self.attributes.iter().find_map(|a| match a {
            Attribute::Slice(s) => Some(s),
            _ => None,
        })
    }
    pub fn space_attr(&self) -> Option<MemorySpace> {
        self.attributes.iter().find_map(|a| match a {
            Attribute::MemorySpace(s) => Some(*s),
            _ => None,
        })
    }
    pub fn tile_attr(&self) -> Option<&Vec<u64>> {
        self.attributes.iter().find_map(|a| match a {
            Attribute::Tile(t) => Some(t),
            _ => None,
        })
    }
}


// ---------------------------------------------------------------------------
// Result-type inference & op validity.
// ---------------------------------------------------------------------------

fn as_tile(t: &IrType, what: &str) -> Result<TileType, String> {
    match t {
        IrType::Tile(tt) => Ok(tt.clone()),
        other => Err(format!("{what}: expected a tile, got `{other}`")),
    }
}

/// Concrete-rank dimension list, if the shape has a statically known rank.
pub fn concrete_dims(s: &ShapeExpr) -> Option<Vec<NatExpr>> {
    let rebuilt = normalize_shape(s).to_shape();
    let mut out = Vec::new();
    let mut cur = rebuilt;
    loop {
        match cur {
            ShapeExpr::Nil => return Some(out),
            ShapeExpr::Cons(n, rest) => {
                out.push(n);
                cur = *rest;
            }
            _ => return None,
        }
    }
}

fn infer_matmul(
    operands: &[IrType],
    phi: &Phi,
    what: &str,
) -> Result<Vec<IrType>, String> {
    if operands.len() != 2 {
        return Err(format!("{what}: expected 2 operands"));
    }
    let a = as_tile(&operands[0], &format!("{what} A"))?;
    let b = as_tile(&operands[1], &format!("{what} B"))?;
    if a.dtype != b.dtype {
        return Err(format!("{what}: operand dtypes differ"));
    }
    let da = concrete_dims(&a.shape)
        .ok_or_else(|| format!("{what}: A has unknown rank"))?;
    let db = concrete_dims(&b.shape)
        .ok_or_else(|| format!("{what}: B has unknown rank"))?;
    if da.len() != 2 || db.len() != 2 {
        return Err(format!("{what}: both operands must be rank-2"));
    }
    let (m, k1) = (da[0].clone(), da[1].clone());
    let (k2, n) = (db[0].clone(), db[1].clone());
    if !phi.entails(&Constraint::NatEq(k1.clone(), k2.clone())) {
        return Err(format!(
            "{what}: cannot prove inner dims equal ({k1} = {k2})"
        ));
    }
    Ok(vec![IrType::tile(a.dtype, ShapeExpr::lit([m, n]))])
}

/// Infer the result types of an operation, validating operand/attribute
/// consistency along the way. `phi` from crate/typecheck/constraints.rs
pub fn infer_results(
    kind: &OpKind,
    operands: &[IrType],
    attrs: &[Attribute],
    _regions: &[Region],
    phi: &Phi,
) -> Result<Vec<IrType>, String> {
    match kind {
        OpKind::Constant => {
            let attr = attrs
                .first()
                .ok_or_else(|| "constant: missing value attribute".to_string())?;
            let ty = match attr {
                Attribute::Shape(s) => IrType::Shape(s.clone()),
                Attribute::Scalar(c) => IrType::Scalar(c.dtype()),
                Attribute::Bool(_) => IrType::Bool,
                other => return Err(format!("constant: unsupported attribute {other:?}")),
            };
            Ok(vec![ty])
        }

        OpKind::Matmul => infer_matmul(operands, phi, "matmul"),

        OpKind::MatmulBlock => {
            let tile = attrs
                .iter()
                .find_map(|a| match a {
                    Attribute::Tile(t) => Some(t.clone()),
                    _ => None,
                })
                .ok_or_else(|| "matmul_block: missing tile attribute".to_string())?;
            if tile.len() != 3 || tile.iter().any(|&x| x == 0) {
                return Err(format!(
                    "matmul_block: tile must be [BM,BN,BK] with positive sizes, got {tile:?}"
                ));
            }
            infer_matmul(operands, phi, "matmul_block")
        }

        OpKind::Materialize => {
            if operands.len() != 1 {
                return Err("materialize: expected 1 operand".into());
            }
            let t = as_tile(&operands[0], "materialize")?;
            let space = attrs
                .iter()
                .find_map(|a| match a {
                    Attribute::MemorySpace(s) => Some(*s),
                    _ => None,
                })
                .ok_or_else(|| "materialize: missing memory-space attribute".to_string())?;
            let layout = attrs
                .iter()
                .find_map(|a| match a {
                    Attribute::Layout(l) => Some(*l),
                    _ => None,
                })
                .unwrap_or(t.layout);
            Ok(vec![IrType::Tile(TileType {
                dtype: t.dtype,
                shape: t.shape,
                layout,
                storage: Storage::Materialized(space),
            })])
        }

        OpKind::Permute => {
            let t = as_tile(&operands[0], "permute")?;
            let p = attrs
                .iter()
                .find_map(|a| match a {
                    Attribute::Permutation(p) => Some(p.clone()),
                    _ => None,
                })
                .ok_or_else(|| "permute: missing permutation attribute".to_string())?;
            let dims = concrete_dims(&t.shape)
                .ok_or_else(|| format!("permute: shape {} has unknown rank", t.shape))?;
            if !is_perm(&p, dims.len()) {
                return Err(format!(
                    "permute: {p:?} is not a permutation of rank {}",
                    dims.len()
                ));
            }
            Ok(vec![IrType::Tile(TileType {
                dtype: t.dtype,
                shape: shapes::permute(t.shape, p),
                layout: t.layout,
                storage: t.storage,
            })])
        }

        OpKind::Concat => {
            if operands.len() != 2 {
                return Err("concat: expected 2 operands".into());
            }
            let a = as_tile(&operands[0], "concat A")?;
            let b = as_tile(&operands[1], "concat B")?;
            if a.dtype != b.dtype {
                return Err("concat: operand dtypes differ".into());
            }
            let axis = attrs
                .iter()
                .find_map(|at| match at {
                    Attribute::Axis(x) => Some(*x),
                    _ => None,
                })
                .ok_or_else(|| "concat: missing axis attribute".to_string())?;
            let da = concrete_dims(&a.shape)
                .ok_or_else(|| "concat: A has unknown rank".to_string())?;
            let db = concrete_dims(&b.shape)
                .ok_or_else(|| "concat: B has unknown rank".to_string())?;
            if da.len() != db.len() {
                return Err(format!(
                    "concat: rank mismatch ({} vs {})",
                    da.len(),
                    db.len()
                ));
            }
            if axis as usize >= da.len() {
                return Err(format!(
                    "concat: axis {axis} out of range for rank {}",
                    da.len()
                ));
            }
            for (i, (x, y)) in da.iter().zip(&db).enumerate() {
                if i == axis as usize {
                    continue;
                }
                if !phi.entails(&Constraint::NatEq(x.clone(), y.clone())) {
                    return Err(format!(
                        "concat: cannot prove non-concat dim {i} equal ({x} = {y})"
                    ));
                }
            }
            let mut out = da.clone();
            out[axis as usize] =
                NatExpr::add(da[axis as usize].clone(), db[axis as usize].clone());
            Ok(vec![IrType::Tile(TileType {
                dtype: a.dtype,
                shape: ShapeExpr::lit(out),
                layout: a.layout,
                storage: a.storage,
            })])
        }

        OpKind::Reshape => {
            let t = as_tile(&operands[0], "reshape")?;
            let s2 = shape_attr_of(attrs, "reshape")?;
            let eq =
                Constraint::NatEq(shapes::product(t.shape.clone()), shapes::product(s2.clone()));
            if !phi.entails(&eq) {
                return Err(format!(
                    "reshape: cannot prove product({}) = product({})",
                    t.shape, s2
                ));
            }
            Ok(vec![IrType::Tile(TileType {
                dtype: t.dtype,
                shape: s2,
                layout: t.layout,
                storage: t.storage,
            })])
        }

        OpKind::Broadcast => {
            let t = as_tile(&operands[0], "broadcast")?;
            let s2 = shape_attr_of(attrs, "broadcast")?;
            let bc = shapes::broadcast_shape(t.shape.clone(), s2.clone());
            if !shape_equal(&bc, &s2) {
                return Err(format!(
                    "broadcast: {} is not broadcastable to {}",
                    t.shape, s2
                ));
            }
            Ok(vec![IrType::Tile(TileType {
                dtype: t.dtype,
                shape: s2,
                layout: t.layout,
                storage: t.storage,
            })])
        }

        OpKind::Slice => {
            let t = as_tile(&operands[0], "slice")?;
            let spec = attrs
                .iter()
                .find_map(|a| match a {
                    Attribute::Slice(s) => Some(s.clone()),
                    _ => None,
                })
                .ok_or_else(|| "slice: missing slice attribute".to_string())?;
            let dims = concrete_dims(&t.shape)
                .ok_or_else(|| format!("slice: shape {} has unknown rank", t.shape))?;
            if spec.dims.len() != dims.len() {
                return Err(format!(
                    "slice: spec rank {} != tile rank {}",
                    spec.dims.len(),
                    dims.len()
                ));
            }
            Ok(vec![IrType::Tile(TileType {
                dtype: t.dtype,
                shape: spec.result_shape(),
                layout: t.layout,
                storage: t.storage,
            })])
        }

        OpKind::Return | OpKind::Yield => Ok(vec![]),
    }
}

fn shape_attr_of(attrs: &[Attribute], what: &str) -> Result<ShapeExpr, String> {
    attrs
        .iter()
        .find_map(|a| match a {
            Attribute::Shape(s) => Some(s.clone()),
            _ => None,
        })
        .ok_or_else(|| format!("{what}: missing shape attribute"))
}

fn is_perm(p: &[u64], len: usize) -> bool {
    if p.len() != len {
        return false;
    }
    let mut seen = vec![false; len];
    for &i in p {
        let i = i as usize;
        if i >= len || seen[i] {
            return false;
        }
        seen[i] = true;
    }
    true
}
