//! NGFF coordinate transformations, as ozp's `spatial.Transformation` (plan 3.9, 8.20):
//! the type, the two ends, an optional name and the other JSON fields (`params`), kept
//! as read, so a transformation is written back as it came. The homogeneous matrix is
//! computed on demand ([`Transformation::matrix`]).
//!
//! Modelled: identity, translation, scale, affine and sequence (ozp's five), and NGFF
//! 0.6's rotation and mapAxis (WP15 DR2, decided 2026-11-18; ozp follows). Every other
//! type, parameters stored in a Zarr array (`"path"`), and an affine that changes the
//! number of axes are carried as read; asking for their matrix is a [`TransformError`].

use serde_json::{Map, Value};

use crate::matrix::Matrix;
use crate::TransformError;

/// The transformation types this crate models; any other is carried as read.
pub const MODELLED_TYPES: [&str; 7] = [
    "identity",
    "translation",
    "scale",
    "affine",
    "rotation",
    "mapAxis",
    "sequence",
];

/// A transformation read as a scale and a translation, per axis.
pub type ScaleTranslation = (Vec<f64>, Vec<f64>);

/// How far a rotation matrix may be from orthonormal with determinant 1.
const ROTATION_TOLERANCE: f64 = 1e-6;

/// One NGFF coordinate transformation (an edge of the transformation graph).
#[derive(Debug, Clone, PartialEq)]
pub struct Transformation {
    /// The NGFF type: one of [`MODELLED_TYPES`] or any other, carried as read.
    pub kind: String,
    /// The input end as read (`{"name": ...}`, `{"name": ..., "path": ...}`), or None.
    pub input: Option<Value>,
    /// The output end as read, or None.
    pub output: Option<Value>,
    pub name: Option<String>,
    /// The other JSON fields as read (`"scale"`, `"affine"`, `"path"`, ...), without a
    /// sequence's `"transformations"`, which are [`Self::steps`].
    pub params: Map<String, Value>,
    /// A sequence's steps, in the order they apply; empty for other types.
    pub steps: Vec<Transformation>,
}

impl Transformation {
    /// A transformation with no ends and no name.
    pub fn new(kind: &str, params: Map<String, Value>) -> Self {
        Self {
            kind: kind.to_string(),
            input: None,
            output: None,
            name: None,
            params,
            steps: Vec::new(),
        }
    }

    /// A sequence of `steps` (the first applies first).
    pub fn sequence(steps: Vec<Transformation>) -> Self {
        let mut t = Self::new("sequence", Map::new());
        t.steps = steps;
        t
    }

    fn with_vector(kind: &str, key: &str, v: &[f64]) -> Self {
        let mut params = Map::new();
        params.insert(key.to_string(), Value::from(v.to_vec()));
        Self::new(kind, params)
    }

    /// A scale transformation.
    pub fn scale(v: &[f64]) -> Self {
        Self::with_vector("scale", "scale", v)
    }

    /// A translation transformation.
    pub fn translation(v: &[f64]) -> Self {
        Self::with_vector("translation", "translation", v)
    }

    /// Whether this crate models it (a sequence only when all its steps are).
    pub fn modelled(&self) -> bool {
        if self.kind == "sequence" {
            return self.steps.iter().all(Transformation::modelled);
        }
        MODELLED_TYPES.contains(&self.kind.as_str())
    }

    fn label(&self) -> String {
        match &self.name {
            Some(n) => format!("{} {n:?}", self.kind),
            None => self.kind.clone(),
        }
    }

    fn param(&self, key: &str) -> Result<&Value, TransformError> {
        match self.params.get(key) {
            Some(v) => Ok(v),
            None if self.params.contains_key("path") => Err(TransformError::new(format!(
                "{}: its parameters are in the Zarr array {}; it is carried as read and \
                 not modelled",
                self.label(),
                self.params["path"]
            ))),
            None => Err(TransformError::new(format!(
                "{}: no {key:?} parameter",
                self.label()
            ))),
        }
    }

    fn vector(&self, key: &str, ndim: usize) -> Result<Vec<f64>, TransformError> {
        let v = flat_numbers(self.param(key)?).ok_or_else(|| {
            TransformError::new(format!("{}: {key:?} is not numbers", self.label()))
        })?;
        if v.len() != ndim {
            return Err(TransformError::new(format!(
                "{}: {} values for {ndim} axes",
                self.label(),
                v.len()
            )));
        }
        Ok(v)
    }

    /// The homogeneous (ndim + 1) x (ndim + 1) matrix over `ndim` axes (a sequence: its
    /// steps composed, the first applied first).
    pub fn matrix(&self, ndim: usize) -> Result<Matrix, TransformError> {
        let mut m = Matrix::identity(ndim);
        match self.kind.as_str() {
            "identity" => {}
            "translation" => {
                for (i, v) in self.vector("translation", ndim)?.into_iter().enumerate() {
                    m.set(i, ndim, v);
                }
            }
            "scale" => {
                for (i, v) in self.vector("scale", ndim)?.into_iter().enumerate() {
                    m.set(i, i, v);
                }
            }
            "affine" => {
                let value = self.param("affine")?;
                let rows = affine_rows(value, ndim)
                    .map_err(|why| TransformError::new(format!("{}: {why}", self.label())))?;
                m = Matrix::from_rows(&rows)?;
            }
            "rotation" => {
                let rows = number_rows(self.param("rotation")?).ok_or_else(|| {
                    TransformError::new(format!("{}: not a matrix of numbers", self.label()))
                })?;
                if rows.len() != ndim || rows.iter().any(|r| r.len() != ndim) {
                    return Err(TransformError::new(format!(
                        "{}: a rotation over {ndim} axes is {ndim} x {ndim}, not {} rows of {:?}",
                        self.label(),
                        rows.len(),
                        rows.iter().map(Vec::len).collect::<Vec<_>>()
                    )));
                }
                for (i, row) in rows.iter().enumerate() {
                    for (j, v) in row.iter().enumerate() {
                        m.set(i, j, *v);
                    }
                }
                check_rotation(&m).map_err(|why| {
                    TransformError::new(format!("{}: not a rotation ({why})", self.label()))
                })?;
            }
            "mapAxis" => {
                let order = axis_order(self.param("mapAxis")?, ndim)
                    .map_err(|why| TransformError::new(format!("{}: {why}", self.label())))?;
                for (i, &src) in order.iter().enumerate() {
                    for j in 0..ndim {
                        m.set(i, j, if j == src { 1.0 } else { 0.0 });
                    }
                }
            }
            "sequence" => {
                for step in &self.steps {
                    m = step.matrix(ndim)?.matmul(&m)?;
                }
            }
            other => {
                return Err(TransformError::new(format!(
                    "transformation type {other:?}: carried as read and not modelled \
                     (modelled: {})",
                    MODELLED_TYPES.join(", ")
                )))
            }
        }
        Ok(m)
    }

    /// The simplest transformation with matrix `m`, as ozp's `from_matrix`: identity,
    /// translation, scale, sequence[scale, translation] or affine (exact comparisons).
    pub fn from_matrix(m: &Matrix) -> Self {
        let offset = m.offset();
        let moved = offset.iter().any(|v| *v != 0.0);
        if m.is_diagonal() {
            let diagonal = m.diagonal();
            if diagonal.iter().all(|v| *v == 1.0) {
                return if moved {
                    Self::translation(&offset)
                } else {
                    Self::new("identity", Map::new())
                };
            }
            return if moved {
                Self::sequence(vec![Self::scale(&diagonal), Self::translation(&offset)])
            } else {
                Self::scale(&diagonal)
            };
        }
        let mut params = Map::new();
        let rows: Vec<Value> = m.rows().into_iter().map(Value::from).collect();
        params.insert("affine".to_string(), Value::Array(rows));
        Self::new("affine", params)
    }

    /// The inverse, from this transformation's output to its input (name kept). A
    /// singular one is an error.
    pub fn invert(&self, ndim: usize) -> Result<Self, TransformError> {
        let m = self
            .matrix(ndim)?
            .inverse()
            .map_err(|e| TransformError::new(format!("{}: {}", self.label(), e.message)))?;
        let mut t = Self::from_matrix(&m);
        t.input = self.output.clone();
        t.output = self.input.clone();
        t.name = self.name.clone();
        Ok(t)
    }

    /// The transformation as `sequence[scale s, translation t]`: `(s, t)`, or None when
    /// it turns, shears or swaps axes.
    pub fn scale_translation(
        &self,
        ndim: usize,
    ) -> Result<Option<ScaleTranslation>, TransformError> {
        let m = self.matrix(ndim)?;
        if !m.is_diagonal() {
            return Ok(None);
        }
        Ok(Some((m.diagonal(), m.offset())))
    }

    /// An NGFF 0.6 transformation object.
    pub fn from_json(obj: &Value) -> Result<Self, TransformError> {
        let map = obj.as_object().ok_or_else(|| {
            TransformError::new(format!("a transformation is a JSON object, not {obj}"))
        })?;
        let kind = map
            .get("type")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                TransformError::new(format!("a transformation without a type: {obj}"))
            })?;
        let mut params = Map::new();
        let mut steps = Vec::new();
        for (k, v) in map {
            match k.as_str() {
                "type" | "input" | "output" | "name" => {}
                "transformations" if kind == "sequence" => {
                    let items = v.as_array().ok_or_else(|| {
                        TransformError::new(format!("sequence: transformations is not a list: {v}"))
                    })?;
                    for item in items {
                        steps.push(Self::from_json(item)?);
                    }
                }
                _ => {
                    params.insert(k.clone(), v.clone());
                }
            }
        }
        Ok(Self {
            kind: kind.to_string(),
            input: map.get("input").filter(|v| !v.is_null()).cloned(),
            output: map.get("output").filter(|v| !v.is_null()).cloned(),
            name: map.get("name").and_then(Value::as_str).map(str::to_string),
            params,
            steps,
        })
    }

    /// The NGFF 0.6 transformation object: type, ends, name, the parameters as read and
    /// a sequence's steps.
    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("type".to_string(), Value::from(self.kind.clone()));
        if let Some(v) = &self.input {
            out.insert("input".to_string(), v.clone());
        }
        if let Some(v) = &self.output {
            out.insert("output".to_string(), v.clone());
        }
        if let Some(n) = &self.name {
            out.insert("name".to_string(), Value::from(n.clone()));
        }
        for (k, v) in &self.params {
            out.insert(k.clone(), v.clone());
        }
        if self.kind == "sequence" {
            out.insert(
                "transformations".to_string(),
                Value::Array(self.steps.iter().map(Self::to_json).collect()),
            );
        }
        Value::Object(out)
    }
}

/// The transformations applied one after another (the first first), as one: its
/// matrix simplified as [`Transformation::from_matrix`].
pub fn compose(ts: &[Transformation], ndim: usize) -> Result<Transformation, TransformError> {
    let mut m = Matrix::identity(ndim);
    for t in ts {
        m = t.matrix(ndim)?.matmul(&m)?;
    }
    Ok(Transformation::from_matrix(&m))
}

fn flat_numbers(v: &Value) -> Option<Vec<f64>> {
    match v {
        Value::Number(n) => n.as_f64().map(|x| vec![x]),
        Value::Array(items) => {
            let mut out = Vec::new();
            for item in items {
                out.extend(flat_numbers(item)?);
            }
            Some(out)
        }
        _ => None,
    }
}

fn number_rows(v: &Value) -> Option<Vec<Vec<f64>>> {
    v.as_array()?
        .iter()
        .map(|row| row.as_array()?.iter().map(Value::as_f64).collect())
        .collect()
}

/// An affine's rows over `ndim` axes, accepted as ozp accepts them: flat n(n+1)
/// values, n rows of n + 1, or (n + 1) x (n + 1) (its last row is ignored).
fn affine_rows(v: &Value, ndim: usize) -> Result<Vec<Vec<f64>>, String> {
    let rows = match number_rows(v) {
        Some(rows) => rows,
        None => {
            let flat = flat_numbers(v).ok_or("the affine is not numbers")?;
            if flat.len() != ndim * (ndim + 1) {
                return Err(format!(
                    "{} values; an affine over {ndim} axes has {}",
                    flat.len(),
                    ndim * (ndim + 1)
                ));
            }
            flat.chunks(ndim + 1).map(<[f64]>::to_vec).collect()
        }
    };
    let rows = if rows.len() == ndim + 1 && rows.iter().all(|r| r.len() == ndim + 1) {
        rows[..ndim].to_vec()
    } else {
        rows
    };
    if rows.len() != ndim || rows.iter().any(|r| r.len() != ndim + 1) {
        return Err(format!(
            "an affine of {} rows of {:?}: this crate models an affine that keeps the \
             number of axes ({ndim} x {}); one that changes it is carried as read",
            rows.len(),
            rows.iter().map(Vec::len).collect::<Vec<_>>(),
            ndim + 1
        ));
    }
    Ok(rows)
}

/// mapAxis: output axis i is input axis order[i]; a permutation of 0 .. ndim.
fn axis_order(v: &Value, ndim: usize) -> Result<Vec<usize>, String> {
    let items = v
        .as_array()
        .ok_or("mapAxis is not a list of axis indices")?;
    let order: Vec<usize> = items
        .iter()
        .map(|x| x.as_u64().map(|u| u as usize))
        .collect::<Option<_>>()
        .ok_or("mapAxis is not a list of axis indices")?;
    if order.len() != ndim {
        return Err(format!("{} axis indices for {ndim} axes", order.len()));
    }
    let mut seen = vec![false; ndim];
    for &i in &order {
        if i >= ndim || seen[i] {
            return Err(format!(
                "{order:?} is not a permutation of 0 .. {}",
                ndim - 1
            ));
        }
        seen[i] = true;
    }
    Ok(order)
}

/// A rotation: orthonormal rows and determinant 1, within ROTATION_TOLERANCE.
fn check_rotation(m: &Matrix) -> Result<(), String> {
    let n = m.ndim();
    for i in 0..n {
        for j in 0..n {
            let dot: f64 = (0..n).map(|k| m.get(i, k) * m.get(j, k)).sum();
            let want = if i == j { 1.0 } else { 0.0 };
            if (dot - want).abs() > ROTATION_TOLERANCE {
                return Err(format!("rows {i} and {j} are not orthonormal"));
            }
        }
    }
    let det = m.det();
    if (det - 1.0).abs() > ROTATION_TOLERANCE {
        return Err(format!("determinant {det}, not 1"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn t(v: Value) -> Transformation {
        Transformation::from_json(&v).unwrap()
    }

    #[test]
    fn a_sequence_applies_its_first_step_first() {
        let seq = t(json!({"type": "sequence", "transformations": [
            {"type": "scale", "scale": [2.0, 3.0]},
            {"type": "translation", "translation": [10.0, 20.0]}]}));
        assert_eq!(seq.matrix(2).unwrap().apply(&[1.0, 1.0]), vec![12.0, 23.0]);
    }

    #[test]
    fn affine_shapes_ozp_accepts() {
        let rows = json!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let flat = json!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let full = json!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 1.0]]);
        let want = t(json!({"type": "affine", "affine": rows}))
            .matrix(2)
            .unwrap();
        assert_eq!(
            t(json!({"type": "affine", "affine": flat}))
                .matrix(2)
                .unwrap(),
            want
        );
        assert_eq!(
            t(json!({"type": "affine", "affine": full}))
                .matrix(2)
                .unwrap(),
            want
        );
        // the spec's example: y = 1j + 2i + 3, x = 4j + 5i + 6 (input j, i)
        assert_eq!(want.apply(&[1.0, 0.0]), vec![4.0, 10.0]);
    }

    #[test]
    fn an_affine_that_changes_the_number_of_axes_is_carried_not_modelled() {
        let three_from_two = t(json!({"type": "affine",
            "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}));
        // 3 rows of 3 is 2 -> 3 axes: not a 2-axis or 3-axis affine this crate models
        assert!(three_from_two.matrix(3).is_err());
    }

    #[test]
    fn the_spec_rotation_example() {
        // x = 0i - 1j, y = 1i + 0j
        let r = t(json!({"type": "rotation", "rotation": [[0, -1], [1, 0]]}));
        assert_eq!(r.matrix(2).unwrap().apply(&[3.0, 4.0]), vec![-4.0, 3.0]);
    }

    #[test]
    fn a_rotation_must_be_one() {
        let shear = t(json!({"type": "rotation", "rotation": [[1, 1], [0, 1]]}));
        assert!(shear
            .matrix(2)
            .unwrap_err()
            .message
            .contains("not a rotation"));
        let mirror = t(json!({"type": "rotation", "rotation": [[1, 0], [0, -1]]}));
        assert!(mirror
            .matrix(2)
            .unwrap_err()
            .message
            .contains("determinant"));
        let wrong = t(json!({"type": "rotation", "rotation": [[1, 0], [0, 1]]}));
        assert!(wrong.matrix(3).is_err());
    }

    #[test]
    fn map_axis_follows_the_spec() {
        // [1, 0]: output axis 0 is input axis 1
        let swap = t(json!({"type": "mapAxis", "mapAxis": [1, 0]}));
        assert_eq!(swap.matrix(2).unwrap().apply(&[3.0, 4.0]), vec![4.0, 3.0]);
        let cycle = t(json!({"type": "mapAxis", "mapAxis": [2, 0, 1]}));
        assert_eq!(
            cycle.matrix(3).unwrap().apply(&[10.0, 20.0, 30.0]),
            vec![30.0, 10.0, 20.0]
        );
        assert_eq!(
            t(json!({"type": "mapAxis", "mapAxis": [0, 1]}))
                .matrix(2)
                .unwrap(),
            Matrix::identity(2)
        );
        for bad in [json!([0, 0]), json!([0, 2]), json!([0]), json!([0.5, 1])] {
            assert!(t(json!({"type": "mapAxis", "mapAxis": bad}))
                .matrix(2)
                .is_err());
        }
    }

    #[test]
    fn other_types_and_stored_parameters_are_carried_not_modelled() {
        let disp = t(json!({"type": "displacements", "path": "field", "interpolation": "linear"}));
        assert!(!disp.modelled());
        assert!(disp.matrix(2).unwrap_err().message.contains("not modelled"));
        let stored = t(json!({"type": "affine", "path": "matrix"}));
        assert!(stored.modelled());
        assert!(stored.matrix(2).unwrap_err().message.contains("Zarr array"));
        assert_eq!(
            disp.to_json(),
            json!({"type": "displacements", "path": "field",
                                          "interpolation": "linear"})
        );
    }

    #[test]
    fn from_matrix_gives_the_simplest_type() {
        let kinds = |rows: Vec<Vec<f64>>| {
            let tr = Transformation::from_matrix(&Matrix::from_rows(&rows).unwrap());
            (
                tr.kind.clone(),
                tr.steps.iter().map(|s| s.kind.clone()).collect::<Vec<_>>(),
            )
        };
        assert_eq!(
            kinds(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]).0,
            "identity"
        );
        assert_eq!(
            kinds(vec![vec![1.0, 0.0, 2.0], vec![0.0, 1.0, 0.0]]).0,
            "translation"
        );
        assert_eq!(
            kinds(vec![vec![2.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]).0,
            "scale"
        );
        assert_eq!(
            kinds(vec![vec![2.0, 0.0, 1.0], vec![0.0, 1.0, 0.0]]),
            (
                "sequence".to_string(),
                vec!["scale".to_string(), "translation".to_string()]
            )
        );
        assert_eq!(
            kinds(vec![vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 0.0]]).0,
            "affine"
        );
    }

    #[test]
    fn invert_swaps_the_ends_and_round_trips() {
        let a = t(
            json!({"type": "affine", "name": "reg", "input": {"name": "a"},
                         "output": {"name": "b"},
                         "affine": [[0.8, -0.6, 5.0], [0.6, 0.8, -3.0]]}),
        );
        let inv = a.invert(2).unwrap();
        assert_eq!(inv.input, Some(json!({"name": "b"})));
        assert_eq!(inv.output, Some(json!({"name": "a"})));
        assert_eq!(inv.name.as_deref(), Some("reg"));
        let p = a.matrix(2).unwrap().apply(&[7.0, -2.0]);
        let back = inv.matrix(2).unwrap().apply(&p);
        assert!((back[0] - 7.0).abs() < 1e-12 && (back[1] + 2.0).abs() < 1e-12);
        let flat = t(json!({"type": "scale", "scale": [0.0, 1.0]}));
        assert!(flat.invert(2).unwrap_err().message.contains("singular"));
    }

    #[test]
    fn scale_translation_reads_diagonal_forms_only() {
        let st = t(json!({"type": "sequence", "transformations": [
            {"type": "scale", "scale": [2.0, 3.0]},
            {"type": "translation", "translation": [1.0, -1.0]}]}));
        assert_eq!(
            st.scale_translation(2).unwrap(),
            Some((vec![2.0, 3.0], vec![1.0, -1.0]))
        );
        let swap = t(json!({"type": "mapAxis", "mapAxis": [1, 0]}));
        assert_eq!(swap.scale_translation(2).unwrap(), None);
    }

    #[test]
    fn compose_simplifies() {
        let c = compose(
            &[
                t(json!({"type": "scale", "scale": [2.0]})),
                t(json!({"type": "scale", "scale": [0.5]})),
            ],
            1,
        )
        .unwrap();
        assert_eq!(c.kind, "identity");
    }

    #[test]
    fn json_round_trips_as_read() {
        let obj = json!({"type": "sequence", "input": {"path": "0"}, "output": {"name": "physical"},
                         "transformations": [{"type": "scale", "scale": [1.5, 2.5]},
                                             {"type": "translation", "translation": [0, 4]}]});
        assert_eq!(t(obj.clone()).to_json(), obj);
    }
}
