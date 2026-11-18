//! The element types the engine reads: every fixed-size numeric type and bool, named as
//! NumPy names them (which are also Zarr v3's names).

use crate::ReadError;

/// An element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    Complex64,
    Complex128,
}

const ALL: [(DType, &str, usize); 14] = [
    (DType::Bool, "bool", 1),
    (DType::Int8, "int8", 1),
    (DType::Int16, "int16", 2),
    (DType::Int32, "int32", 4),
    (DType::Int64, "int64", 8),
    (DType::UInt8, "uint8", 1),
    (DType::UInt16, "uint16", 2),
    (DType::UInt32, "uint32", 4),
    (DType::UInt64, "uint64", 8),
    (DType::Float16, "float16", 2),
    (DType::Float32, "float32", 4),
    (DType::Float64, "float64", 8),
    (DType::Complex64, "complex64", 8),
    (DType::Complex128, "complex128", 16),
];

impl DType {
    /// The type named `name` (NumPy / Zarr v3 name).
    pub fn parse(name: &str) -> Result<Self, ReadError> {
        ALL.iter()
            .find(|(_, n, _)| *n == name)
            .map(|(t, _, _)| *t)
            .ok_or_else(|| {
                ReadError::new(format!(
                    "data type {name:?} is not one the engine reads (it reads {})",
                    ALL.iter()
                        .map(|(_, n, _)| *n)
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })
    }

    /// The NumPy / Zarr v3 name.
    pub fn name(self) -> &'static str {
        ALL.iter()
            .find(|(t, _, _)| *t == self)
            .map(|(_, n, _)| *n)
            .unwrap_or("?")
    }

    /// Bytes per element.
    pub fn size(self) -> usize {
        ALL.iter()
            .find(|(t, _, _)| *t == self)
            .map(|(_, _, s)| *s)
            .unwrap_or(0)
    }

    pub fn is_complex(self) -> bool {
        matches!(self, DType::Complex64 | DType::Complex128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_sizes() {
        for (t, n, s) in ALL {
            assert_eq!(DType::parse(n).unwrap(), t);
            assert_eq!(t.name(), n);
            assert_eq!(t.size(), s);
        }
        assert!(DType::parse("string").is_err());
    }
}
