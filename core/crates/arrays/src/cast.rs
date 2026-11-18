//! Element conversion between data types, as NumPy's astype does for in-range values:
//! integers and floats through Rust's `as` (an out-of-range float saturates and NaN
//! becomes 0 where NumPy's result is platform-dependent), float16 through float64 (every
//! float16 is exact there), bool as 0 / 1 and nonzero -> true. Complex numbers only to
//! themselves.

use num_traits::AsPrimitive;

use crate::dtype::DType;
use crate::ReadError;

/// One element of the source type (native byte order) into one of the target type.
pub type Convert = Box<dyn Fn(&[u8], &mut [u8]) + Send + Sync>;

/// One element read as f64.
type ReadF64 = Box<dyn Fn(&[u8]) -> f64 + Send + Sync>;

/// An f64 written as one element.
type WriteF64 = Box<dyn Fn(f64, &mut [u8]) + Send + Sync>;

macro_rules! read_as_f64 {
    ($t:ty) => {
        |s: &[u8]| -> f64 { <$t>::from_ne_bytes(s.try_into().unwrap()) as f64 }
    };
}

/// A plain numeric type of the engine.
macro_rules! numeric {
    ($d:expr, $body:ident, $($args:tt)*) => {
        match $d {
            DType::Int8 => $body!(i8, $($args)*),
            DType::Int16 => $body!(i16, $($args)*),
            DType::Int32 => $body!(i32, $($args)*),
            DType::Int64 => $body!(i64, $($args)*),
            DType::UInt8 => $body!(u8, $($args)*),
            DType::UInt16 => $body!(u16, $($args)*),
            DType::UInt32 => $body!(u32, $($args)*),
            DType::UInt64 => $body!(u64, $($args)*),
            DType::Float32 => $body!(f32, $($args)*),
            DType::Float64 => $body!(f64, $($args)*),
            _ => None,
        }
    };
}

macro_rules! to_numeric {
    ($u:ty, $t:ty) => {
        Some(Box::new(|s: &[u8], d: &mut [u8]| {
            let v = <$t>::from_ne_bytes(s.try_into().unwrap());
            let w: $u = v.as_();
            d.copy_from_slice(&w.to_ne_bytes());
        }) as Convert)
    };
}

macro_rules! from_numeric {
    ($t:ty, $to:expr) => {
        numeric!($to, to_numeric, $t)
    };
}

macro_rules! f64_to {
    ($u:ty, ) => {
        Some(Box::new(|v: f64, d: &mut [u8]| {
            let w: $u = v.as_();
            d.copy_from_slice(&w.to_ne_bytes());
        }) as WriteF64)
    };
}

macro_rules! numeric_from_f64 {
    ($t:ty, ) => {
        Some(Box::new(read_as_f64!($t)) as ReadF64)
    };
}

/// Read any non-complex element as f64 (bool as 0 / 1, float16 exactly).
fn reader(from: DType) -> Option<ReadF64> {
    match from {
        DType::Bool => Some(Box::new(|s: &[u8]| if s[0] != 0 { 1.0 } else { 0.0 })),
        DType::Float16 => Some(Box::new(|s: &[u8]| {
            half::f16::from_ne_bytes(s.try_into().unwrap()).to_f64()
        })),
        other => numeric!(other, numeric_from_f64,),
    }
}

/// Write an f64 as any non-complex element (to bool: nonzero, NaN included -> true).
fn writer(to: DType) -> Option<WriteF64> {
    match to {
        DType::Bool => Some(Box::new(|v: f64, d: &mut [u8]| d[0] = u8::from(v != 0.0))),
        DType::Float16 => Some(Box::new(|v: f64, d: &mut [u8]| {
            d.copy_from_slice(&half::f16::from_f64(v).to_ne_bytes())
        })),
        other => numeric!(other, f64_to,),
    }
}

/// The conversion from `from` to `to`.
pub fn converter(from: DType, to: DType) -> Result<Convert, ReadError> {
    if from == to {
        return Ok(Box::new(|s: &[u8], d: &mut [u8]| d.copy_from_slice(s)));
    }
    if from.is_complex() || to.is_complex() {
        return Err(ReadError::new(format!(
            "converting {} to {} is not supported (complex numbers convert only to themselves)",
            from.name(),
            to.name()
        )));
    }
    // plain numeric to plain numeric: exact `as` between the two types (no f64 detour,
    // which would round 64-bit integers)
    let direct: Option<Convert> = numeric!(from, from_numeric, to);
    if let Some(f) = direct {
        return Ok(f);
    }
    // bool and float16 go through f64, which holds every value of both exactly
    let (r, w) = (reader(from), writer(to));
    match (r, w) {
        (Some(r), Some(w)) => Ok(Box::new(move |s: &[u8], d: &mut [u8]| w(r(s), d))),
        _ => Err(ReadError::new(format!(
            "converting {} to {} is not supported",
            from.name(),
            to.name()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv<T: Copy>(from: DType, to: DType, v: T, out_size: usize) -> Vec<u8> {
        let f = converter(from, to).unwrap();
        let s = unsafe {
            std::slice::from_raw_parts(&v as *const T as *const u8, std::mem::size_of::<T>())
        };
        let mut d = vec![0u8; out_size];
        f(s, &mut d);
        d
    }

    #[test]
    fn numeric_casts_as_rust_as() {
        assert_eq!(conv(DType::Float32, DType::UInt8, 3.7f32, 1), vec![3]);
        assert_eq!(
            conv(DType::Int16, DType::Float32, -5i16, 4),
            (-5.0f32).to_ne_bytes().to_vec()
        );
        assert_eq!(
            conv(DType::UInt64, DType::Int64, u64::MAX, 8),
            (-1i64).to_ne_bytes().to_vec()
        );
        // 64-bit integers do not go through f64
        let big = (1u64 << 60) + 1;
        assert_eq!(
            conv(DType::UInt64, DType::UInt64, big, 8),
            big.to_ne_bytes().to_vec()
        );
        assert_eq!(
            conv(DType::UInt64, DType::Int64, big, 8),
            (big as i64).to_ne_bytes().to_vec()
        );
    }

    #[test]
    fn bool_and_float16() {
        assert_eq!(
            conv(DType::Bool, DType::Float32, 1u8, 4),
            1.0f32.to_ne_bytes().to_vec()
        );
        assert_eq!(conv(DType::Float32, DType::Bool, 0.5f32, 1), vec![1]);
        assert_eq!(conv(DType::Float32, DType::Bool, 0.0f32, 1), vec![0]);
        let h = half::f16::from_f32(1.5);
        assert_eq!(
            conv(DType::Float16, DType::Float64, h, 8),
            1.5f64.to_ne_bytes().to_vec()
        );
        assert_eq!(
            conv(DType::UInt8, DType::Float16, 7u8, 2),
            half::f16::from_f32(7.0).to_ne_bytes().to_vec()
        );
    }

    #[test]
    fn complex_only_to_itself() {
        assert!(converter(DType::Complex64, DType::Float32).is_err());
        assert!(converter(DType::Complex64, DType::Complex64).is_ok());
    }
}
