//! scipy.ndimage's filters on in-memory blocks, giving scipy's results (Workplan WP15,
//! task DR7; decisions DR-3 a: our own kernels with scipy as the reference, and DR-6 a:
//! a crate of its own, unpublished for now). It knows nothing about Zarr, lazy arrays or
//! Python.
//!
//! A block is a C-ordered slice with its shape. Every separable filter runs as scipy
//! runs it: one 1-D pass per axis ([`correlate1d`]), each line extended at the edges by
//! a [`Mode`], summed in f64 in scipy's order, and written to the output type before the
//! next axis. Keeping scipy's order of operations is what makes the values equal, not
//! just close (checked against scipy 1.17.1's C source, `ni_filters.c` and
//! `ni_support.c`).
//!
//! A pass can work on part of an array: [`Span`] says which part of the axis the block
//! holds and which part to compute, so a region filtered from a padded box equals the
//! same region of the whole filtered array.

mod correlate;
mod gaussian;
mod label;
mod ndcorrelate;
mod pass;
mod rank;
mod running;
mod window;

pub use correlate::{correlate1d, correlate1d_on, pass_bytes, Span};
pub use gaussian::{gaussian_filter, gaussian_weights, numpy_sum, radius};
pub use label::{label, label_bytes};
pub use ndcorrelate::{correlate_on, Correlate};
pub use pass::{separable, terms, Combine, Line, Pass};
pub use rank::{rank_filter_on, RankFilter, Select};
pub use running::{min_max1d_on, running_pass_bytes, uniform1d_on};
pub use window::Footprint;

/// The crate's version, as `dyna_zarr_core` reports it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A filter's arguments or block did not fit; the message says which.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterError(pub String);

/// How the message of [`FilterError::from_line_start`] starts.
const FROM_LINE_START: &str = "scipy's value depends on where the line starts: ";

impl FilterError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// A pass asked for part of a line whose values, in scipy, depend on everything the
    /// line holds before that part: a sliding minimum or maximum through a NaN (scipy's
    /// loop keeps a NaN or drops it depending on the values before it). Only a run from
    /// the line's start gives scipy's values; a caller reading part of an array has to
    /// read it another way.
    pub fn line_start(what: &str) -> Self {
        Self(format!("{FROM_LINE_START}{what}"))
    }

    /// Whether this is [`FilterError::line_start`]'s error.
    pub fn from_line_start(&self) -> bool {
        self.0.starts_with(FROM_LINE_START)
    }
}

impl std::fmt::Display for FilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FilterError {}

/// How a line is extended beyond the array's edges: scipy.ndimage's five filter modes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    /// d c b a | a b c d | d c b a (scipy's default; also "grid-mirror")
    Reflect,
    /// d c b | a b c d | c b a (a line of one element: as Nearest, as scipy)
    Mirror,
    /// a a a | a b c d | d d d
    Nearest,
    /// a b c d | a b c d | a b c d (also "grid-wrap"; for filters scipy treats both alike)
    Wrap,
    /// k k k | a b c d | k k k, with k = cval (also "grid-constant")
    Constant(f64),
}

impl Mode {
    /// scipy's mode name (and its aliases for filters), with `cval` for constant.
    pub fn parse(name: &str, cval: f64) -> Option<Self> {
        Some(match name {
            "reflect" | "grid-mirror" => Self::Reflect,
            "mirror" => Self::Mirror,
            "nearest" => Self::Nearest,
            "wrap" | "grid-wrap" => Self::Wrap,
            "constant" | "grid-constant" => Self::Constant(cval),
            _ => return None,
        })
    }

    /// The index in `0 .. n` that position `p` of an axis of length `n` reads, or None
    /// for the constant fill. scipy's `NI_ExtendLine` copies values outwards; each of
    /// its loops is periodic (reflect: 2n, mirror: 2n - 2, wrap: n), including when the
    /// extension is longer than the line.
    pub fn index(self, p: i64, n: i64) -> Option<i64> {
        if (0..n).contains(&p) {
            return Some(p);
        }
        Some(match self {
            Self::Constant(_) => return None,
            Self::Nearest => p.clamp(0, n - 1),
            Self::Wrap => p.rem_euclid(n),
            Self::Reflect => {
                let period = 2 * n;
                let q = p.rem_euclid(period);
                if q < n {
                    q
                } else {
                    period - 1 - q
                }
            }
            Self::Mirror if n == 1 => 0,
            Self::Mirror => {
                let period = 2 * n - 2;
                let q = p.rem_euclid(period);
                if q < n {
                    q
                } else {
                    period - q
                }
            }
        })
    }

    /// The fill value (0 for every mode but constant).
    pub fn cval(self) -> f64 {
        match self {
            Self::Constant(c) => c,
            _ => 0.0,
        }
    }
}

/// An element type scipy.ndimage filters, with scipy's conversions to and from the f64
/// line buffers: `(double)value` in, `(type)value` out (C's casts).
pub trait Element: Copy + Send + Sync + 'static {
    /// numpy's name of the type.
    const NAME: &'static str;
    fn to_f64(self) -> f64;
    /// C's `(type)value` as x86-64 compilers emit it: floats round to nearest, integers
    /// truncate toward zero, and integers out of range wrap (a uint8 400.0 gives 144,
    /// -200.0 gives 56). Values beyond the conversion instruction's range give its
    /// "integer indefinite" value (the lowest integer), as on x86-64. ARM compilers
    /// saturate instead, so scipy's own results differ there for values out of range.
    fn from_f64(v: f64) -> Self;
    /// scipy's `CASE_FILTER_OUT_SAFE`, which its n-D filters use for unsigned outputs:
    /// `v > -1 ? (type)v : -(type)(-v)`. The same as [`Element::from_f64`] except for
    /// uint64 below -2^63 (signed and floating-point types: `from_f64`).
    #[inline(always)]
    fn from_f64_safe(v: f64) -> Self {
        Self::from_f64(v)
    }
    /// `self + other` as numpy adds in this type (integers wrap).
    fn add_wrapping(self, other: Self) -> Self;
    /// `self * other` as numpy multiplies in this type (integers wrap).
    fn mul_wrapping(self, other: Self) -> Self;
    /// The square root in this type, for floating-point types (None for integers).
    fn sqrt(self) -> Option<Self>;
}

/// x86's `cvttsd2si` to 32 bits: truncation, the lowest i32 outside its range or NaN.
fn cvtt32(v: f64) -> i32 {
    if v > -2_147_483_649.0 && v < 2_147_483_648.0 {
        v as i32
    } else {
        i32::MIN
    }
}

/// x86-64's `cvttsd2si` to 64 bits: truncation, the lowest i64 outside its range or NaN.
fn cvtt64(v: f64) -> i64 {
    if (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&v) {
        v as i64
    } else {
        i64::MIN
    }
}

macro_rules! element {
    ($($t:ty, $name:literal, $kind:ident, |$v:ident| $out:expr;)*) => {$(
        impl Element for $t {
            const NAME: &'static str = $name;
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn from_f64($v: f64) -> Self {
                $out
            }
            element!(@arith $kind $t);
        }
    )*};
    (@arith int $t:ty) => {
        #[inline(always)]
        fn add_wrapping(self, other: Self) -> Self {
            <$t>::wrapping_add(self, other)
        }
        #[inline(always)]
        fn mul_wrapping(self, other: Self) -> Self {
            <$t>::wrapping_mul(self, other)
        }
        fn sqrt(self) -> Option<Self> {
            None
        }
    };
    (@arith uint $t:ty) => {
        element!(@arith int $t);
        #[inline(always)]
        fn from_f64_safe(v: f64) -> Self {
            unsigned_safe::<$t>(v)
        }
    };
    (@arith float $t:ty) => {
        #[inline(always)]
        fn add_wrapping(self, other: Self) -> Self {
            self + other
        }
        #[inline(always)]
        fn mul_wrapping(self, other: Self) -> Self {
            self * other
        }
        fn sqrt(self) -> Option<Self> {
            Some(<$t>::sqrt(self))
        }
    };
}

element! {
    u8, "uint8", uint, |v| cvtt32(v) as u8;
    i8, "int8", int, |v| cvtt32(v) as i8;
    u16, "uint16", uint, |v| cvtt32(v) as u16;
    i16, "int16", int, |v| cvtt32(v) as i16;
    // 32-bit integers: unsigned through the 64-bit conversion, as compilers emit it
    u32, "uint32", uint, |v| cvtt64(v) as u32;
    i32, "int32", int, |v| cvtt32(v);
    // the usual unsigned 64-bit sequence: values from 2^63 up are shifted down first
    u64, "uint64", uint, |v| if v < 9_223_372_036_854_775_808.0 {
        cvtt64(v) as u64
    } else {
        (cvtt64(v - 9_223_372_036_854_775_808.0) as u64) ^ (1 << 63)
    };
    i64, "int64", int, |v| cvtt64(v);
    f32, "float32", float, |v| v as f32;
    f64, "float64", float, |v| v;
}

/// Unsigned types: scipy's safe cast for real (see [`Element::from_f64_safe`]).
pub(crate) trait Unsigned: Element {
    fn negate(self) -> Self;
}

macro_rules! unsigned_safe {
    ($($t:ty),*) => {$(
        impl Unsigned for $t {
            fn negate(self) -> Self {
                self.wrapping_neg()
            }
        }
    )*};
}

unsigned_safe!(u8, u16, u32, u64);

/// `v > -1 ? (type)v : -(type)(-v)` for an unsigned type.
#[inline(always)]
pub(crate) fn unsigned_safe<U: Unsigned>(v: f64) -> U {
    if v > -1.0 {
        U::from_f64(v)
    } else {
        U::from_f64(-v).negate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extended(mode: Mode, n: i64) -> Vec<Option<i64>> {
        (-6..n + 6).map(|p| mode.index(p, n)).collect()
    }

    #[test]
    fn modes_extend_as_scipy() {
        // scipy's NI_ExtendLine on a b c d (0 1 2 3), six positions each side
        let s = |v: &[i64]| v.iter().map(|&x| Some(x)).collect::<Vec<_>>();
        assert_eq!(
            extended(Mode::Reflect, 4),
            s(&[2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1])
        );
        assert_eq!(
            extended(Mode::Mirror, 4),
            s(&[0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3])
        );
        assert_eq!(
            extended(Mode::Nearest, 4),
            s(&[0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3])
        );
        assert_eq!(
            extended(Mode::Wrap, 4),
            s(&[2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
        );
        assert_eq!(extended(Mode::Constant(7.0), 4)[..6], [None; 6]);
        // one element: every mode but constant reads it
        for m in [Mode::Reflect, Mode::Mirror, Mode::Nearest, Mode::Wrap] {
            assert!(extended(m, 1).iter().all(|x| *x == Some(0)), "{m:?}");
        }
    }

    #[test]
    fn casts_are_c_casts() {
        assert_eq!(u8::from_f64(2.6), 2);
        assert_eq!(i8::from_f64(-2.6), -2);
        assert_eq!(u8::from_f64(400.0), 144);
        assert_eq!(u8::from_f64(-200.0), 56);
        assert_eq!(i16::from_f64(40000.0), -25536);
        assert_eq!(u32::from_f64(-1.0), u32::MAX);
        assert_eq!(i32::from_f64(3e9), i32::MIN);
        assert_eq!(u64::from_f64(1.8e19), 18_000_000_000_000_000_000);
        assert_eq!(f32::from_f64(0.1), 0.1f32);
    }

    #[test]
    fn mode_names() {
        assert_eq!(Mode::parse("grid-wrap", 0.0), Some(Mode::Wrap));
        assert_eq!(Mode::parse("constant", 2.0), Some(Mode::Constant(2.0)));
        assert_eq!(Mode::parse("reflect101", 0.0), None);
    }
}
