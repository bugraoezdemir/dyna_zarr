//! Pointwise operations as lazy views (Workplan WP15, task DR7 step 7), with NumPy's
//! values: the ufuncs whose results NumPy defines exactly (arithmetic, comparisons,
//! logic, bit operations, rounding, the square root), `where` and `clip`.
//!
//! An operation runs in one element type, its loop type, as a ufunc's loop does: every
//! array operand is converted to it (as NumPy casts a ufunc's inputs, and as `astype`
//! converts in-range values) and a scalar operand is given in it. The result has the loop type, or bool for the
//! comparisons, the logical operations and the tests (isnan, ...). `where` reads its
//! condition as bool. Integers wrap on overflow, as NumPy's loops do; shifts by the
//! type's width or more give 0 (or -1 for a negative value shifted right), as NumPy
//! defines them.
//!
//! Floats follow NumPy's loops where IEEE leaves a choice: `maximum` / `minimum`
//! propagate the first NaN (the first operand's when both are NaN) and give the second
//! operand when the two compare equal (so `maximum(-0.0, 0.0)` is `0.0`); `clip` gives
//! the bound when it is NaN or beyond the value, the value otherwise (so
//! `clip(-0.0, 0.0, 1.0)` is `-0.0`); `sign` keeps a NaN as it is and gives `0.0` for
//! either zero. dyna's lowering checks each operation against the NumPy it runs with
//! before handing it here (a NumPy whose rule differs keeps the Python path).
//!
//! A pointwise view over pointwise views is one expression ([`Pointwise`]): each array
//! underneath is read once per region (however many operations use it), and the
//! operations run tile by tile, every intermediate tile in cache.

use std::sync::Arc;

use rayon::prelude::*;

use crate::downsample::{split, split_cost, DEFAULT_MAX_READ};
use crate::dtype::DType;
use crate::out::Out;
use crate::pool::install;
use crate::source::{c_strides, region, Source, SourceRef, SMALL};
use crate::ReadError;

/// An operation, by its NumPy ufunc name (and `where`, `clip`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Add,
    AddNanFirst,
    SubtractNanFirst,
    MultiplyNanFirst,
    DivideNanFirst,
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LeftShift,
    RightShift,
    Copysign,
    Negative,
    Positive,
    Absolute,
    Fabs,
    Square,
    Sign,
    Floor,
    Ceil,
    Trunc,
    Rint,
    Sqrt,
    Reciprocal,
    Invert,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Equal,
    NotEqual,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    LogicalNot,
    IsNan,
    IsInf,
    IsFinite,
    Signbit,
    Where,
    Clip,
}

const NAMES: [(Op, &str); 45] = [
    (Op::Add, "add"),
    // where both operands are NaN, the first's (as NumPy's GCC-built loops keep it;
    // the plain names keep the second's, as its MSVC-built loops do)
    (Op::AddNanFirst, "add:nan_first"),
    (Op::SubtractNanFirst, "subtract:nan_first"),
    (Op::MultiplyNanFirst, "multiply:nan_first"),
    (Op::DivideNanFirst, "divide:nan_first"),
    (Op::Subtract, "subtract"),
    (Op::Multiply, "multiply"),
    (Op::Divide, "divide"),
    (Op::Maximum, "maximum"),
    (Op::Minimum, "minimum"),
    (Op::BitwiseAnd, "bitwise_and"),
    (Op::BitwiseOr, "bitwise_or"),
    (Op::BitwiseXor, "bitwise_xor"),
    (Op::LeftShift, "left_shift"),
    (Op::RightShift, "right_shift"),
    (Op::Copysign, "copysign"),
    (Op::Negative, "negative"),
    (Op::Positive, "positive"),
    (Op::Absolute, "absolute"),
    (Op::Fabs, "fabs"),
    (Op::Square, "square"),
    (Op::Sign, "sign"),
    (Op::Floor, "floor"),
    (Op::Ceil, "ceil"),
    (Op::Trunc, "trunc"),
    (Op::Rint, "rint"),
    (Op::Sqrt, "sqrt"),
    (Op::Reciprocal, "reciprocal"),
    (Op::Invert, "invert"),
    (Op::Greater, "greater"),
    (Op::GreaterEqual, "greater_equal"),
    (Op::Less, "less"),
    (Op::LessEqual, "less_equal"),
    (Op::Equal, "equal"),
    (Op::NotEqual, "not_equal"),
    (Op::LogicalAnd, "logical_and"),
    (Op::LogicalOr, "logical_or"),
    (Op::LogicalXor, "logical_xor"),
    (Op::LogicalNot, "logical_not"),
    (Op::IsNan, "isnan"),
    (Op::IsInf, "isinf"),
    (Op::IsFinite, "isfinite"),
    (Op::Signbit, "signbit"),
    (Op::Where, "where"),
    (Op::Clip, "clip"),
];

impl Op {
    /// The operation NumPy names `name`.
    pub fn parse(name: &str) -> Result<Self, ReadError> {
        NAMES
            .iter()
            .find(|(_, n)| *n == name)
            .map(|(op, _)| *op)
            .ok_or_else(|| {
                ReadError::new(format!(
                    "pointwise operation {name:?} is not one the engine has (it has {})",
                    NAMES.iter().map(|(_, n)| *n).collect::<Vec<_>>().join(", ")
                ))
            })
    }

    pub fn name(self) -> &'static str {
        NAMES
            .iter()
            .find(|(op, _)| *op == self)
            .map(|(_, n)| *n)
            .unwrap_or("?")
    }

    /// How many operands it takes.
    pub fn arity(self) -> usize {
        use Op::*;
        match self {
            Negative | Positive | Absolute | Fabs | Square | Sign | Floor | Ceil | Trunc | Rint
            | Sqrt | Reciprocal | Invert | LogicalNot | IsNan | IsInf | IsFinite | Signbit => 1,
            Where | Clip => 3,
            _ => 2,
        }
    }

    /// Whether its result is bool (else the loop type).
    pub fn gives_bool(self) -> bool {
        use Op::*;
        matches!(
            self,
            Greater
                | GreaterEqual
                | Less
                | LessEqual
                | Equal
                | NotEqual
                | LogicalAnd
                | LogicalOr
                | LogicalXor
                | LogicalNot
                | IsNan
                | IsInf
                | IsFinite
                | Signbit
        )
    }

    /// Whether it runs in loop type `t` (the loops NumPy has for it).
    pub fn supports(self, t: DType) -> bool {
        use Op::*;
        let float = matches!(t, DType::Float32 | DType::Float64);
        let int = matches!(
            t,
            DType::Int8
                | DType::Int16
                | DType::Int32
                | DType::Int64
                | DType::UInt8
                | DType::UInt16
                | DType::UInt32
                | DType::UInt64
        );
        let boolean = t == DType::Bool;
        match self {
            Divide | DivideNanFirst | Sqrt | Reciprocal | Fabs | Copysign | Rint => float,
            LeftShift | RightShift => int,
            BitwiseAnd | BitwiseOr | BitwiseXor | Invert => int || boolean,
            // NumPy squares bools in int8: no bool loop
            Negative | Positive | Sign | Subtract | SubtractNanFirst | Square => int || float,
            _ => int || float || boolean,
        }
    }

    /// The type operand `k` is read in, for loop type `t`.
    fn operand_type(self, k: usize, t: DType) -> DType {
        if self == Op::Where && k == 0 {
            DType::Bool
        } else {
            t
        }
    }
}

/// A bool element: 0 or 1 (any nonzero byte read is true).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct B(pub u8);

impl B {
    #[inline]
    fn of(v: bool) -> Self {
        B(u8::from(v))
    }
}

/// An element type of a loop: the operations NumPy's loops compute for it. An operation
/// a type has no loop for ([`Op::supports`]) is never called on it.
pub(crate) trait Elem: Copy + Default + Send + Sync + 'static {
    fn add(self, b: Self) -> Self;
    fn sub(self, b: Self) -> Self;
    fn mul(self, b: Self) -> Self;
    fn div(self, b: Self) -> Self;
    fn maximum(self, b: Self) -> Self;
    fn minimum(self, b: Self) -> Self;
    fn and(self, b: Self) -> Self;
    fn or(self, b: Self) -> Self;
    fn xor(self, b: Self) -> Self;
    fn shl(self, b: Self) -> Self;
    fn shr(self, b: Self) -> Self;
    fn copysign(self, b: Self) -> Self;
    fn neg(self) -> Self;
    fn abs(self) -> Self;
    fn square(self) -> Self;
    fn sign(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn trunc(self) -> Self;
    fn rint(self) -> Self;
    fn sqrt(self) -> Self;
    fn recip(self) -> Self;
    fn not(self) -> Self;
    fn less(self, b: Self) -> bool;
    fn less_eq(self, b: Self) -> bool;
    fn same(self, b: Self) -> bool;
    fn is_nan(self) -> bool;
    fn is_inf(self) -> bool;
    fn signbit(self) -> bool;
    fn truth(self) -> bool;
    fn clip(self, lo: Self, hi: Self) -> Self;
}

macro_rules! int_elem {
    ($t:ty, $u:ty, $signed:expr) => {
        impl Elem for $t {
            #[inline]
            fn add(self, b: Self) -> Self {
                self.wrapping_add(b)
            }
            #[inline]
            fn sub(self, b: Self) -> Self {
                self.wrapping_sub(b)
            }
            #[inline]
            fn mul(self, b: Self) -> Self {
                self.wrapping_mul(b)
            }
            #[inline]
            fn div(self, _: Self) -> Self {
                self // no integer loop (Op::supports)
            }
            #[inline]
            fn maximum(self, b: Self) -> Self {
                if self > b {
                    self
                } else {
                    b
                }
            }
            #[inline]
            fn minimum(self, b: Self) -> Self {
                if self < b {
                    self
                } else {
                    b
                }
            }
            #[inline]
            fn and(self, b: Self) -> Self {
                self & b
            }
            #[inline]
            fn or(self, b: Self) -> Self {
                self | b
            }
            #[inline]
            fn xor(self, b: Self) -> Self {
                self ^ b
            }
            #[inline]
            fn shl(self, b: Self) -> Self {
                // NumPy's npy_lshift: the shift read as unsigned; its width or more -> 0
                let s = b as $u;
                if (s as u64) < <$t>::BITS as u64 {
                    ((self as $u) << s) as $t
                } else {
                    0
                }
            }
            #[inline]
            fn shr(self, b: Self) -> Self {
                // npy_rshift: arithmetic for signed types; its width or more -> 0 or -1
                let s = b as $u;
                if (s as u64) < <$t>::BITS as u64 {
                    self >> s
                } else if $signed && self < (0 as $t) {
                    !(0 as $t)
                } else {
                    0
                }
            }
            #[inline]
            fn copysign(self, _: Self) -> Self {
                self // no integer loop
            }
            #[inline]
            fn neg(self) -> Self {
                self.wrapping_neg()
            }
            #[inline]
            fn abs(self) -> Self {
                if $signed && self < (0 as $t) {
                    self.wrapping_neg()
                } else {
                    self
                }
            }
            #[inline]
            fn square(self) -> Self {
                self.wrapping_mul(self)
            }
            #[inline]
            fn sign(self) -> Self {
                if self > (0 as $t) {
                    1
                } else if $signed && self < (0 as $t) {
                    !(0 as $t) // -1
                } else {
                    0
                }
            }
            #[inline]
            fn floor(self) -> Self {
                self
            }
            #[inline]
            fn ceil(self) -> Self {
                self
            }
            #[inline]
            fn trunc(self) -> Self {
                self
            }
            #[inline]
            fn rint(self) -> Self {
                self
            }
            #[inline]
            fn sqrt(self) -> Self {
                self // no integer loop
            }
            #[inline]
            fn recip(self) -> Self {
                self // no integer loop
            }
            #[inline]
            fn not(self) -> Self {
                !self
            }
            #[inline]
            fn less(self, b: Self) -> bool {
                self < b
            }
            #[inline]
            fn less_eq(self, b: Self) -> bool {
                self <= b
            }
            #[inline]
            fn same(self, b: Self) -> bool {
                self == b
            }
            #[inline]
            fn is_nan(self) -> bool {
                false
            }
            #[inline]
            fn is_inf(self) -> bool {
                false
            }
            #[inline]
            fn signbit(self) -> bool {
                $signed && self < (0 as $t)
            }
            #[inline]
            fn truth(self) -> bool {
                self != 0
            }
            #[inline]
            fn clip(self, lo: Self, hi: Self) -> Self {
                let v = if lo > self { lo } else { self };
                if hi < v {
                    hi
                } else {
                    v
                }
            }
        }
    };
}

int_elem!(i8, u8, true);
int_elem!(i16, u16, true);
int_elem!(i32, u32, true);
int_elem!(i64, u64, true);
int_elem!(u8, u8, false);
int_elem!(u16, u16, false);
int_elem!(u32, u32, false);
int_elem!(u64, u64, false);

macro_rules! float_elem {
    ($t:ty) => {
        impl Elem for $t {
            #[inline]
            fn add(self, b: Self) -> Self {
                self + b
            }
            #[inline]
            fn sub(self, b: Self) -> Self {
                self - b
            }
            #[inline]
            fn mul(self, b: Self) -> Self {
                self * b
            }
            #[inline]
            fn div(self, b: Self) -> Self {
                self / b
            }
            #[inline]
            fn maximum(self, b: Self) -> Self {
                // the first NaN; else the first only when greater (NaN b: b)
                if self.is_nan() || self > b {
                    self
                } else {
                    b
                }
            }
            #[inline]
            fn minimum(self, b: Self) -> Self {
                if self.is_nan() || self < b {
                    self
                } else {
                    b
                }
            }
            #[inline]
            fn and(self, _: Self) -> Self {
                self // no float loop
            }
            #[inline]
            fn or(self, _: Self) -> Self {
                self
            }
            #[inline]
            fn xor(self, _: Self) -> Self {
                self
            }
            #[inline]
            fn shl(self, _: Self) -> Self {
                self
            }
            #[inline]
            fn shr(self, _: Self) -> Self {
                self
            }
            #[inline]
            fn copysign(self, b: Self) -> Self {
                self.copysign(b)
            }
            #[inline]
            fn neg(self) -> Self {
                -self
            }
            #[inline]
            fn abs(self) -> Self {
                self.abs()
            }
            #[inline]
            fn square(self) -> Self {
                self * self
            }
            #[inline]
            fn sign(self) -> Self {
                if self > 0.0 {
                    1.0
                } else if self < 0.0 {
                    -1.0
                } else if self == 0.0 {
                    0.0
                } else {
                    self // NaN as it is
                }
            }
            #[inline]
            fn floor(self) -> Self {
                self.floor()
            }
            #[inline]
            fn ceil(self) -> Self {
                self.ceil()
            }
            #[inline]
            fn trunc(self) -> Self {
                self.trunc()
            }
            #[inline]
            fn rint(self) -> Self {
                self.round_ties_even()
            }
            #[inline]
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            #[inline]
            fn recip(self) -> Self {
                1.0 / self
            }
            #[inline]
            fn not(self) -> Self {
                self // no float loop
            }
            #[inline]
            fn less(self, b: Self) -> bool {
                self < b
            }
            #[inline]
            fn less_eq(self, b: Self) -> bool {
                self <= b
            }
            #[inline]
            fn same(self, b: Self) -> bool {
                self == b
            }
            #[inline]
            fn is_nan(self) -> bool {
                self.is_nan()
            }
            #[inline]
            fn is_inf(self) -> bool {
                self.is_infinite()
            }
            #[inline]
            fn signbit(self) -> bool {
                self.is_sign_negative()
            }
            #[inline]
            fn truth(self) -> bool {
                self != 0.0
            }
            #[inline]
            fn clip(self, lo: Self, hi: Self) -> Self {
                let v = if lo.is_nan() || lo > self { lo } else { self };
                if hi.is_nan() || hi < v {
                    hi
                } else {
                    v
                }
            }
        }
    };
}

float_elem!(f32);
float_elem!(f64);

impl Elem for B {
    #[inline]
    fn add(self, b: Self) -> Self {
        B::of(self.truth() || b.truth())
    }
    #[inline]
    fn sub(self, b: Self) -> Self {
        B::of(self.truth() ^ b.truth()) // no bool loop
    }
    #[inline]
    fn mul(self, b: Self) -> Self {
        B::of(self.truth() && b.truth())
    }
    #[inline]
    fn div(self, _: Self) -> Self {
        self // no bool loop
    }
    #[inline]
    fn maximum(self, b: Self) -> Self {
        self.add(b)
    }
    #[inline]
    fn minimum(self, b: Self) -> Self {
        self.mul(b)
    }
    #[inline]
    fn and(self, b: Self) -> Self {
        self.mul(b)
    }
    #[inline]
    fn or(self, b: Self) -> Self {
        self.add(b)
    }
    #[inline]
    fn xor(self, b: Self) -> Self {
        B::of(self.truth() ^ b.truth())
    }
    #[inline]
    fn shl(self, _: Self) -> Self {
        self // no bool loop
    }
    #[inline]
    fn shr(self, _: Self) -> Self {
        self
    }
    #[inline]
    fn copysign(self, _: Self) -> Self {
        self
    }
    #[inline]
    fn neg(self) -> Self {
        self
    }
    #[inline]
    fn abs(self) -> Self {
        B::of(self.truth())
    }
    #[inline]
    fn square(self) -> Self {
        B::of(self.truth())
    }
    #[inline]
    fn sign(self) -> Self {
        B::of(self.truth())
    }
    #[inline]
    fn floor(self) -> Self {
        B::of(self.truth())
    }
    #[inline]
    fn ceil(self) -> Self {
        B::of(self.truth())
    }
    #[inline]
    fn trunc(self) -> Self {
        B::of(self.truth())
    }
    #[inline]
    fn rint(self) -> Self {
        B::of(self.truth())
    }
    #[inline]
    fn sqrt(self) -> Self {
        self
    }
    #[inline]
    fn recip(self) -> Self {
        self
    }
    #[inline]
    fn not(self) -> Self {
        B::of(!self.truth())
    }
    #[inline]
    fn less(self, b: Self) -> bool {
        !self.truth() && b.truth()
    }
    #[inline]
    fn less_eq(self, b: Self) -> bool {
        !self.truth() || b.truth()
    }
    #[inline]
    fn same(self, b: Self) -> bool {
        self.truth() == b.truth()
    }
    #[inline]
    fn is_nan(self) -> bool {
        false
    }
    #[inline]
    fn is_inf(self) -> bool {
        false
    }
    #[inline]
    fn signbit(self) -> bool {
        false
    }
    #[inline]
    fn truth(self) -> bool {
        self.0 != 0
    }
    #[inline]
    fn clip(self, lo: Self, hi: Self) -> Self {
        let (x, lo, hi) = (self.truth(), lo.truth(), hi.truth());
        let v = if lo && !x { lo } else { x };
        B::of(if !hi && v { hi } else { v })
    }
}

/// An operand of a kernel: elements, or one value for every element.
#[derive(Clone, Copy)]
pub(crate) enum In<'a, T> {
    Arr(&'a [T]),
    Val(T),
}

impl<'a, T: Copy> In<'a, T> {
    fn part(&self, lo: usize, hi: usize) -> In<'a, T> {
        match self {
            In::Arr(a) => In::Arr(&a[lo..hi]),
            In::Val(v) => In::Val(*v),
        }
    }
}

fn map1<T: Copy, U>(a: In<T>, out: &mut [U], f: impl Fn(T) -> U) {
    match a {
        In::Arr(x) => {
            for (o, &p) in out.iter_mut().zip(x) {
                *o = f(p);
            }
        }
        In::Val(p) => {
            for o in out.iter_mut() {
                *o = f(p);
            }
        }
    }
}

fn map2<T: Copy, U>(a: In<T>, b: In<T>, out: &mut [U], f: impl Fn(T, T) -> U) {
    match (a, b) {
        (In::Arr(x), In::Arr(y)) => {
            for ((o, &p), &q) in out.iter_mut().zip(x).zip(y) {
                *o = f(p, q);
            }
        }
        (In::Arr(x), In::Val(q)) => {
            for (o, &p) in out.iter_mut().zip(x) {
                *o = f(p, q);
            }
        }
        (In::Val(p), In::Arr(y)) => {
            for (o, &q) in out.iter_mut().zip(y) {
                *o = f(p, q);
            }
        }
        (In::Val(p), In::Val(q)) => {
            for o in out.iter_mut() {
                *o = f(p, q);
            }
        }
    }
}

fn map3<T: Copy, U>(a: In<T>, b: In<T>, c: In<T>, out: &mut [U], f: impl Fn(T, T, T) -> U) {
    match (b, c) {
        // clip's usual form: an array between two scalar bounds
        (In::Val(q), In::Val(r)) => map1(a, out, |p| f(p, q, r)),
        _ => {
            for (i, o) in out.iter_mut().enumerate() {
                let at = |x: &In<T>| match x {
                    In::Arr(v) => v[i],
                    In::Val(v) => *v,
                };
                *o = f(at(&a), at(&b), at(&c));
            }
        }
    }
}

/// `f(p, q)`, and where both are NaN the second operand's NaN (IEEE leaves which one
/// open; NumPy's MSVC-built loops keep the second). `f(q, q)` of a NaN is that NaN,
/// quieted, so no bits are touched by hand.
#[inline(always)]
fn nan_second<T: Elem>(p: T, q: T, f: impl Fn(T, T) -> T) -> T {
    if p.is_nan() && q.is_nan() {
        f(q, q)
    } else {
        f(p, q)
    }
}

/// As [`nan_second`], keeping the first operand's NaN (NumPy's GCC-built loops).
#[inline(always)]
fn nan_first<T: Elem>(p: T, q: T, f: impl Fn(T, T) -> T) -> T {
    if p.is_nan() && q.is_nan() {
        f(p, p)
    } else {
        f(p, q)
    }
}

/// `op` (its result in the loop type) on `ins` into `out`; `cond` is `where`'s
/// condition.
fn same<T: Elem>(op: Op, ins: &[In<T>], cond: Option<In<B>>, out: &mut [T]) {
    use Op::*;
    let a = ins[0];
    match op {
        Add => map2(a, ins[1], out, |p, q| nan_second(p, q, T::add)),
        Subtract => map2(a, ins[1], out, |p, q| nan_second(p, q, T::sub)),
        Multiply => map2(a, ins[1], out, |p, q| nan_second(p, q, T::mul)),
        Divide => map2(a, ins[1], out, |p, q| nan_second(p, q, T::div)),
        AddNanFirst => map2(a, ins[1], out, |p, q| nan_first(p, q, T::add)),
        SubtractNanFirst => map2(a, ins[1], out, |p, q| nan_first(p, q, T::sub)),
        MultiplyNanFirst => map2(a, ins[1], out, |p, q| nan_first(p, q, T::mul)),
        DivideNanFirst => map2(a, ins[1], out, |p, q| nan_first(p, q, T::div)),
        Maximum => map2(a, ins[1], out, T::maximum),
        Minimum => map2(a, ins[1], out, T::minimum),
        BitwiseAnd => map2(a, ins[1], out, T::and),
        BitwiseOr => map2(a, ins[1], out, T::or),
        BitwiseXor => map2(a, ins[1], out, T::xor),
        LeftShift => map2(a, ins[1], out, T::shl),
        RightShift => map2(a, ins[1], out, T::shr),
        Copysign => map2(a, ins[1], out, T::copysign),
        Negative => map1(a, out, T::neg),
        Positive => map1(a, out, |v| v),
        Absolute | Fabs => map1(a, out, T::abs),
        Square => map1(a, out, T::square),
        Sign => map1(a, out, T::sign),
        Floor => map1(a, out, T::floor),
        Ceil => map1(a, out, T::ceil),
        Trunc => map1(a, out, T::trunc),
        Rint => map1(a, out, T::rint),
        Sqrt => map1(a, out, T::sqrt),
        Reciprocal => map1(a, out, T::recip),
        Invert => map1(a, out, T::not),
        Clip => map3(a, ins[1], ins[2], out, T::clip),
        Where => {
            let (x, y) = (ins[1], ins[2]);
            let c = cond.unwrap_or(In::Val(B(1)));
            match (c, x, y) {
                (In::Arr(c), In::Arr(x), In::Arr(y)) => {
                    for (i, o) in out.iter_mut().enumerate() {
                        *o = if c[i].truth() { x[i] } else { y[i] };
                    }
                }
                _ => {
                    let at = |v: &In<T>, i: usize| match v {
                        In::Arr(v) => v[i],
                        In::Val(v) => *v,
                    };
                    for (i, o) in out.iter_mut().enumerate() {
                        let t = match c {
                            In::Arr(c) => c[i].truth(),
                            In::Val(c) => c.truth(),
                        };
                        *o = if t { at(&x, i) } else { at(&y, i) };
                    }
                }
            }
        }
        _ => {} // the bool results: `boolean`
    }
}

/// `op` (a bool result) on `ins` into `out`.
fn boolean<T: Elem>(op: Op, ins: &[In<T>], out: &mut [B]) {
    use Op::*;
    let a = ins[0];
    match op {
        Greater => map2(a, ins[1], out, |p, q| B::of(q.less(p))),
        GreaterEqual => map2(a, ins[1], out, |p, q| B::of(q.less_eq(p))),
        Less => map2(a, ins[1], out, |p, q| B::of(p.less(q))),
        LessEqual => map2(a, ins[1], out, |p, q| B::of(p.less_eq(q))),
        Equal => map2(a, ins[1], out, |p, q| B::of(p.same(q))),
        NotEqual => map2(a, ins[1], out, |p, q| B::of(!p.same(q))),
        LogicalAnd => map2(a, ins[1], out, |p, q| B::of(p.truth() && q.truth())),
        LogicalOr => map2(a, ins[1], out, |p, q| B::of(p.truth() || q.truth())),
        LogicalXor => map2(a, ins[1], out, |p, q| B::of(p.truth() != q.truth())),
        LogicalNot => map1(a, out, |p| B::of(!p.truth())),
        IsNan => map1(a, out, |p| B::of(p.is_nan())),
        IsInf => map1(a, out, |p| B::of(p.is_inf())),
        IsFinite => map1(a, out, |p| B::of(!p.is_nan() && !p.is_inf())),
        Signbit => map1(a, out, |p| B::of(p.signbit())),
        _ => {}
    }
}

/// Elements per parallel task.
const TASK: usize = 1 << 15;

/// `f(first, out)` over `out` in tasks of [`TASK`] elements on the current pool when
/// `parallel`, else once over all of it.
fn tasks<U: Send>(out: &mut [U], parallel: bool, f: impl Fn(usize, &mut [U]) + Sync) {
    if parallel && out.len() > TASK {
        out.par_chunks_mut(TASK)
            .enumerate()
            .for_each(|(k, o)| f(k * TASK, o));
    } else {
        f(0, out);
    }
}

/// `op` (a loop-type result) on `ins` and `cond` into `out`.
fn run_same<T: Elem>(op: Op, ins: &[In<T>], cond: Option<In<B>>, out: &mut [T], parallel: bool) {
    tasks(out, parallel, |lo, o| {
        let hi = lo + o.len();
        let ins: Vec<In<T>> = ins.iter().map(|i| i.part(lo, hi)).collect();
        same(op, &ins, cond.map(|c| c.part(lo, hi)), o);
    });
}

/// `op` (a bool result) on `ins` into `out`.
fn run_bool<T: Elem>(op: Op, ins: &[In<T>], out: &mut [B], parallel: bool) {
    tasks(out, parallel, |lo, o| {
        let hi = lo + o.len();
        let ins: Vec<In<T>> = ins.iter().map(|i| i.part(lo, hi)).collect();
        boolean(op, &ins, o);
    });
}

/// A vector of `T` from the bytes of one or more elements (native byte order).
fn elements<T: Copy + Default>(bytes: &[u8]) -> Vec<T> {
    let size = std::mem::size_of::<T>();
    let mut v = vec![T::default(); bytes.len() / size];
    // SAFETY: `v` holds bytes.len() / size elements of plain data, written byte by byte
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, v.len() * size)
    };
    v
}

/// The bytes of a slice of plain elements.
fn bytes_of_slice<T>(v: &[T]) -> &[u8] {
    // SAFETY: plain numeric elements (or B), viewed as their bytes
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Run `$body` with `$T` the element type of the loop type `$t`.
macro_rules! with_type {
    ($t:expr, $T:ident, $body:expr) => {
        match $t {
            DType::Int8 => {
                type $T = i8;
                $body
            }
            DType::Int16 => {
                type $T = i16;
                $body
            }
            DType::Int32 => {
                type $T = i32;
                $body
            }
            DType::Int64 => {
                type $T = i64;
                $body
            }
            DType::UInt8 => {
                type $T = u8;
                $body
            }
            DType::UInt16 => {
                type $T = u16;
                $body
            }
            DType::UInt32 => {
                type $T = u32;
                $body
            }
            DType::UInt64 => {
                type $T = u64;
                $body
            }
            DType::Float32 => {
                type $T = f32;
                $body
            }
            DType::Float64 => {
                type $T = f64;
                $body
            }
            DType::Bool => {
                type $T = B;
                $body
            }
            other => Err(ReadError::new(format!(
                "pointwise operations in {}",
                other.name()
            ))),
        }
    };
}

fn check(op: Op, loop_t: DType, count: usize) -> Result<(), ReadError> {
    if count != op.arity() {
        return Err(ReadError::new(format!(
            "{} takes {} operands, not {count}",
            op.name(),
            op.arity()
        )));
    }
    if !op.supports(loop_t) {
        return Err(ReadError::new(format!(
            "{} has no {} loop",
            op.name(),
            loop_t.name()
        )));
    }
    Ok(())
}

/// `op` in loop type `loop_t` on `n` elements: each operand its elements' bytes (`n`
/// of them, in the operand's type: the loop type, bool for `where`'s condition) or one
/// element's bytes (`scalar`). The result's bytes: `n` elements of the loop type, or
/// bool. What the bindings' kernel function and dyna's check run.
pub fn apply(
    op: &str,
    loop_t: DType,
    operands: &[(&[u8], bool)],
    n: usize,
) -> Result<Vec<u8>, ReadError> {
    let op = Op::parse(op)?;
    check(op, loop_t, operands.len())?;
    for (k, (b, scalar)) in operands.iter().enumerate() {
        let size = op.operand_type(k, loop_t).size();
        let want = if *scalar { size } else { n * size };
        if b.len() != want {
            return Err(ReadError::new(format!(
                "operand {k} of {}: {} bytes, not {want}",
                op.name(),
                b.len()
            )));
        }
    }
    with_type!(loop_t, T, {
        let vals: Vec<Vec<T>> = operands
            .iter()
            .enumerate()
            .map(|(k, (b, _))| {
                if op == Op::Where && k == 0 {
                    Vec::new()
                } else {
                    elements::<T>(b)
                }
            })
            .collect();
        let ins: Vec<In<T>> = operands
            .iter()
            .enumerate()
            .filter(|(k, _)| !(op == Op::Where && *k == 0))
            .map(|(k, (_, scalar))| {
                if *scalar {
                    In::Val(vals[k][0])
                } else {
                    In::Arr(&vals[k])
                }
            })
            .collect();
        let cond_v: Vec<B> = if op == Op::Where {
            elements::<B>(operands[0].0)
        } else {
            Vec::new()
        };
        let cond = (op == Op::Where).then(|| {
            if operands[0].1 {
                In::Val(cond_v[0])
            } else {
                In::Arr(&cond_v[..])
            }
        });
        let ins = if op == Op::Where {
            // where(c, x, y): the kernel takes [_, x, y] with the condition apart
            let mut v = vec![In::Val(T::default())];
            v.extend(ins);
            v
        } else {
            ins
        };
        Ok(if op.gives_bool() {
            let mut out = vec![B::default(); n];
            run_bool::<T>(op, &ins, &mut out, false);
            bytes_of_slice(&out).to_vec()
        } else {
            let mut out = vec![T::default(); n];
            run_same::<T>(op, &ins, cond, &mut out, false);
            bytes_of_slice(&out).to_vec()
        })
    })
}

/// An operand of a [`Pointwise`] view.
pub enum Operand {
    /// An array of the view's shape (converted to the operand's type when read).
    Array(SourceRef),
    /// One element (its bytes, in the operand's type) for every position.
    Scalar(Vec<u8>),
}

/// A node of a fused expression; its output type is kept beside it.
#[derive(Clone)]
enum Node {
    /// The region of `leaves[k]`, in its own type.
    Leaf(usize),
    /// One element for every position (its bytes).
    Scalar(Vec<u8>),
    /// An operation in its loop type on earlier nodes.
    Op {
        op: Op,
        loop_t: DType,
        args: Vec<usize>,
    },
}

/// Bytes of an aligned buffer (u64 words hold every element type).
fn words(bytes: usize) -> Vec<u64> {
    vec![0u64; bytes.div_ceil(8)]
}

fn as_bytes(w: &[u64]) -> &[u8] {
    // SAFETY: plain words, viewed as bytes
    unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 8) }
}

fn as_bytes_mut(w: &mut [u64]) -> &mut [u8] {
    // SAFETY: plain words, viewed as bytes
    unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr() as *mut u8, w.len() * 8) }
}

/// `n` elements of `T` at the start of `b`, which is aligned for `T` (a word buffer at an
/// offset of whole elements).
fn typed<T>(b: &[u8], n: usize) -> &[T] {
    assert!(b.len() >= n * std::mem::size_of::<T>());
    assert_eq!(b.as_ptr() as usize % std::mem::align_of::<T>(), 0);
    // SAFETY: checked length and alignment; plain numeric elements (or B)
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const T, n) }
}

fn typed_mut<T>(b: &mut [u8], n: usize) -> &mut [T] {
    assert!(b.len() >= n * std::mem::size_of::<T>());
    assert_eq!(b.as_ptr() as usize % std::mem::align_of::<T>(), 0);
    // SAFETY: checked length and alignment; plain numeric elements (or B)
    unsafe { std::slice::from_raw_parts_mut(b.as_mut_ptr() as *mut T, n) }
}

/// One element converted as NumPy's astype converts in-range values (and as [`Cast`]
/// does): numeric to numeric with `as`, bool as 0 / 1, nonzero (NaN included) -> true.
trait ConvertTo<D> {
    fn to(self) -> D;
}

macro_rules! convert_numeric {
    ($s:ty => $($d:ty),*) => {
        $(impl ConvertTo<$d> for $s {
            #[inline]
            fn to(self) -> $d {
                self as $d
            }
        })*
        impl ConvertTo<B> for $s {
            #[inline]
            fn to(self) -> B {
                B::of(self != (0 as $s))
            }
        }
        impl ConvertTo<$s> for B {
            #[inline]
            fn to(self) -> $s {
                (self.0 != 0) as u8 as $s
            }
        }
    };
}

convert_numeric!(i8 => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
convert_numeric!(i16 => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
convert_numeric!(i32 => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
convert_numeric!(i64 => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
convert_numeric!(u8 => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
convert_numeric!(u16 => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
convert_numeric!(u32 => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
convert_numeric!(u64 => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
convert_numeric!(f32 => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
convert_numeric!(f64 => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl ConvertTo<B> for B {
    #[inline]
    fn to(self) -> B {
        B::of(self.0 != 0)
    }
}

fn convert_typed<S: Copy + ConvertTo<D>, D>(s: &[S], d: &mut [D]) {
    for (o, &v) in d.iter_mut().zip(s) {
        *o = v.to();
    }
}

/// `n` elements of type `from` (bytes `s`) converted into type `to` (bytes `d`).
fn convert(s: &[u8], from: DType, d: &mut [u8], to: DType, n: usize) -> Result<(), ReadError> {
    macro_rules! to_type {
        ($S:ty) => {{
            let s = typed::<$S>(s, n);
            with_type!(to, D, {
                convert_typed::<$S, D>(s, typed_mut::<D>(d, n));
                Ok(())
            })
        }};
    }
    with_type!(from, S, to_type!(S))
}

/// The operations of a fused pointwise expression, in order (the last one is the result),
/// and the arrays it reads.
#[derive(Clone)]
struct Expr {
    nodes: Vec<(Node, DType)>,
    leaves: Vec<SourceRef>,
}

impl Expr {
    /// The node of array `src`: its own expression merged in when it is a pointwise view,
    /// else a leaf (one per array, however often it is used).
    fn array(&mut self, src: &SourceRef) -> usize {
        if let Some(p) = src.as_pointwise() {
            return self.merge(&p.expr);
        }
        let id = Arc::as_ptr(src) as *const ();
        if let Some(k) = self
            .leaves
            .iter()
            .position(|l| Arc::as_ptr(l) as *const () == id)
        {
            if let Some(i) = self
                .nodes
                .iter()
                .position(|(n, _)| matches!(n, Node::Leaf(j) if *j == k))
            {
                return i;
            }
        }
        self.leaves.push(src.clone());
        self.nodes
            .push((Node::Leaf(self.leaves.len() - 1), src.dtype()));
        self.nodes.len() - 1
    }

    /// `other`'s nodes appended (its leaves shared with this one's); its result's node.
    fn merge(&mut self, other: &Expr) -> usize {
        let mut map = Vec::with_capacity(other.nodes.len());
        for (node, t) in &other.nodes {
            let i = match node {
                Node::Leaf(k) => self.array(&other.leaves[*k]),
                Node::Scalar(b) => {
                    self.nodes.push((Node::Scalar(b.clone()), *t));
                    self.nodes.len() - 1
                }
                Node::Op { op, loop_t, args } => {
                    let args = args.iter().map(|&a| map[a]).collect();
                    self.nodes.push((
                        Node::Op {
                            op: *op,
                            loop_t: *loop_t,
                            args,
                        },
                        *t,
                    ));
                    self.nodes.len() - 1
                }
            };
            map.push(i);
        }
        map[map.len() - 1]
    }

    fn out_t(&self) -> DType {
        self.nodes[self.nodes.len() - 1].1
    }
}

/// Elements of one tile: every node's tile of an expression stays in cache.
const TILE: usize = 4096;

/// One tile's buffers: per node (its result), per node and operand (a conversion).
struct Scratch {
    temps: Vec<Vec<u64>>,
    convs: Vec<Vec<Vec<u64>>>,
}

/// A pointwise operation on arrays of one shape and scalars, read lazily. Pointwise views
/// among its operands are merged into one expression: each array underneath is read once
/// per region, and the operations run tile by tile in cache.
pub struct Pointwise {
    expr: Expr,
    shape: Vec<u64>,
    max_read: u64,
}

impl Pointwise {
    /// `op` (a NumPy ufunc name, `where` or `clip`) in loop type `loop_t` on `operands`
    /// (at least one array; all arrays of one shape; `where`'s condition read as bool,
    /// the others in the loop type).
    pub fn new(op: &str, operands: Vec<Operand>, loop_t: DType) -> Result<Self, ReadError> {
        let op = Op::parse(op)?;
        check(op, loop_t, operands.len())?;
        let mut shape: Option<Vec<u64>> = None;
        let mut expr = Expr {
            nodes: Vec::new(),
            leaves: Vec::new(),
        };
        let mut args = Vec::with_capacity(operands.len());
        for (k, o) in operands.into_iter().enumerate() {
            let want = op.operand_type(k, loop_t);
            args.push(match o {
                Operand::Array(src) => {
                    match &shape {
                        None => shape = Some(src.shape().to_vec()),
                        Some(s) if s.as_slice() != src.shape() => {
                            return Err(ReadError::new(format!(
                                "{}: operands of shapes {s:?} and {:?} (the engine does not \
                                 broadcast)",
                                op.name(),
                                src.shape()
                            )))
                        }
                        _ => {}
                    }
                    if src.dtype().is_complex() || src.dtype() == DType::Float16 {
                        return Err(ReadError::new(format!(
                            "{}: an operand of {}",
                            op.name(),
                            src.dtype().name()
                        )));
                    }
                    expr.array(&src)
                }
                Operand::Scalar(b) => {
                    if b.len() != want.size() {
                        return Err(ReadError::new(format!(
                            "{}: a scalar of {} bytes for {}",
                            op.name(),
                            b.len(),
                            want.name()
                        )));
                    }
                    expr.nodes.push((Node::Scalar(b), want));
                    expr.nodes.len() - 1
                }
            });
        }
        let shape = shape
            .ok_or_else(|| ReadError::new(format!("{}: no array among the operands", op.name())))?;
        let out_t = if op.gives_bool() { DType::Bool } else { loop_t };
        expr.nodes.push((Node::Op { op, loop_t, args }, out_t));
        Ok(Self {
            expr,
            shape,
            max_read: DEFAULT_MAX_READ,
        })
    }

    /// Read at most about `bytes` of arrays and result at once (a larger request is
    /// split first).
    pub fn with_max_read(mut self, bytes: u64) -> Self {
        self.max_read = bytes.max(1);
        self
    }

    /// Bytes per element of the arrays read and the result.
    fn per_element(&self) -> u64 {
        let ins: usize = self.expr.leaves.iter().map(|l| l.dtype().size()).sum();
        (ins + self.expr.out_t().size()) as u64
    }

    /// Per node and operand: whether the operand must be converted first.
    fn needs_conversion(&self, i: usize) -> Vec<bool> {
        match &self.expr.nodes[i].0 {
            Node::Op { op, loop_t, args } => args
                .iter()
                .enumerate()
                .map(|(k, &a)| {
                    !matches!(self.expr.nodes[a].0, Node::Scalar(_))
                        && self.expr.nodes[a].1 != op.operand_type(k, *loop_t)
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Bytes of one tile's buffer for node `i` (an operation before the last), and of
    /// each of its conversions.
    fn tile_bytes(&self, i: usize) -> (usize, Vec<usize>) {
        let n = self.expr.nodes.len();
        let temp = match self.expr.nodes[i].0 {
            Node::Op { .. } if i + 1 < n => TILE * self.expr.nodes[i].1.size(),
            _ => 0,
        };
        let convs = self
            .needs_conversion(i)
            .into_iter()
            .map(|c| if c { TILE * 8 } else { 0 })
            .collect();
        (temp, convs)
    }

    fn scratch(&self) -> Scratch {
        let n = self.expr.nodes.len();
        let plan: Vec<(usize, Vec<usize>)> = (0..n).map(|i| self.tile_bytes(i)).collect();
        let buf = |b: usize| if b > 0 { words(b) } else { Vec::new() };
        Scratch {
            temps: plan.iter().map(|(t, _)| buf(*t)).collect(),
            convs: plan
                .iter()
                .map(|(_, c)| c.iter().map(|&b| buf(b)).collect())
                .collect(),
        }
    }

    /// Bytes of one [`Scratch`] (and its vectors' headers).
    fn scratch_cost(&self) -> u64 {
        let n = self.expr.nodes.len();
        (0..n)
            .map(|i| {
                let (t, c) = self.tile_bytes(i);
                (t + c.iter().sum::<usize>() + 24 * (1 + c.len())) as u64
            })
            .sum::<u64>()
            + 48
    }

    /// One tile `lo .. lo + m` of the expression into `out` (the result's bytes for the
    /// tile), the arrays' regions in `leaves`.
    fn tile(
        &self,
        leaves: &[Vec<u64>],
        lo: usize,
        m: usize,
        s: &mut Scratch,
        out: &mut [u8],
    ) -> Result<(), ReadError> {
        let nodes = &self.expr.nodes;
        let last = nodes.len() - 1;
        for i in 0..nodes.len() {
            let Node::Op { op, loop_t, args } = &nodes[i].0 else {
                continue;
            };
            let (op, loop_t) = (*op, *loop_t);
            let (before, rest) = s.temps.split_at_mut(i);
            let convs = &mut s.convs[i];
            // the bytes of operand node `a` for this tile, in its own type
            let source = |a: usize| -> &[u8] {
                let t = nodes[a].1.size();
                match &nodes[a].0 {
                    Node::Leaf(k) => &as_bytes(&leaves[*k])[lo * t..(lo + m) * t],
                    Node::Scalar(b) => b,
                    Node::Op { .. } => &as_bytes(&before[a])[..m * t],
                }
            };
            for (k, &a) in args.iter().enumerate() {
                if !convs[k].is_empty() {
                    let want = op.operand_type(k, loop_t);
                    convert(source(a), nodes[a].1, as_bytes_mut(&mut convs[k]), want, m)?;
                }
            }
            let dst: &mut [u8] = if i == last {
                &mut *out
            } else {
                as_bytes_mut(&mut rest[0])
            };
            let convs = &s.convs[i];
            let operand = |k: usize| -> (&[u8], bool) {
                let a = args[k];
                if !convs[k].is_empty() {
                    (as_bytes(&convs[k]), false)
                } else {
                    (source(a), matches!(nodes[a].0, Node::Scalar(_)))
                }
            };
            with_type!(loop_t, T, {
                let mut ins: Vec<In<T>> = Vec::with_capacity(args.len());
                let mut cond: Option<In<B>> = None;
                for k in 0..args.len() {
                    let (b, scalar) = operand(k);
                    if op == Op::Where && k == 0 {
                        cond = Some(if scalar {
                            In::Val(typed::<B>(b, 1)[0])
                        } else {
                            In::Arr(typed::<B>(b, m))
                        });
                        ins.push(In::Val(T::default()));
                    } else {
                        ins.push(if scalar {
                            In::Val(elements::<T>(b)[0])
                        } else {
                            In::Arr(typed::<T>(b, m))
                        });
                    }
                }
                if op.gives_bool() {
                    boolean(op, &ins, typed_mut::<B>(dst, m));
                } else {
                    same(op, &ins, cond, typed_mut::<T>(dst, m));
                }
                Ok::<(), ReadError>(())
            })?;
        }
        Ok(())
    }

    fn part(&self, start: &[u64], r: &[u64], out: &Out, threads: usize) -> Result<(), ReadError> {
        let stop: Vec<u64> = start.iter().zip(r).map(|(s, n)| s + n).collect();
        let n = r.iter().product::<u64>() as usize;
        let mut leaves = Vec::with_capacity(self.expr.leaves.len());
        for l in &self.expr.leaves {
            let mut w = words(n * l.dtype().size());
            {
                let bytes = &mut as_bytes_mut(&mut w)[..n * l.dtype().size()];
                let o = Out::new(bytes, r, &vec![0; r.len()], l.dtype())?;
                l.read_into(start, &stop, &o, threads)?;
            }
            leaves.push(w);
        }
        let out_t = self.expr.out_t();
        let size = out_t.size();
        let mut res = words(n * size);
        {
            let bytes = &mut as_bytes_mut(&mut res)[..n * size];
            let go = |(k, chunk): (usize, &mut [u8]), s: &mut Scratch| {
                self.tile(&leaves, k * TILE, chunk.len() / size, s, chunk)
            };
            if threads > 1 && n > TILE {
                install(threads, || {
                    bytes
                        .par_chunks_mut(TILE * size)
                        .enumerate()
                        .try_for_each_init(|| self.scratch(), |s, c| go(c, s))
                })??;
            } else {
                let mut s = self.scratch();
                for c in bytes.chunks_mut(TILE * size).enumerate() {
                    go(c, &mut s)?;
                }
            }
        }
        drop(leaves);
        out.copy_from(r, as_bytes(&res), &c_strides(r, size), size, None);
        Ok(())
    }

    /// One piece at its fullest: the arrays read before the last one held while it is
    /// read (with what reading it costs), then the arrays, the result and the tiles.
    fn part_cost(&self, start: &[u64], r: &[u64], threads: usize) -> Result<u64, ReadError> {
        if r.contains(&0) {
            return Ok(0);
        }
        let stop: Vec<u64> = start.iter().zip(r).map(|(s, n)| s + n).collect();
        let n = r.iter().product::<u64>();
        let mut held = 0u64;
        let mut peak = 0u64;
        for l in &self.expr.leaves {
            let b = (n * l.dtype().size() as u64).div_ceil(8) * 8;
            peak = peak.max(held + b + l.cost(start, &stop, threads)?);
            held += b;
        }
        let tiles = threads.max(1) as u64 * self.scratch_cost();
        let computing = held + n * self.expr.out_t().size() as u64 + 8 + tiles;
        Ok(peak.max(computing) + SMALL)
    }
}

impl Source for Pointwise {
    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.expr.out_t()
    }
    fn grid(&self) -> Option<Vec<u64>> {
        self.expr.leaves.iter().find_map(|l| l.grid())
    }
    fn as_pointwise(&self) -> Option<&Pointwise> {
        Some(self)
    }

    fn read_into(
        &self,
        start: &[u64],
        stop: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        let r = region(&self.shape, start, stop)?;
        out.check(&r, self.dtype())?;
        let per = self.per_element();
        let bytes = |_: &[u64], r: &[u64]| r.iter().product::<u64>() * per;
        let run = |s: &[u64], r: &[u64], o: &Out| self.part(s, r, o, threads);
        split(start, &r, out, self.max_read, &bytes, &run)
    }

    fn cost(&self, start: &[u64], stop: &[u64], threads: usize) -> Result<u64, ReadError> {
        let r = region(&self.shape, start, stop)?;
        let per = self.per_element();
        let bytes = |_: &[u64], r: &[u64]| r.iter().product::<u64>() * per;
        let cost = |s: &[u64], r: &[u64]| self.part_cost(s, r, threads);
        split_cost(start, &r, self.max_read, &bytes, &cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::read;

    /// An in-memory source for the tests: bytes of `dtype` in C order.
    struct Mem {
        shape: Vec<u64>,
        dtype: DType,
        data: Vec<u8>,
        reads: std::sync::atomic::AtomicUsize,
    }

    impl Source for Mem {
        fn shape(&self) -> &[u64] {
            &self.shape
        }
        fn dtype(&self) -> DType {
            self.dtype
        }
        fn read_into(
            &self,
            start: &[u64],
            stop: &[u64],
            out: &Out,
            _: usize,
        ) -> Result<(), ReadError> {
            self.reads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let r = region(&self.shape, start, stop)?;
            let item = self.dtype.size();
            let strides = c_strides(&self.shape, item);
            let off: usize = start
                .iter()
                .zip(&strides)
                .map(|(s, t)| *s as usize * t)
                .sum();
            out.copy_from(&r, &self.data[off..], &strides, item, None);
            Ok(())
        }
        fn cost(&self, _: &[u64], _: &[u64], _: usize) -> Result<u64, ReadError> {
            Ok(0)
        }
    }

    fn mem<T: Copy>(shape: &[u64], dtype: DType, v: &[T]) -> SourceRef {
        Arc::new(Mem {
            shape: shape.to_vec(),
            dtype,
            data: bytes_of_slice(v).to_vec(),
            reads: Default::default(),
        })
    }

    #[test]
    fn integers_wrap_and_shift_as_numpy() {
        let a = [100i8, -128, 7, -1];
        let b = [100i8, -1, 9, -1];
        let ab: Vec<(&[u8], bool)> = vec![(bytes_of_slice(&a), false), (bytes_of_slice(&b), false)];
        let add = elements::<i8>(&apply("add", DType::Int8, &ab, 4).unwrap());
        assert_eq!(add, vec![-56, 127, 16, -2]);
        let shl = elements::<i8>(&apply("left_shift", DType::Int8, &ab, 4).unwrap());
        assert_eq!(shl, vec![0, 0, 0, 0]); // 100 << 100, ... by width or more (or < 0)
        let shr = elements::<i8>(&apply("right_shift", DType::Int8, &ab, 4).unwrap());
        assert_eq!(shr, vec![0, -1, 0, -1]);
        let abs = elements::<i8>(&apply("absolute", DType::Int8, &ab[..1], 4).unwrap());
        assert_eq!(abs, vec![100, -128, 7, 1]);
    }

    #[test]
    fn float_extremes_and_clip_follow_numpy() {
        let nan = f64::NAN;
        let a = [-0.0f64, 0.0, nan, 1.0, -nan];
        let b = [0.0f64, -0.0, 1.0, nan, nan];
        let ab: Vec<(&[u8], bool)> = vec![(bytes_of_slice(&a), false), (bytes_of_slice(&b), false)];
        let max = elements::<f64>(&apply("maximum", DType::Float64, &ab, 5).unwrap());
        let bits: Vec<u64> = max.iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits[0], 0.0f64.to_bits()); // equal: the second
        assert_eq!(bits[1], (-0.0f64).to_bits());
        assert!(max[2].is_nan() && max[3].is_nan());
        assert!(max[4].is_sign_negative()); // both NaN: the first
        let lo = 0.0f64.to_ne_bytes();
        let hi = 1.0f64.to_ne_bytes();
        let clip = elements::<f64>(
            &apply(
                "clip",
                DType::Float64,
                &[(bytes_of_slice(&a), false), (&lo, true), (&hi, true)],
                5,
            )
            .unwrap(),
        );
        assert_eq!(clip[0].to_bits(), (-0.0f64).to_bits());
        assert_eq!(clip[3], 1.0);
        assert!(clip[4].is_nan() && clip[4].is_sign_negative());
    }

    #[test]
    fn a_view_reads_regions_with_casts_scalars_and_where() {
        let shape = [3u64, 4, 5];
        let x: Vec<u16> = (0..60).map(|i| (i * 1000) as u16).collect();
        let y: Vec<f32> = (0..60).map(|i| i as f32 * 0.5 - 7.0).collect();
        let xs = mem(&shape, DType::UInt16, &x);
        let ys = mem(&shape, DType::Float32, &y);
        // x * 0.25 > y in float64: uint16 and float32 both converted
        let q = Pointwise::new(
            "multiply",
            vec![
                Operand::Array(xs.clone()),
                Operand::Scalar(0.25f64.to_ne_bytes().to_vec()),
            ],
            DType::Float64,
        )
        .unwrap();
        let gt = Pointwise::new(
            "greater",
            vec![Operand::Array(Arc::new(q)), Operand::Array(ys.clone())],
            DType::Float64,
        )
        .unwrap();
        assert_eq!(gt.dtype(), DType::Bool);
        let w: SourceRef = Arc::new(gt);
        let pick = Pointwise::new(
            "where",
            vec![
                Operand::Array(w),
                Operand::Array(ys.clone()),
                Operand::Scalar((-1.0f32).to_ne_bytes().to_vec()),
            ],
            DType::Float32,
        )
        .unwrap()
        .with_max_read(64);
        for threads in [1, 3] {
            let got = elements::<f32>(&read(&pick, &[1, 1, 2], &[3, 4, 5], threads).unwrap());
            let mut want = Vec::new();
            for z in 1..3 {
                for yy in 1..4 {
                    for xx in 2..5 {
                        let i = z * 20 + yy * 5 + xx;
                        let c = x[i] as f64 * 0.25 > y[i] as f64;
                        want.push(if c { y[i] } else { -1.0 });
                    }
                }
            }
            assert_eq!(got, want);
        }
    }

    #[test]
    fn errors_name_the_way_out() {
        let s = mem(&[2], DType::Int16, &[1i16, 2]);
        let t = mem(&[3], DType::Int16, &[1i16, 2, 3]);
        assert!(Pointwise::new("sin", vec![Operand::Array(s.clone())], DType::Float64).is_err());
        assert!(Pointwise::new("sqrt", vec![Operand::Array(s.clone())], DType::Int16).is_err());
        assert!(Pointwise::new(
            "add",
            vec![Operand::Array(s.clone()), Operand::Array(t)],
            DType::Int16
        )
        .is_err());
        assert!(Pointwise::new(
            "add",
            vec![Operand::Scalar(vec![0, 0]), Operand::Scalar(vec![0, 0])],
            DType::Int16
        )
        .is_err());
        assert!(Pointwise::new("negative", vec![Operand::Array(s)], DType::Bool).is_err());
    }

    #[test]
    fn a_chain_is_one_expression_reading_each_array_once() {
        // where(g > 0.5 * g2, g, -g2) over two arrays, through four pointwise views:
        // merged into one expression, each array read once per region
        let shape = [5u64, 7, 9000]; // more than one tile per row
        let n = 5 * 7 * 9000;
        let g: Vec<f32> = (0..n).map(|i| ((i * 37) % 101) as f32 - 50.0).collect();
        let h: Vec<i16> = (0..n).map(|i| ((i * 13) % 61) as i16 - 30).collect();
        let gm = Arc::new(Mem {
            shape: shape.to_vec(),
            dtype: DType::Float32,
            data: bytes_of_slice(&g).to_vec(),
            reads: Default::default(),
        });
        let hm = Arc::new(Mem {
            shape: shape.to_vec(),
            dtype: DType::Int16,
            data: bytes_of_slice(&h).to_vec(),
            reads: Default::default(),
        });
        let gs: SourceRef = gm.clone();
        let hs: SourceRef = hm.clone();
        let half = Pointwise::new(
            "multiply",
            vec![
                Operand::Array(hs.clone()),
                Operand::Scalar(0.5f32.to_ne_bytes().to_vec()),
            ],
            DType::Float32,
        )
        .unwrap();
        let gt = Pointwise::new(
            "greater",
            vec![Operand::Array(gs.clone()), Operand::Array(Arc::new(half))],
            DType::Float32,
        )
        .unwrap();
        let neg =
            Pointwise::new("negative", vec![Operand::Array(hs.clone())], DType::Int16).unwrap();
        let w = Pointwise::new(
            "where",
            vec![
                Operand::Array(Arc::new(gt)),
                Operand::Array(gs.clone()),
                Operand::Array(Arc::new(neg)),
            ],
            DType::Float32,
        )
        .unwrap();
        assert_eq!(w.expr.leaves.len(), 2);
        for threads in [1, 4] {
            gm.reads.store(0, std::sync::atomic::Ordering::Relaxed);
            hm.reads.store(0, std::sync::atomic::Ordering::Relaxed);
            let got = elements::<f32>(&read(&w, &[1, 2, 5], &[5, 7, 9000], threads).unwrap());
            assert_eq!(gm.reads.load(std::sync::atomic::Ordering::Relaxed), 1);
            assert_eq!(hm.reads.load(std::sync::atomic::Ordering::Relaxed), 1);
            let mut want = Vec::new();
            for z in 1..5 {
                for y in 2..7 {
                    for x in 5..9000 {
                        let i = (z * 7 + y) * 9000 + x;
                        let c = g[i] > h[i] as f32 * 0.5;
                        want.push(if c { g[i] } else { -(h[i] as f32) });
                    }
                }
            }
            assert_eq!(got, want);
        }
    }
}
