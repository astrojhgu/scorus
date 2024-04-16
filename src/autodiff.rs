#![allow(clippy::redundant_closure)]
// The majority of the code in this module was forked from https://github.com/ibab/rust-ad in 2016.
// The copyright notice is reproduced below:
//
// ```
// Copyright (c) 2014 Igor Babuschkin
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// ```
//
// This crate is licensed under the terms described in the README.md, which is located at the root
// directory of this crate.

use num::traits::{Float, FloatConst, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};
use std::f64;
use std::num::FpCategory;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[derive(Copy, Clone, Debug)]
pub struct F<T>
where
    T: Float,
{
    pub x: T,
    pub dx: T,
}

impl<T> Neg for F<T>
where
    T: Float,
{
    type Output = F<T>;
    #[inline]
    fn neg(self) -> F<T> {
        F {
            x: -self.x,
            dx: -self.dx,
        }
    }
}

impl<T> Add<F<T>> for F<T>
where
    T: Float,
{
    type Output = F<T>;
    #[inline]
    fn add(self, rhs: F<T>) -> F<T> {
        F {
            x: self.x + rhs.x,
            dx: self.dx + rhs.dx,
        }
    }
}

impl<T> Add<T> for F<T>
where
    T: Float,
{
    type Output = F<T>;
    #[inline]
    fn add(self, rhs: T) -> F<T> {
        F {
            x: self.x + rhs,
            dx: self.dx,
        }
    }
}

impl<T> AddAssign for F<T>
where
    T: Float + AddAssign,
{
    #[inline]
    fn add_assign(&mut self, rhs: F<T>) {
        self.x += rhs.x;
        self.dx += rhs.dx;
    }
}

impl<T> AddAssign<T> for F<T>
where
    T: Float + AddAssign,
{
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        self.x += rhs;
    }
}

impl<T> Sub<F<T>> for F<T>
where
    T: Float,
{
    type Output = F<T>;
    #[inline]
    fn sub(self, rhs: F<T>) -> F<T> {
        F {
            x: self.x - rhs.x,
            dx: self.dx - rhs.dx,
        }
    }
}

impl<T> Sub<T> for F<T>
where
    T: Float,
{
    type Output = F<T>;
    #[inline]
    fn sub(self, rhs: T) -> F<T> {
        F {
            x: self.x - rhs,
            dx: self.dx,
        }
    }
}

impl<T> SubAssign for F<T>
where
    T: Float + SubAssign,
{
    #[inline]
    fn sub_assign(&mut self, rhs: F<T>) {
        self.x -= rhs.x;
        self.dx -= rhs.dx;
    }
}

impl<T> SubAssign<T> for F<T>
where
    T: Float + SubAssign,
{
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        self.x -= rhs;
    }
}

/*
 * Multiplication
 */

impl<T> Mul<F<T>> for F<T>
where
    T: Float,
{
    type Output = F<T>;
    #[inline]
    fn mul(self, rhs: F<T>) -> F<T> {
        F {
            x: self.x * rhs.x,
            dx: self.dx * rhs.x + self.x * rhs.dx,
        }
    }
}

// Multiply by double precision floats (treated as constants)

impl<T> Mul<T> for F<T>
where
    T: Float,
{
    type Output = F<T>;
    #[inline]
    fn mul(self, rhs: T) -> F<T> {
        // rhs is treated as a constant
        F {
            x: self.x * rhs,
            dx: self.dx * rhs,
        }
    }
}

// Multiply assign operators

impl<T> MulAssign for F<T>
where
    T: Float + MulAssign,
{
    #[inline]
    fn mul_assign(&mut self, rhs: F<T>) {
        *self = *self * rhs;
    }
}

impl<T> MulAssign<T> for F<T>
where
    T: Float + MulAssign,
{
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        // rhs is treated as a constant
        self.x *= rhs;
        self.dx *= rhs;
    }
}

// MulAssign<F> for f64 is not implemented deliberately, because this operation erases the
// tracking of the derivative information.

/*
 * Division
 */

impl<T> Div<F<T>> for F<T>
where
    T: Float,
{
    type Output = F<T>;
    #[inline]
    fn div(self, rhs: F<T>) -> F<T> {
        F {
            x: self.x / rhs.x,
            dx: (self.dx * rhs.x - self.x * rhs.dx) / (rhs.x * rhs.x),
        }
    }
}

// Division by double precision floats

impl<T> Div<T> for F<T>
where
    T: Float,
{
    type Output = F<T>;
    #[inline]
    fn div(self, rhs: T) -> F<T> {
        F {
            x: self.x / rhs,
            dx: self.dx / rhs,
        }
    }
}

impl<T> DivAssign for F<T>
where
    T: Float + DivAssign,
{
    #[inline]
    fn div_assign(&mut self, rhs: F<T>) {
        *self = *self / rhs; // reuse quotient rule implementation
    }
}

impl<T> DivAssign<T> for F<T>
where
    T: Float + DivAssign,
{
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.dx /= rhs;
    }
}

/*
 * Remainder function
 */

impl<T> Rem<F<T>> for F<T>
where
    T: Float,
{
    type Output = F<T>;
    #[inline]
    fn rem(self, rhs: F<T>) -> F<T> {
        // This is an approximation. There are places where the derivative doesn't exist.
        F {
            x: self.x % rhs.x, // x % y = x - [x/|y|]*|y|
            dx: self.dx - (self.x / rhs.x).trunc() * rhs.dx,
        }
    }
}

impl<T> Rem<T> for F<T>
where
    T: Float,
{
    type Output = F<T>;
    #[inline]
    fn rem(self, rhs: T) -> F<T> {
        // This is an approximation. There are places where the derivative doesn't exist.
        F {
            x: self.x % rhs, // x % y = x - [x/|y|]*|y|
            dx: self.dx,
        }
    }
}

impl<T> RemAssign for F<T>
where
    T: Float,
{
    #[inline]
    fn rem_assign(&mut self, rhs: F<T>) {
        *self = *self % rhs; // resuse non-trivial implementation
    }
}

impl<T> RemAssign<T> for F<T>
where
    T: Float,
{
    #[inline]
    fn rem_assign(&mut self, rhs: T) {
        *self = *self % rhs; // resuse non-trivial implementation
    }
}

impl<T> Default for F<T>
where
    T: Float + Default,
{
    #[inline]
    fn default() -> Self {
        F {
            x: T::default(),
            dx: T::zero(),
        }
    }
}

impl<T> PartialEq<F<T>> for F<T>
where
    T: Float,
{
    #[inline]
    fn eq(&self, rhs: &F<T>) -> bool {
        self.x == rhs.x
    }
}

impl<T> PartialOrd<F<T>> for F<T>
where
    T: Float,
{
    #[inline]
    fn partial_cmp(&self, other: &F<T>) -> Option<::std::cmp::Ordering> {
        PartialOrd::partial_cmp(&self.x, &other.x)
    }
}

impl<T> ToPrimitive for F<T>
where
    T: Float,
{
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.x.to_i64()
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.x.to_u64()
    }
    #[inline]
    fn to_isize(&self) -> Option<isize> {
        self.x.to_isize()
    }
    #[inline]
    fn to_i8(&self) -> Option<i8> {
        self.x.to_i8()
    }
    #[inline]
    fn to_i16(&self) -> Option<i16> {
        self.x.to_i16()
    }
    #[inline]
    fn to_i32(&self) -> Option<i32> {
        self.x.to_i32()
    }
    #[inline]
    fn to_usize(&self) -> Option<usize> {
        self.x.to_usize()
    }
    #[inline]
    fn to_u8(&self) -> Option<u8> {
        self.x.to_u8()
    }
    #[inline]
    fn to_u16(&self) -> Option<u16> {
        self.x.to_u16()
    }
    #[inline]
    fn to_u32(&self) -> Option<u32> {
        self.x.to_u32()
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.x.to_f32()
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.x.to_f64()
    }
}

impl<T> NumCast for F<T>
where
    T: Float + NumCast,
{
    fn from<U: ToPrimitive>(n: U) -> Option<F<T>> {
        n.to_f64().map(|x| F {
            x: T::from(x).unwrap(),
            dx: T::zero(),
        })
    }
}
impl<T> FromPrimitive for F<T>
where
    T: Float + FromPrimitive,
{
    #[inline]
    fn from_isize(n: isize) -> Option<Self> {
        FromPrimitive::from_isize(n).map(|x: T| F::cst(x))
    }
    #[inline]
    fn from_i8(n: i8) -> Option<Self> {
        FromPrimitive::from_i8(n).map(|x: T| F::cst(x))
    }
    #[inline]
    fn from_i16(n: i16) -> Option<Self> {
        FromPrimitive::from_i16(n).map(|x: T| F::cst(x))
    }
    #[inline]
    fn from_i32(n: i32) -> Option<Self> {
        FromPrimitive::from_i32(n).map(|x: T| F::cst(x))
    }
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        FromPrimitive::from_i64(n).map(|x: T| F::cst(x))
    }
    #[inline]
    fn from_i128(n: i128) -> Option<Self> {
        FromPrimitive::from_i128(n).map(|x: T| F::cst(x))
    }
    #[inline]
    fn from_usize(n: usize) -> Option<Self> {
        FromPrimitive::from_usize(n).map(|x: T| F::cst(x))
    }
    #[inline]
    fn from_u8(n: u8) -> Option<Self> {
        FromPrimitive::from_u8(n).map(|x: T| F::cst(x))
    }
    #[inline]
    fn from_u16(n: u16) -> Option<Self> {
        FromPrimitive::from_u16(n).map(|x: T| F::cst(x))
    }
    #[inline]
    fn from_u32(n: u32) -> Option<Self> {
        FromPrimitive::from_u32(n).map(|x: T| F::cst(x))
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        FromPrimitive::from_u64(n).map(|x: T| F::cst(x))
    }
    #[inline]
    fn from_u128(n: u128) -> Option<Self> {
        FromPrimitive::from_u128(n).map(|x: T| F::cst(x))
    }
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        FromPrimitive::from_f32(n).map(|x: T| F::cst(x))
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        FromPrimitive::from_f64(n).map(|x: T| F::cst(x))
    }
}

impl<T> Zero for F<T>
where
    T: Float,
{
    #[inline]
    fn zero() -> F<T> {
        F {
            x: T::zero(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.x.is_zero()
    }
}

impl<T> One for F<T>
where
    T: Float,
{
    #[inline]
    fn one() -> F<T> {
        F {
            x: T::one(),
            dx: T::zero(),
        }
    }
}

impl<T> Num for F<T>
where
    T: Float + Num + ToPrimitive,
{
    type FromStrRadixErr = <T as Num>::FromStrRadixErr;

    //::num_traits::ParseFloatError;

    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(src, radix).map(|x| F::cst(x))
    }
}

impl<T> FloatConst for F<T>
where
    T: Float + FloatConst,
{
    #[inline]
    fn E() -> F<T> {
        F {
            x: T::E(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn FRAC_1_PI() -> F<T> {
        F {
            x: T::FRAC_1_PI(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn FRAC_1_SQRT_2() -> F<T> {
        F {
            x: T::FRAC_1_SQRT_2(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn FRAC_2_PI() -> F<T> {
        F {
            x: T::FRAC_2_PI(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn FRAC_2_SQRT_PI() -> F<T> {
        F {
            x: T::FRAC_2_SQRT_PI(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn FRAC_PI_2() -> F<T> {
        F {
            x: T::FRAC_PI_2(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn FRAC_PI_3() -> F<T> {
        F {
            x: T::FRAC_PI_3(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn FRAC_PI_4() -> F<T> {
        F {
            x: T::FRAC_PI_4(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn FRAC_PI_6() -> F<T> {
        F {
            x: T::FRAC_PI_6(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn FRAC_PI_8() -> F<T> {
        F {
            x: T::FRAC_PI_8(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn LN_10() -> F<T> {
        F {
            x: T::LN_10(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn LN_2() -> F<T> {
        F {
            x: T::LN_2(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn LOG10_E() -> F<T> {
        F {
            x: T::LOG10_E(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn LOG2_E() -> F<T> {
        F {
            x: T::LOG2_E(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn PI() -> F<T> {
        F {
            x: T::PI(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn SQRT_2() -> F<T> {
        F {
            x: T::SQRT_2(),
            dx: T::zero(),
        }
    }
}

impl<T> Float for F<T>
where
    T: Float,
{
    #[inline]
    fn nan() -> F<T> {
        F {
            x: T::nan(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn infinity() -> F<T> {
        F {
            x: T::infinity(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn neg_infinity() -> F<T> {
        F {
            x: T::neg_infinity(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn neg_zero() -> F<T> {
        F {
            x: T::neg_zero(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn min_value() -> F<T> {
        F {
            x: T::min_value(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn min_positive_value() -> F<T> {
        F {
            x: T::min_positive_value(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn max_value() -> F<T> {
        F {
            x: T::max_value(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn is_nan(self) -> bool {
        self.x.is_nan() || self.dx.is_nan()
    }
    #[inline]
    fn is_infinite(self) -> bool {
        self.x.is_infinite() || self.dx.is_infinite()
    }
    #[inline]
    fn is_finite(self) -> bool {
        self.x.is_finite() && self.dx.is_finite()
    }
    #[inline]
    fn is_normal(self) -> bool {
        self.x.is_normal() && self.dx.is_normal()
    }
    #[inline]
    fn classify(self) -> FpCategory {
        self.x.classify()
    }

    #[inline]
    fn floor(self) -> F<T> {
        F {
            x: self.x.floor(),
            dx: self.dx,
        }
    }
    #[inline]
    fn ceil(self) -> F<T> {
        F {
            x: self.x.ceil(),
            dx: self.dx,
        }
    }
    #[inline]
    fn round(self) -> F<T> {
        F {
            x: self.x.round(),
            dx: self.dx,
        }
    }
    #[inline]
    fn trunc(self) -> F<T> {
        F {
            x: self.x.trunc(),
            dx: self.dx,
        }
    }
    #[inline]
    fn fract(self) -> F<T> {
        F {
            x: self.x.fract(),
            dx: self.dx,
        }
    }
    #[inline]
    fn abs(self) -> F<T> {
        F {
            x: self.x.abs(),
            dx: if self.x >= T::zero() {
                self.dx
            } else {
                -self.dx
            },
        }
    }
    #[inline]
    fn signum(self) -> F<T> {
        F {
            x: self.x.signum(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn is_sign_positive(self) -> bool {
        self.x.is_sign_positive()
    }
    #[inline]
    fn is_sign_negative(self) -> bool {
        self.x.is_sign_negative()
    }
    #[inline]
    fn mul_add(self, a: F<T>, b: F<T>) -> F<T> {
        self * a + b
    }
    #[inline]
    fn recip(self) -> F<T> {
        F {
            x: self.x.recip(),
            dx: -self.dx / (self.x * self.x),
        }
    }
    #[inline]
    fn powi(self, n: i32) -> F<T> {
        F {
            x: self.x.powi(n),
            dx: self.dx * T::from(n).unwrap() * self.x.powi(n - 1),
        }
    }
    #[inline]
    fn powf(self, n: F<T>) -> F<T> {
        F {
            x: Float::powf(self.x, n.x),
            dx: (Float::ln(self.x) * n.dx + n.x * self.dx / self.x) * Float::powf(self.x, n.x),
        }
    }
    #[inline]
    fn sqrt(self) -> F<T> {
        F {
            x: self.x.sqrt(),
            dx: self.dx / (T::one() + T::one()) / self.x.sqrt(),
        }
    }

    #[inline]
    fn exp(self) -> F<T> {
        F {
            x: Float::exp(self.x),
            dx: self.dx * Float::exp(self.x),
        }
    }
    #[inline]
    fn exp2(self) -> F<T> {
        F {
            x: Float::exp2(self.x),
            dx: self.dx * Float::ln(T::one() + T::one()) * Float::exp(self.x),
        }
    }
    #[inline]
    fn ln(self) -> F<T> {
        F {
            x: Float::ln(self.x),
            dx: self.dx * self.x.recip(),
        }
    }
    #[inline]
    fn log(self, b: F<T>) -> F<T> {
        F {
            x: Float::log(self.x, b.x),
            dx: -Float::ln(self.x) * b.dx / (b.x * Float::powi(Float::ln(b.x), 2))
                + self.dx / (self.x * Float::ln(b.x)),
        }
    }
    #[inline]
    fn log2(self) -> F<T> {
        Float::log(
            self,
            F {
                x: T::one() + T::one(),
                dx: T::zero(),
            },
        )
    }
    #[inline]
    fn log10(self) -> F<T> {
        Float::log(
            self,
            F {
                x: T::from(10.0).unwrap(),
                dx: T::zero(),
            },
        )
    }
    #[inline]
    fn max(self, other: F<T>) -> F<T> {
        if self.x < other.x {
            other
        } else {
            self
        }
    }
    #[inline]
    fn min(self, other: F<T>) -> F<T> {
        if self.x > other.x {
            other
        } else {
            self
        }
    }
    #[inline]
    fn abs_sub(self, other: F<T>) -> F<T> {
        if self > other {
            F {
                x: Float::abs_sub(self.x, other.x),
                dx: (self - other).dx,
            }
        } else {
            F {
                x: T::zero(),
                dx: T::zero(),
            }
        }
    }
    #[inline]
    fn cbrt(self) -> F<T> {
        F {
            x: Float::cbrt(self.x),
            dx: T::from(1.0 / 3.0).unwrap() * self.x.powf(T::from(-2.0 / 3.0).unwrap()) * self.dx,
        }
    }
    #[inline]
    fn hypot(self, other: F<T>) -> F<T> {
        Float::sqrt(Float::powi(self, 2) + Float::powi(other, 2))
    }
    #[inline]
    fn sin(self) -> F<T> {
        F {
            x: Float::sin(self.x),
            dx: self.dx * Float::cos(self.x),
        }
    }
    #[inline]
    fn cos(self) -> F<T> {
        F {
            x: Float::cos(self.x),
            dx: -self.dx * Float::sin(self.x),
        }
    }
    #[inline]
    fn tan(self) -> F<T> {
        let t = Float::tan(self.x);
        F {
            x: t,
            dx: self.dx * (t * t + T::one()),
        }
    }
    #[inline]
    fn asin(self) -> F<T> {
        F {
            x: Float::asin(self.x),
            dx: self.dx / Float::sqrt(T::one() - Float::powi(self.x, 2)),
        }
    }
    #[inline]
    fn acos(self) -> F<T> {
        F {
            x: Float::acos(self.x),
            dx: -self.dx / Float::sqrt(T::one() - Float::powi(self.x, 2)),
        }
    }
    #[inline]
    fn atan(self) -> F<T> {
        F {
            x: Float::atan(self.x),
            dx: self.dx / Float::sqrt(Float::powi(self.x, 2) + T::one()),
        }
    }
    #[inline]
    fn atan2(self, other: F<T>) -> F<T> {
        F {
            x: Float::atan2(self.x, other.x),
            dx: (other.x * self.dx - self.x * other.dx)
                / (Float::powi(self.x, 2) + Float::powi(other.x, 2)),
        }
    }
    #[inline]
    fn sin_cos(self) -> (F<T>, F<T>) {
        let (s, c) = Float::sin_cos(self.x);
        let sn = F {
            x: s,
            dx: self.dx * c,
        };
        let cn = F {
            x: c,
            dx: -self.dx * s,
        };
        (sn, cn)
    }
    #[inline]
    fn exp_m1(self) -> F<T> {
        F {
            x: Float::exp_m1(self.x),
            dx: self.dx * Float::exp(self.x),
        }
    }
    #[inline]
    fn ln_1p(self) -> F<T> {
        F {
            x: Float::ln_1p(self.x),
            dx: self.dx / (self.x + T::one()),
        }
    }
    #[inline]
    fn sinh(self) -> F<T> {
        F {
            x: Float::sinh(self.x),
            dx: self.dx * Float::cosh(self.x),
        }
    }
    #[inline]
    fn cosh(self) -> F<T> {
        F {
            x: Float::cosh(self.x),
            dx: self.dx * Float::sinh(self.x),
        }
    }
    #[inline]
    fn tanh(self) -> F<T> {
        F {
            x: Float::tanh(self.x),
            dx: self.dx * (T::one() - Float::powi(Float::tanh(self.x), 2)),
        }
    }
    #[inline]
    fn asinh(self) -> F<T> {
        F {
            x: Float::asinh(self.x),
            dx: self.dx * (Float::powi(self.x, 2) + T::one()),
        }
    }
    #[inline]
    fn acosh(self) -> F<T> {
        F {
            x: Float::acosh(self.x),
            dx: self.dx * (Float::powi(self.x, 2) - T::one()),
        }
    }
    #[inline]
    fn atanh(self) -> F<T> {
        F {
            x: Float::atanh(self.x),
            dx: self.dx * (-Float::powi(self.x, 2) + T::one()),
        }
    }
    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        self.x.integer_decode()
    }

    #[inline]
    fn epsilon() -> F<T> {
        F {
            x: T::epsilon(),
            dx: T::zero(),
        }
    }
    #[inline]
    fn to_degrees(self) -> F<T> {
        F {
            x: Float::to_degrees(self.x),
            dx: T::zero(),
        }
    }
    #[inline]
    fn to_radians(self) -> F<T> {
        F {
            x: Float::to_radians(self.x),
            dx: T::zero(),
        }
    }
}

impl<T> std::iter::Sum for F<T>
where
    T: Float + AddAssign,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        let mut res = Self::zero();
        for x in iter {
            res += x;
        }
        res
    }
}

impl<T> std::iter::Sum<T> for F<T>
where
    T: Float + AddAssign,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = T>,
    {
        iter.map(|x| F::cst(x)).sum()
    }
}

impl<T> F<T>
where
    T: Float,
{
    /// Create a new constant. Use this also to convert from a variable to a constant.
    /// This constructor panics if `x` cannot be converted to `f64`.
    #[inline]
    pub fn cst<U: ToPrimitive>(x: U) -> F<T> {
        F {
            x: T::from(x).unwrap(),
            dx: T::zero(),
        }
    }

    /// Create a new variable. Use this also to convert from a constant to a variable.
    /// This constructor panics if `x` cannot be converted to `f64`.
    #[inline]
    pub fn var<U: ToPrimitive>(x: U) -> F<T> {
        F {
            x: T::from(x).unwrap(),
            dx: T::one(),
        }
    }

    /// Compare two `F`s in full, including the derivative part.
    pub fn full_eq(&self, rhs: &F<T>) -> bool {
        self.x == rhs.x && self.dx == rhs.dx
    }

    /// Get the value of this variable.
    #[inline]
    pub fn value(&self) -> T {
        self.x
    }

    /// Get the current derivative of this variable. This will be zero if this `F` is a
    /// constant.
    #[inline]
    pub fn deriv(&self) -> T {
        self.dx
    }
}

pub fn diff<G, T>(f: G, x0: T) -> T
where
    G: FnOnce(F<T>) -> F<T>,
    T: Float,
{
    f(F::var(x0)).deriv()
}

pub fn grad<G, T>(f: G, x0: &[T]) -> Vec<T>
where
    G: Fn(&[F<T>]) -> F<T>,
    T: Float,
{
    let mut nums: Vec<F<T>> = x0.iter().map(|&x| F::cst(x)).collect();

    let mut results = Vec::new();

    for i in 0..nums.len() {
        nums[i] = F::var(nums[i]);
        results.push(f(&nums).deriv());
        nums[i] = F::cst(nums[i]);
    }

    results
}

pub fn eval<G, T>(f: G, x0: &[T]) -> T
where
    G: Fn(&[F<T>]) -> F<T>,
    T: Float,
{
    let nums: Vec<F<T>> = x0.iter().map(|&x| F::cst(x)).collect();
    f(&nums).x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience macro for comparing `F`s in full.
    macro_rules! assert_full_eq {
        ($x:expr, $y:expr) => {
            assert!(F::full_eq(&$x, &$y));
        };
    }

    #[test]
    fn basic_arithmetic_test() {
        // Test basic arithmetic on F.
        let mut x = F::var(1.0);
        let y = F::var(2.0);

        assert_full_eq!(-x, F { x: -1.0, dx: -1.0 }); // negation

        assert_full_eq!(x + y, F { x: 3.0, dx: 2.0 }); // addition
        assert_full_eq!(x + 2.0, F { x: 3.0, dx: 1.0 }); // addition
                                                         //assert_full_eq!(2.0 + x, F { x: 3.0, dx: 1.0 }); // addition
        x += y;
        assert_full_eq!(x, F { x: 3.0, dx: 2.0 }); // assign add
        x += 1.0;
        assert_full_eq!(x, F { x: 4.0, dx: 2.0 }); // assign add

        assert_full_eq!(x - y, F { x: 2.0, dx: 1.0 }); // subtraction
        assert_full_eq!(x - 1.0, F { x: 3.0, dx: 2.0 }); // subtraction
                                                         //assert_full_eq!(1.0 - x, F { x: -3.0, dx: -2.0 }); // subtraction
        x -= y;
        assert_full_eq!(x, F { x: 2.0, dx: 1.0 }); // subtract assign
        x -= 1.0;
        assert_full_eq!(x, F { x: 1.0, dx: 1.0 }); // subtract assign

        assert_full_eq!(x * y, F { x: 2.0, dx: 3.0 }); // multiplication
        assert_full_eq!(x * 2.0, F { x: 2.0, dx: 2.0 }); // multiplication
                                                         //assert_full_eq!(2.0 * x, F { x: 2.0, dx: 2.0 }); // multiplication
        x *= y;
        assert_full_eq!(x, F { x: 2.0, dx: 3.0 }); // multiply assign
        x *= 2.0;
        assert_full_eq!(x, F { x: 4.0, dx: 6.0 }); // multiply assign

        assert_full_eq!(x / y, F { x: 2.0, dx: 2.0 }); // division
        assert_full_eq!(x / 2.0, F { x: 2.0, dx: 3.0 }); // division
                                                         //assert_full_eq!(2.0 / x, F { x: 0.5, dx: -0.75 }); // division
        x /= y;
        assert_full_eq!(x, F { x: 2.0, dx: 2.0 }); // divide assign
        x /= 2.0;
        assert_full_eq!(x, F { x: 1.0, dx: 1.0 }); // divide assign

        assert_full_eq!(x % y, F { x: 1.0, dx: 1.0 }); // mod
        assert_full_eq!(x % 2.0, F { x: 1.0, dx: 1.0 }); // mod
                                                         //assert_full_eq!(2.0 % x, F { x: 0.0, dx: -2.0 }); // mod
        x %= y;
        assert_full_eq!(x, F { x: 1.0, dx: 1.0 }); // mod assign
        x %= 2.0;
        assert_full_eq!(x, F { x: 1.0, dx: 1.0 }); // mod assign
    }

    // Test the min and max functions
    #[test]
    fn min_max_test() {
        // Test basic arithmetic on F.
        let a = F::var(1.0);
        let mut b = F::cst(2.0);

        b = b.min(a);
        assert_full_eq!(b, F { x: 1.0, dx: 1.0 });

        b = F::cst(2.0);
        b = a.min(b);
        assert_full_eq!(b, F { x: 1.0, dx: 1.0 });

        let b = F::cst(2.0);

        let c = a.max(b);
        assert_full_eq!(c, F { x: 2.0, dx: 0.0 });

        // Make sure that our min and max are consistent with the internal implementation to avoid
        // inconsistencies in the future. In particular we look at tie breaking.

        let b = F::cst(1.0);
        let minf = a.x.min(b.x);
        assert_full_eq!(
            a.min(b),
            F {
                x: minf,
                dx: if minf == a.x { a.dx } else { b.dx }
            }
        );

        let maxf = a.x.max(b.x);
        assert_full_eq!(
            a.max(b),
            F {
                x: maxf,
                dx: if maxf == a.x { a.dx } else { b.dx }
            }
        );
    }

    // Test iterator sum
    #[test]
    fn sum_test() {
        let v = vec![1.0, 2.0].into_iter();
        let ad_v = vec![F::var(1.0), F::var(2.0)].into_iter();
        assert_full_eq!(ad_v.clone().sum(), F { x: 3.0, dx: 2.0 });
        assert_full_eq!(v.sum::<F<f64>>(), F { x: 3.0, dx: 0.0 });
    }
}
