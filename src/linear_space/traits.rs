#![allow(clippy::eq_op)]
use num::{
    traits::{
        float::Float
        , Num
    }
};
use std::ops::{Add, Mul, Sub};

pub trait LinearSpace<Scalar>
where
    Scalar: Num + Copy,
    Self: Sized,
    for<'a> &'a Self: Add<Output = Self>,
    for<'a> &'a Self: Sub<Output = Self>,
    for<'a> &'a Self: Mul<Scalar, Output = Self>,
{
    fn dimension(&self) -> usize;
    fn zeros_like(rhs: &Self) -> Self {
        rhs - rhs
    }
}

pub trait IndexableLinearSpace<Scalar>:
    LinearSpace<Scalar>
    + std::ops::Index<usize, Output = Scalar>
    + std::ops::IndexMut<usize, Output = Scalar>
where
    Scalar: Num + Copy,
    Self: Sized,
    for<'a> &'a Self: Add<Output = Self>,
    for<'a> &'a Self: Sub<Output = Self>,
    for<'a> &'a Self: Mul<Scalar, Output = Self>,
{
    fn element_wise_prod(&self, rhs: &Self) -> Self {
        let mut result = self - self;
        for i in 0..self.dimension() {
            result[i] = result[i] * rhs[i]
        }
        result
    }
}

pub trait InnerProdSpace<Scalar>: IndexableLinearSpace<Scalar>
where
    Scalar: Num + Copy,
    Self: Sized,
    for<'a> &'a Self: Add<Output = Self>,
    for<'a> &'a Self: Sub<Output = Self>,
    for<'a> &'a Self: Mul<Scalar, Output = Self>,
{
    fn dot(&self, rhs: &Self) -> Scalar;
}

pub trait PDInnerProdSpace<Scalar>: InnerProdSpace<Scalar>
where
    Scalar: Float + Copy,
    Self: Sized,
    for<'a> &'a Self: Add<Output = Self>,
    for<'a> &'a Self: Sub<Output = Self>,
    for<'a> &'a Self: Mul<Scalar, Output = Self>,
{
    fn distance_to(&self, rhs: &Self) -> Scalar {
        let d = self - rhs;
        d.dot(&d).sqrt()
        //self.dot(rhs).sqrt()
    }
}
