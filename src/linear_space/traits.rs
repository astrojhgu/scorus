use num_traits::float::Float;
use num_traits::Num;
use std::ops::{Add, Mul, Sub};

pub trait LinearSpace<Scalar>
where
    Scalar: Num,
    Self: Sized,
    for<'a> &'a Self: Add<Output = Self>,
    for<'a> &'a Self: Sub<Output = Self>,
    for<'a> &'a Self: Mul<Scalar, Output = Self>,
{
    fn dimension(&self) -> usize;
}

pub trait InnerProdSpace<Scalar>: LinearSpace<Scalar>
where
    Scalar: Num,
    Self: Sized,
    for<'a> &'a Self: Add<Output = Self>,
    for<'a> &'a Self: Sub<Output = Self>,
    for<'a> &'a Self: Mul<Scalar, Output = Self>,
{
    fn dot(&self, rhs: &Self) -> Scalar;
}

pub trait PDInnerProdSpace<Scalar>: InnerProdSpace<Scalar>
where
    Scalar: Float,
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

