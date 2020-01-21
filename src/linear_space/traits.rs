use num_traits::float::Float;
use num_traits::Num;
use std::ops::{Add, Mul, Sub};

pub trait LinearSpace<Scalar>
where
    Scalar: Num+Copy,
    Self: Sized,
    for<'a> &'a Self: Add<Output = Self>,
    for<'a> &'a Self: Sub<Output = Self>,
    for<'a> &'a Self: Mul<Scalar, Output = Self>,
{
    fn dimension(&self) -> usize;
}

pub trait FiniteLinearSpace<Scalar>: LinearSpace<Scalar>+std::ops::Index<usize, Output=Scalar>+std::ops::IndexMut<usize, Output=Scalar>
where Scalar: Num+Copy,
Self: Sized,
for<'a> &'a Self: Add<Output = Self>,
for<'a> &'a Self: Sub<Output = Self>,
for<'a> &'a Self: Mul<Scalar, Output = Self>,
{
    fn element_wise_prod(&self, rhs: &Self)->Self{
        let mut result=self-self;
        for i in 0..self.dimension(){
            result[i]=result[i]*rhs[i]
        }
        result
    }
}


pub trait InnerProdSpace<Scalar>: LinearSpace<Scalar>
where
    Scalar: Num+Copy,
    Self: Sized,
    for<'a> &'a Self: Add<Output = Self>,
    for<'a> &'a Self: Sub<Output = Self>,
    for<'a> &'a Self: Mul<Scalar, Output = Self>,
{
    fn dot(&self, rhs: &Self) -> Scalar;
}

pub trait PDInnerProdSpace<Scalar>: InnerProdSpace<Scalar>
where
    Scalar: Float+Copy,
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
