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

#[cfg(test)]
mod tests {
    #[derive(Clone)]
    struct MyVec(Vec<f64>);

    impl std::ops::Add for &MyVec {
        type Output = MyVec;

        fn add(self, rhs: &MyVec) -> MyVec {
            MyVec(
                self.0
                    .iter()
                    .zip(rhs.0.iter())
                    .map(|(a, b)| a + b)
                    .collect(),
            )
        }
    }

    impl<'a> std::ops::Sub for &'a MyVec {
        type Output = MyVec;

        fn sub(self, rhs: &'a MyVec) -> MyVec {
            MyVec(
                self.0
                    .iter()
                    .zip(rhs.0.iter())
                    .map(|(a, b)| a - b)
                    .collect(),
            )
        }
    }

    impl std::ops::Mul<f64> for &MyVec {
        type Output = MyVec;

        fn mul(self, rhs: f64) -> MyVec {
            MyVec(self.0.iter().map(|a| a * rhs).collect())
        }
    }

    impl<'a> super::LinearSpace<'a, f64> for MyVec {}

    impl<'a> super::InnerProdSpace<'a, f64> for MyVec {
        fn dot(&self, rhs: &Self) -> f64 {
            self.0.iter().zip(rhs.0.iter()).map(|(a, b)| a * b).sum()
        }
    }

    impl<'a> super::PDInnerProdSpace<'a, f64> for MyVec {}

    fn add<'a, T>(a: &'a T) -> f64
    where
        T: super::PDInnerProdSpace<'a, f64>,
        &'a T:
            std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<f64, Output = T>,
    {
        a.distance_to(&a)
    }

    #[test]
    fn it_works() {
        let a: i32 = (&1) + (&1);
        assert_eq!(add(&MyVec(vec![1.2, 1.2])), 0.0);
    }
}
