use rand::Rng;
use rand::distributions::{Standard, Distribution};
use rand::distributions::range::SampleRange;
use num_traits::float::{Float, FloatConst};
use super::super::coordinates::SphCoord;

pub fn rand<T, U>(rng: &mut U) -> SphCoord<T>
where
    T: Float + FloatConst + SampleRange,
    Standard:Distribution<T>,
    U: Rng,
{
    //gen pol:
    let two = T::one() + T::one();
    let x = rng.gen_range(T::zero(), two);
    let pol = (T::one() - x).acos();
    let az = rng.gen_range(T::zero(), two * T::PI());
    SphCoord::new(pol, az)
}
