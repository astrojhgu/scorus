use crate::coordinates::SphCoord;
use num_traits::float::{Float, FloatConst};
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

pub fn rand<T, U>(rng: &mut U) -> SphCoord<T>
where
    T: Float + FloatConst + SampleUniform,
    Standard: Distribution<T>,
    U: Rng,
{
    //gen pol:
    let two = T::one() + T::one();
    let x = rng.gen_range(T::zero(), two);
    let pol = (T::one() - x).acos();
    let az = rng.gen_range(T::zero(), two * T::PI());
    SphCoord::new(pol, az)
}
