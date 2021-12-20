use crate::coordinates::SphCoord;
use num::traits::float::{Float, FloatConst};
use rand::{
    Rng
    ,distributions::{
        uniform::SampleUniform
        ,Distribution
        , Standard
        , Uniform
    }
};

pub fn rand<T, U>(rng: &mut U) -> SphCoord<T>
where
    T: Float + FloatConst + SampleUniform,
    Standard: Distribution<T>,
    U: Rng,
{
    //gen pol:
    let two = T::one() + T::one();
    let x = rng.sample(Uniform::new(T::zero(), two));
    let pol = (T::one() - x).acos();
    let az = rng.sample(Uniform::new(T::zero(), two * T::PI()));
    SphCoord::new(pol, az)
}
