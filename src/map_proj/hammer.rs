use super::super::coordinates::{SphCoord, Vec2d};
use num_traits::float::Float;
use num_traits::float::FloatConst;

pub fn proj<T>(p: SphCoord<T>) -> Vec2d<T>
where
    T: Float + FloatConst,
{
    let pi = T::PI();
    let two_pi = T::from(2).unwrap() * pi;
    //let two_sqrt_2 = T::from(2).unwrap() * T::SQRT_2();
    let two = T::from(2).unwrap();
    let mut ra = p.az;
    let dec = T::FRAC_PI_2() - p.pol;
    while ra > pi {
        ra = ra - two_pi;
    }
    while ra <= -pi {
        ra = ra + two_pi;
    }

    #[allow(non_snake_case)]
    let X = (T::one() + dec.cos() * (ra / two).cos()).sqrt();

    let x = dec.cos() * (ra / two).sin() / X;
    let y = dec.sin() / X / two;
    Vec2d::new(x, y)
}

pub fn iproj<T>(p: Vec2d<T>) -> Option<SphCoord<T>>
where
    T: Float + FloatConst,
{
    let pi = T::PI();
    let two_pi = T::from(2).unwrap() * pi;
    let two_sqrt_2 = T::from(2).unwrap() * T::SQRT_2();
    let two = T::one() + T::one();
    let x = p.x * two_sqrt_2;
    let y = p.y * two_sqrt_2;

    let z2 = T::one() - (x * x) / T::from(16).unwrap() - (y * y) / T::from(4).unwrap();
    if z2 < T::from(0.5).unwrap() {
        return None;
    }

    let z = z2.sqrt();
    let mut ra = two * (z * x / (two * (two * z2 - T::one()))).atan();
    let dec = (z * y).asin();
    while ra < T::zero() {
        ra = ra + two_pi;
    }
    while ra > two_pi {
        ra = ra - two_pi;
    }
    Some(SphCoord::new(T::FRAC_PI_2() - dec, ra))
}
