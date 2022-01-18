#![allow(clippy::many_single_char_names)]
use crate::coordinates::{SphCoord, Vec2d};
use num::traits::{float::Float, FloatConst};
use std::fmt::Debug;

/*
https://en.wikipedia.org/wiki/Mollweide_projection
*/

pub fn regulate_az<T>(mut az: T) -> T
where
    T: Float + FloatConst,
{
    let pi = T::PI();
    let two = T::one() + T::one();
    let twopi = two * pi;
    while az > pi {
        az = az - twopi;
    }
    while az <= -pi {
        az = az + twopi;
    }
    az
}

fn theta_iter<T>(theta: T, pol: T) -> T
where
    T: Float + FloatConst,
{
    let two: T = T::one() + T::one();
    theta
        - (two * theta + (two * theta).sin() - T::PI() * pol.cos())
            / (two + two * (two * theta).cos())
}

fn get_r<T>() -> T
where
    T: Float + FloatConst,
{
    let two = T::one() + T::one();
    T::FRAC_1_SQRT_2() / two
}

fn solve_theta<T>(pol: T) -> T
where
    T: Float + FloatConst + Debug,
{
    let mut theta0 = T::FRAC_PI_2() - pol;
    for _i in 0..1000 {
        let theta1 = theta_iter(theta0, pol);

        if theta1.is_nan() {
            return theta0;
        }
        //println!("{:?} {:?} {:?}", (theta1-theta0).abs(), theta0, pol);
        if (theta1 - theta0).abs() <= T::epsilon() {
            return theta1;
        } else {
            theta0 = theta1;
        }
    }
    theta0
}

pub fn proj<T>(p: SphCoord<T>) -> Vec2d<T>
where
    T: Float + FloatConst + Debug,
{
    let r: T = get_r();
    let lambda = regulate_az(p.az);
    let theta = solve_theta(p.pol);
    let two = T::one() + T::one();
    let x = r * two * T::SQRT_2() / T::PI() * lambda * theta.cos();
    let y = r * T::SQRT_2() * theta.sin();
    Vec2d::new(x, y)
}

pub fn iproj<T>(p: Vec2d<T>) -> Option<SphCoord<T>>
where
    T: Float + FloatConst + Debug,
{
    let two = T::one() + T::one();

    let x = p.x;
    let y = p.y;
    let r: T = get_r();

    if x * x + (y * y) * (two + two) > T::one() {
        return None;
    }

    let a = y / r / T::SQRT_2();
    let theta = a.asin();
    let phi = ((two * theta + (two * theta).sin()) / T::PI()).asin();
    let lambda = T::PI() * x / (two * r * T::SQRT_2() * theta.cos());
    Some(SphCoord::new(T::FRAC_PI_2() - phi, lambda))
}
