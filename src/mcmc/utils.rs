use std;

use rand::Rng;
use rand::distributions::Standard;
use rand::distributions::Distribution;
use rand::distributions::range::SampleRange;

use num_traits::float::Float;
use num_traits::identities::one;
use num_traits::identities::zero;
//use num_traits::NumCast;
use std::ops::IndexMut;
//use super::mcmc_errors::McmcErr;
use super::super::utils::HasLen;
//use super::super::utils::ItemSwapable;
//use super::super::utils::Resizeable;

/*
pub fn shuffle<T, U>(arr: &T, rng: &mut U) -> T
where
    T: HasLen + Clone + ItemSwapable,
    U: Rng,
{
    let mut x = arr.clone();
    let l = arr.len();
    for i in (1..l).rev() {
        let i1 = rng.gen_range(0, i + 1);
        x.swap_items(i, i1);
    }
    x
}
*/

pub fn draw_z<T, U>(rng: &mut U, a: T) -> T
where
    T: Float + std::cmp::PartialOrd + SampleRange,
    Standard:Distribution<T>,
    U: Rng,
{
    let sqrt_a: T = a.sqrt();
    let unit: T = one();
    let two = unit + unit;
    let p: T = rng.gen_range(zero(), two * (sqrt_a - unit / sqrt_a));
    let y: T = unit / sqrt_a + p / (two);
    y * y
}

pub fn scale_vec<T, U>(x1: &U, x2: &U, z: T) -> U
where
    T: Float,
    U: Clone + IndexMut<usize, Output = T> + HasLen,
{
    let mut result = (*x1).clone();
    let unit: T = one();
    for l in 0..x1.len() {
        result[l] = z * x1[l] + (unit - z) * x2[l];
    }

    result
}
