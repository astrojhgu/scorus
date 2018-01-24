extern crate num_traits;
extern crate rand;

use std;
//use utils::Resizeable;
use std::ops::IndexMut;
use std::ops::Index;
//use num_traits::float::Float;
//use num_traits::NumCast;
//use num_traits::identities::one;
//use num_traits::identities::zero;
use super::super::utils::HasLength;

pub fn get_one_init_realization<U, T, R>(y1: &U, y2: &U, rng: &mut R) -> U
where
    U: HasLength + Clone + IndexMut<usize, Output = T> + Index<usize, Output = T>,
    T: num_traits::NumCast
        + rand::Rand
        + std::cmp::PartialOrd
        + rand::distributions::range::SampleRange
        + std::fmt::Display
        + std::marker::Copy,
    R: rand::Rng,
{
    let mut result = y1.clone();

    for i in 0..result.length() {
        result[i] = rng.gen_range(y1[i], y2[i]);
    }
    result
}
