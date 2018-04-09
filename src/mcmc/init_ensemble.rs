use rand::{Rand, Rng};
use rand::distributions::range::SampleRange;
use num_traits::NumCast;
use std::cmp::PartialOrd;
use std::fmt::Display;
use std::marker::Copy;
//use utils::Resizeable;
use std::ops::IndexMut;
use std::ops::Index;
//use num_traits::float::Float;
//use num_traits::NumCast;
//use num_traits::identities::one;
//use num_traits::identities::zero;
use super::super::utils::HasLen;

pub fn get_one_init_realization<U, T, R>(y1: &U, y2: &U, rng: &mut R) -> U
where
    U: HasLen + Clone + IndexMut<usize, Output = T> + Index<usize, Output = T>,
    T: NumCast + Rand + PartialOrd + SampleRange + Display + Copy,
    R: Rng,
{
    let mut result = y1.clone();

    for i in 0..result.len() {
        result[i] = rng.gen_range(y1[i], y2[i]);
    }
    result
}
