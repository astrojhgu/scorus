use num_traits::NumCast;
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use std::cmp::PartialOrd;
use std::fmt::Display;
use std::marker::Copy;
//use utils::Resizeable;
use std::ops::Index;
use std::ops::IndexMut;
//use num_traits::float::Float;
//use num_traits::NumCast;
//use num_traits::identities::one;
//use num_traits::identities::zero;
use crate::utils::HasLen;

pub fn get_one_init_realization<U, T, R>(y1: &U, y2: &U, rng: &mut R) -> U
where
    U: HasLen + Clone + IndexMut<usize, Output = T> + Index<usize, Output = T>,
    T: NumCast + PartialOrd + SampleUniform + Display + Copy,
    Standard: Distribution<T>,
    R: Rng,
{
    let mut result = y1.clone();

    for i in 0..result.len() {
        result[i] = rng.gen_range(y1[i], y2[i]);
    }
    result
}
