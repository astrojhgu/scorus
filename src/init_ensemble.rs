extern crate num_traits;
extern crate rand;
extern crate std;
use utils::HasLength;
//use utils::Resizeable;
use std::ops::IndexMut;
use std::ops::Index;
//use num_traits::float::Float;
//use num_traits::NumCast;
//use num_traits::identities::one;
//use num_traits::identities::zero;

pub fn get_one_init_realization<U, T, R>(y1: &U, y2: &U, rng: &mut R) -> U
where
    U: HasLength + Clone + IndexMut<usize, Output = T> + Index<usize, Output = T>,
    T: num_traits::NumCast
        + rand::Rand
        + std::cmp::PartialOrd
        + rand::distributions::range::SampleRange
        + std::marker::Sync
        + std::marker::Send
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
