use crate::utils::HasLen;
use num_traits::float::Float;
use num_traits::identities::zero;
use std::marker::Copy;
use std::ops::IndexMut;

pub fn eval<T, V>(x: T, p: &V) -> T
where
    T: Float + Copy,
    V: Clone + IndexMut<usize, Output = T> + HasLen,
{
    let mut result = zero::<T>();
    for i in 0..(*p).len() {
        result = result + p[i] * x.powi(i as i32);
    }
    result
}
