#![cfg(not(target_family = "wasm"))]
#![allow(clippy::needless_range_loop)]
//use ::std;
use num::traits::{
    float::Float,
    identities::{one, zero},
};
use special::Gamma;
use std::fmt::Display;
use std::marker::Copy;
use std::ops::IndexMut;

use crate::utils::HasLen;

pub fn ln_factorial<T>(x: T) -> T
where
    T: Float + Copy + Gamma,
{
    (x + one()).ln_gamma().0
}

pub fn coefficient<T>(n: usize, k: usize) -> T
where
    T: Float + Copy + Gamma,
{
    if k > n {
        return zero();
    }
    let m = (n - k) / 2;
    if m * 2 != (n - k) {
        return zero();
    }

    (if m % 2 == 0 { one::<T>() } else { -one::<T>() })
        * (ln_factorial(T::from(2 * n - 2 * m).unwrap())
            - ln_factorial(T::from(m).unwrap())
            - ln_factorial(T::from(n - m).unwrap())
            - ln_factorial(T::from(n - 2 * m).unwrap())
            - T::from(n).unwrap() * T::from(2).unwrap().ln())
        .exp()
}

pub fn legendre2poly<T, V>(p: &V) -> V
where
    T: Float + Copy + Gamma + Display,
    V: Clone + IndexMut<usize, Output = T> + HasLen,
{
    let mut result = p.clone();
    for i in 0..result.len() {
        result[i] = zero();
    }
    let n = p.len();
    for i in 0..n {
        //println!("{}", p[i]);
        for j in 0..=i {
            //println!("{} {} {}", i,j,p[i]*coefficient::<T>(i,j));
            result[j] = result[j] + p[i] * coefficient(i, j);
        }
    }
    result
}

fn next_p<T>(n: usize, x: T, pn1: T, pn2: T) -> T
where
    T: Float + Copy,
{
    (T::from(2 * n - 1).unwrap() * x * pn1 - T::from(n - 1).unwrap() * pn2) / T::from(n).unwrap()
}

pub fn eval<T>(n: usize, x: T) -> T
where
    T: Float + Copy,
{
    let one = one::<T>();

    if n == 0 {
        return one;
    }
    if n == 1 {
        return x;
    }

    let mut pn1 = x;
    let mut pn2 = one;

    let mut n1 = 2_usize;
    loop {
        let pn = next_p(n1, x, pn1, pn2);
        if n1 == n {
            return pn;
        }
        pn2 = pn1;
        pn1 = pn;
        n1 += 1;
    }
}

pub fn eval_all_orders<T>(n: usize, x: T) -> Vec<T>
where
    T: Float + Copy,
{
    let one = one::<T>();

    if n == 0 {
        return vec![one];
    }
    if n == 1 {
        return vec![one, x];
    }

    let mut result = vec![zero::<T>(); n + 1];
    result[0] = one;
    result[1] = x;

    let mut pn1 = x;
    let mut pn2 = one;

    let mut n1 = 2_usize;
    loop {
        let pn = next_p(n1, x, pn1, pn2);
        result[n1] = pn;
        if n1 == n {
            return result;
        }
        pn2 = pn1;
        pn1 = pn;
        n1 += 1;
    }
}
