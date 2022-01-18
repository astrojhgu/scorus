use special::Gamma;

use num::traits::{
    float::Float,
    identities::{one, zero},
};
use std::ops::IndexMut;

use crate::basic::log_cn;
use crate::utils::types::InitFromLen;

pub fn log_bernstein_base1<T>(x: T, n: usize, nu: usize) -> T
where
    T: Gamma + Float,
{
    if x == zero() || x == one() {
        T::neg_infinity()
    } else {
        log_cn(T::from(n).unwrap(), T::from(nu).unwrap())
            + T::from(nu).unwrap() * x.ln()
            + T::from(n - nu).unwrap() * (one::<T>() - x).ln()
    }
}

pub fn bernstein_base<V, T>(x: &V, n: usize, nu: usize) -> V
where
    V: InitFromLen<ElmType = T> + IndexMut<usize, Output = T>,
    T: Gamma + Float + std::fmt::Debug,
{
    let mut result = V::init(x.len());
    let c = log_cn(T::from(n).unwrap(), T::from(nu).unwrap());
    for i in 0..x.len() {
        result[i] = if x[i] <= zero() || x[i] >= one() {
            zero()
        } else {
            (c + T::from(nu).unwrap() * x[i].ln()
                + T::from(n - nu).unwrap() * (one::<T>() - x[i]).ln())
            .exp()
        };
    }
    result
}

pub fn bernstein_poly<V, W, T>(x: &V, p: &W) -> V
where
    V: InitFromLen<ElmType = T> + IndexMut<usize, Output = T>,
    W: InitFromLen<ElmType = T> + IndexMut<usize, Output = T>,
    T: Gamma + Float + std::fmt::Debug,
{
    let n = p.len() - 1;
    //let mut result=V::init(x.len());
    (0..=n)
        .map(|nu| (p[nu], bernstein_base(x, n, nu)))
        .fold(V::init(x.len()), |a, (p1, b)| {
            let mut result = V::init(x.len());
            for i in 0..x.len() {
                result[i] = a[i] + p1 * b[i];
            }
            //eprintln!("{:?}", b[0]);
            result
        })
}
