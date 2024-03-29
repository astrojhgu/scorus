use crate::linear_space::LinearSpace;
use num::traits::{
    float::Float,
    identities::{one, zero},
    NumCast,
};

use rand::{
    distributions::{uniform::SampleUniform, Distribution, Standard, Uniform},
    seq::SliceRandom,
    Rng,
};
use std::ops::{Add, Mul, Sub};

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
    T: Float + std::cmp::PartialOrd + SampleUniform,
    Standard: Distribution<T>,
    U: Rng,
{
    let sqrt_a: T = a.sqrt();
    let unit: T = one();
    let two = unit + unit;
    let p: T = rng.sample(Uniform::new(zero::<T>(), two * (sqrt_a - unit / sqrt_a)));
    let y: T = unit / sqrt_a + p / (two);
    y * y
}

pub fn scale_vec<T, V>(x1: &V, x2: &V, z: T) -> V
where
    T: Float,
    V: LinearSpace<T>,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    &(x1 * z) + &(x2 * (T::one() - z))
}

fn exchange_prob<T>(lp1: T, lp2: T, beta1: T, beta2: T) -> T
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
{
    let x = ((beta2 - beta1) * (-lp2 + lp1)).exp();

    if x > one::<T>() {
        one::<T>()
    } else {
        x
    }
}

pub fn swap_walkers<T, U, V>(ensemble: &mut [V], logprob: &mut [T], rng: &mut U, beta_list: &[T])
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    U: Rng,
    V: Clone + LinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    //let mut new_ensemble = ensemble_logprob.0.clone();
    //let mut new_logprob = ensemble_logprob.1.clone();
    let nbeta = beta_list.len();
    let nwalker_per_beta = ensemble.len() / nbeta;
    assert!(nbeta * nwalker_per_beta == ensemble.len());
    let mut jvec: Vec<usize> = (0..nwalker_per_beta).collect();
    if ensemble.len() == logprob.len() {
        for i in (1..nbeta).rev() {
            //println!("ibeta={}", i);
            let beta1 = beta_list[i];
            let beta2 = beta_list[i - 1];
            if beta1 > beta2 {
                panic!("beta list must be in decreasing order, with no duplicatation");
            }
            //rng.shuffle(&mut jvec);
            jvec.shuffle(rng);
            //let jvec=shuffle(&jvec, &mut rng);
            for (j2, &j1) in jvec.iter().enumerate() {
                let lp1 = logprob[i * nwalker_per_beta + j1];
                let lp2 = logprob[(i - 1) * nwalker_per_beta + j2];
                let ep = exchange_prob(lp1, lp2, beta1, beta2);
                //println!("{}",ep);
                let r: T = rng.sample(Uniform::new(T::zero(), T::one()));
                if r < ep {
                    ensemble.swap(i * nwalker_per_beta + j1, (i - 1) * nwalker_per_beta + j2);
                    logprob.swap(i * nwalker_per_beta + j1, (i - 1) * nwalker_per_beta + j2);
                }
            }
        }
    }
}
