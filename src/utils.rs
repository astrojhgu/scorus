extern crate rand;
extern crate std;
use num_traits::float::Float;
use num_traits::identities::one;
use num_traits::identities::zero;
use num_traits::NumCast;
use std::ops::IndexMut;
pub trait HasLength {
    fn length(&self) -> usize;
}

pub trait Resizeable {
    fn resize(&mut self, usize);
}

pub trait ItemSwapable {
    fn swap_items(&mut self, i: usize, j: usize);
}

pub fn shuffle<T: HasLength + Clone + ItemSwapable, U: rand::Rng>(arr: &T, rng: &mut U) -> T {
    let mut x = arr.clone();
    let l = arr.length();
    for i in (1..l).rev() {
        let i1 = rng.gen_range(0, i + 1);
        x.swap_items(i, i1);
    }
    x
}

pub fn draw_z<
    T: Float + rand::Rand + std::cmp::PartialOrd + rand::distributions::range::SampleRange,
    U: rand::Rng,
>(
    rng: &mut U,
    a: T,
) -> T {
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
    U: Clone + IndexMut<usize, Output = T> + HasLength,
{
    let mut result = (*x1).clone();
    let unit: T = one();
    for l in 0..x1.length() {
        result[l] = z * x1[l] + (unit - z) * x2[l];
    }

    result
}

pub fn exchange_prob<T>(lp1: T, lp2: T, beta1: T, beta2: T) -> T
where
    T: Float
        + NumCast
        + rand::Rand
        + std::cmp::PartialOrd
        + rand::distributions::range::SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display,
{
    let x = ((beta2 - beta1) * (-lp2 + lp1)).exp();
    let unit: T = one();
    match x > unit {
        true => unit,
        false => x,
    }
}

pub fn swap_walkers<T, U, V, W, X>(
    ensemble_logprob: &(W, X),
    rng: &mut U,
    beta_list: &X,
    perform_swap: bool,
) -> (W, X)
where
    T: Float
        + NumCast
        + rand::Rand
        + std::cmp::PartialOrd
        + rand::distributions::range::SampleRange
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display,
    U: rand::Rng,
    V: Clone + IndexMut<usize, Output = T> + HasLength + std::marker::Sync + std::marker::Send,
    W: Clone
        + IndexMut<usize, Output = V>
        + HasLength
        + std::marker::Sync
        + std::marker::Send
        + Drop
        + ItemSwapable,
    X: Clone
        + IndexMut<usize, Output = T>
        + HasLength
        + std::marker::Sync
        + Resizeable
        + std::marker::Send
        + Drop
        + ItemSwapable,
{
    let mut new_ensemble = ensemble_logprob.0.clone();
    let mut new_logprob = ensemble_logprob.1.clone();
    let nbeta = beta_list.length();
    let nwalker_per_beta = new_ensemble.length() / nbeta;
    if nwalker_per_beta * nbeta != new_ensemble.length() {
        panic!("Error nensemble/nbeta%0!=0");
    }

    if perform_swap && new_ensemble.length() == new_logprob.length() {
        for i in (1..nbeta).rev() {
            let beta1 = beta_list[i];
            let beta2 = beta_list[i - 1];
            if beta1 >= beta2 {
                panic!("beta list must be in decreasing order, with no duplicatation");
            }
            let mut jvec = Vec::new();
            jvec.reserve(nwalker_per_beta);
            for j in 0..nwalker_per_beta {
                jvec.push(j);
            }
            rng.shuffle(&mut jvec);
            //let jvec=shuffle(&jvec, &mut rng);
            for j in 0..nwalker_per_beta {
                let j1 = jvec[j];
                let j2 = j;

                let lp1 = new_logprob[i * nwalker_per_beta + j1];
                let lp2 = new_logprob[(i - 1) * nwalker_per_beta + j2];
                let ep = exchange_prob(lp1, lp2, beta1, beta2);

                let r: T = rng.gen_range(zero(), one());
                if r < ep {
                    new_ensemble
                        .swap_items(i * nwalker_per_beta + j1, (i - 1) * nwalker_per_beta + j2);
                    new_logprob
                        .swap_items(i * nwalker_per_beta + j1, (i - 1) * nwalker_per_beta + j2);
                }
            }
        }
    }

    (new_ensemble, new_logprob)
}
