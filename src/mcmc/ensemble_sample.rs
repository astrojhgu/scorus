#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::type_complexity)]
#![allow(clippy::mutex_atomic)]

use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std;

use num_traits::float::Float;
use num_traits::NumCast;
use rand::distributions::uniform::SampleUniform;
use rand::distributions::Distribution;
use rand::distributions::Standard;
use rand::seq::SliceRandom;
use rand::Rng;
use std::marker::{Send, Sync};

use std::ops::{Add, Mul, Sub};

//use std::sync::Arc;
use super::utils::{draw_z, scale_vec};

use crate::linear_space::IndexableLinearSpace;

pub enum UpdateFlagSpec<'a, T>
where
    T: Float + SampleUniform,
{
    Prob(T),
    All,
    Func(&'a mut dyn FnMut() -> Vec<bool>),
}

impl<'a, T> UpdateFlagSpec<'a, T>
where
    T: Float + SampleUniform,
{
    pub fn generate_update_flags<U>(&mut self, n: usize, rng: &mut U) -> Vec<bool>
    where
        U: Rng,
    {
        match self {
            UpdateFlagSpec::Prob(ref prob) => loop {
                let result: Vec<_> = (0..n)
                    .map(|_| rng.gen_range(T::zero(), T::one()) < *prob)
                    .collect();
                if result.iter().any(|&b| b) {
                    break result;
                }
            },
            UpdateFlagSpec::All => (0..n).map(|_| true).collect(),
            UpdateFlagSpec::Func(f) => f(),
        }
    }
}

pub fn propose_move<T, V>(p1: &V, p2: &V, z: T, update_flags: Option<&[bool]>) -> V
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Display,
    Standard: Distribution<T>,
    V: Clone + IndexableLinearSpace<T> + Sync + Send + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    let mut result = scale_vec(p1, p2, z);
    if let Some(update_flags) = update_flags {
        for i in 0..update_flags.len() {
            if !update_flags[i] {
                result[i] = p1[i];
            }
        }
    }
    result
}

pub fn init_logprob<T, V, F>(flogprob: &F, ensemble: &[V], logprob: &mut [T])
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + Sync + Send,
    V: Sync + Send + Sized,
    F: Fn(&V) -> T + Send + Sync,
{
    let new_logprob = ensemble.par_iter().map(flogprob).collect::<Vec<_>>();
    logprob.copy_from_slice(&new_logprob);
}

pub fn sample<'a, T, U, V, F>(
    flogprob: &F,
    ensemble: &mut [V],
    cached_logprob: &mut [T],
    rng: &mut U,
    a: T,
    ufs: &mut UpdateFlagSpec<'a, T>,
) where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Display,
    Standard: Distribution<T>,
    U: Rng,
    V: Clone + IndexableLinearSpace<T> + Sync + Send + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    F: Fn(&V) -> T + Send + Sync,
{
    sample_pt(flogprob, ensemble, cached_logprob, rng, a, ufs, &[T::one()])
}

pub fn sample_pt<'a, T, U, V, F>(
    flogprob: &F,
    ensemble: &mut [V],
    cached_logprob: &mut [T],
    rng: &mut U,
    a: T,
    ufs: &mut UpdateFlagSpec<'a, T>,
    beta_list: &[T],
) where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Display,
    Standard: Distribution<T>,
    U: Rng,
    V: Clone + IndexableLinearSpace<T> + Sync + Send + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    F: Fn(&V) -> T + Send + Sync,
{
    //    let cached_logprob = &ensemble_logprob.1;
    //let pflogprob=Arc::new(&flogprob);
    let nwalkers = ensemble.len();
    let nbetas = beta_list.len();
    let nwalkers_per_beta = nwalkers / nbetas;
    assert!(nwalkers_per_beta * nbetas == nwalkers);
    assert!(nwalkers > 0);
    assert!(nwalkers_per_beta % 2 == 0);
    assert!(ensemble.len() == cached_logprob.len());

    let mut pair_id = Vec::new();
    for ibeta in 0..nbetas {
        let offset = ibeta * nwalkers_per_beta;
        let mut b: Vec<_> = (0..nwalkers_per_beta).map(|i| i + offset).collect();
        b.shuffle(rng);
        let mut pid1 = b
            .chunks(2)
            .map(|b1| (b1[0], b1[1]))
            .chain(b.chunks(2).map(|b1| (b1[1], b1[0])))
            .collect();
        pair_id.append(&mut pid1);
    }

    pair_id.sort();
    let z_list: Vec<_> = (0..nwalkers).map(|_| draw_z(rng, a)).collect();
    let flags: Vec<_> = ensemble
        .iter()
        .map(|e| ufs.generate_update_flags(e.dimension(), rng))
        .collect();

    let proposed_pt: Vec<_> = pair_id
        .iter()
        .zip(z_list.iter().zip(flags.iter()))
        .map(|(&(i1, i2), (&z, f))| propose_move(&ensemble[i1], &ensemble[i2], z, Some(f)))
        .collect();

    let new_logprob: Vec<_> = proposed_pt.par_iter().map(|p| flogprob(p)).collect();

    //let nphi = T::from(flags.iter().filter(|&&x| x).count()).unwrap();
    let nphi: Vec<_> = flags
        .iter()
        .map(|f| f.iter().filter(|&&x| x).count())
        .collect();

    let expanded_beta_list = beta_list
        .iter()
        .map(|&b| vec![b; nwalkers_per_beta])
        .collect::<Vec<_>>()
        .concat();

    for (beta, (pt, (ppt, (z, (nphi1, (new_lp, old_lp)))))) in expanded_beta_list.into_iter().zip(
        ensemble.iter_mut().zip(
            proposed_pt.into_iter().zip(
                z_list.into_iter().zip(
                    nphi.into_iter()
                        .zip(new_logprob.into_iter().zip(cached_logprob.iter_mut())),
                ),
            ),
        ),
    ) {
        let delta_lp = new_lp - *old_lp;
        let q = (T::from(nphi1 - 1).unwrap() * z.ln() + delta_lp * beta).exp();
        if rng.gen_range(T::zero(), T::one()) < q {
            *pt = ppt;
            *old_lp = new_lp;
        }
    }
}
