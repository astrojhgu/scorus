#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::type_complexity)]
#![allow(clippy::mutex_atomic)]
use rayon::scope;
use std;

use num_traits::float::Float;
use num_traits::identities::one;
use num_traits::NumCast;
use rand::distributions::uniform::SampleUniform;
use rand::distributions::Distribution;
use rand::distributions::Standard;
use rand::seq::SliceRandom;
use rand::Rng;
use std::marker::{Send, Sync};

use std::ops::{Add, Mul, Sub};
use std::sync::Mutex;

//use std::sync::Arc;
use super::utils::{draw_z, scale_vec};

use crate::linear_space::FiniteLinearSpace;

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
    V: Clone + FiniteLinearSpace<T> + Sync + Send + Sized,
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

pub fn init_logprob<T, V, F>(flogprob: &F, ensemble: &[V], logprob: &mut [T], nthread: usize)
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + Sync + Send,
    V: Sync + Send + Sized,
    F: Fn(&V) -> T + Send + Sync,
{
    let nwalkers = ensemble.len();
    assert_eq!(nwalkers, logprob.len());

    let new_logprob = Mutex::new(vec![T::zero(); nwalkers]);

    let atomic_k = Mutex::new(0);

    let create_task = || {
        let atomic_k = &atomic_k;
        let new_logprob = &new_logprob;
        let ensemble = ensemble;
        let flogprob = flogprob;
        //let rvec=Arc::clone(&rvec);
        move || loop {
            let k: usize;
            {
                let mut k1 = atomic_k.lock().unwrap();
                k = *k1;
                *k1 += 1;
            }
            if k >= nwalkers {
                break;
            }

            let lp_y = flogprob(&ensemble[k]);

            {
                let mut nlp = new_logprob.lock().unwrap();
                nlp[k] = lp_y;
            }
        }
    };

    if nthread > 1 {
        scope(|s| {
            for _ in 0..nthread {
                s.spawn(|_| create_task()());
            }
        });
    } else {
        let task = create_task();
        task();
    }

    let new_logprob = new_logprob.into_inner().unwrap();
    logprob.copy_from_slice(&new_logprob);
}

pub fn sample<'a, T, U, V, F>(
    flogprob: &F,
    ensemble: &mut [V],
    cached_logprob: &mut [T],
    rng: &mut U,
    a: T,
    ufs: &mut UpdateFlagSpec<'a, T>,
    nthread: usize,
) where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Display,
    Standard: Distribution<T>,
    U: Rng,
    V: Clone + FiniteLinearSpace<T> + Sync + Send + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    F: Fn(&V) -> T + Send + Sync,
{
    sample_pt(
        flogprob,
        ensemble,
        cached_logprob,
        rng,
        a,
        ufs,
        &[T::one()],
        nthread,
    )
}

pub fn sample_pt<'a, T, U, V, F>(
    flogprob: &F,
    ensemble: &mut [V],
    cached_logprob: &mut [T],
    rng: &mut U,
    a: T,
    ufs: &mut UpdateFlagSpec<'a, T>,
    beta_list: &[T],
    nthread: usize,
) where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Display,
    Standard: Distribution<T>,
    U: Rng,
    V: Clone + FiniteLinearSpace<T> + Sync + Send + Sized,
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

    let pair_id: Vec<Vec<(usize, usize)>> = (0..nbetas)
        .map(|_| {
            let mut a: Vec<usize> = (0..nwalkers_per_beta).collect();
            a.shuffle(rng);
            a.chunks(2)
                .map(|a| (a[0], a[1]))
                .chain(a.chunks(2).map(|a| (a[1], a[0])))
                .collect()
        })
        .collect();

    let proposed_pt_z: Vec<Vec<_>> = pair_id
        .iter()
        .enumerate()
        .map(|(ibeta, pair_id1)| {
            pair_id1
                .iter()
                .map(|(i1, i2)| {
                    let z = draw_z(rng, a);
                    let offset = ibeta * nwalkers_per_beta;
                    let flags = ufs.generate_update_flags(ensemble[offset + i1].dimension(), rng);
                    (
                        propose_move(
                            &ensemble[offset + i1],
                            &ensemble[offset + i2],
                            z,
                            Some(&flags),
                        ),
                        z,
                        flags,
                    )
                })
                .collect()
        })
        .collect();

    let new_logprob = Mutex::new(vec![vec![T::zero(); nwalkers_per_beta]; nbetas]);

    let atomic_k = Mutex::new(0);

    {
        let create_task = || {
            let atomic_k = &atomic_k;
            let new_logprob = &new_logprob;
            let proposed_pt_z = &proposed_pt_z;
            let flogprob = flogprob;
            //let rvec=Arc::clone(&rvec);
            move || loop {
                let k: usize;
                {
                    let mut k1 = atomic_k.lock().unwrap();
                    k = *k1;
                    *k1 += 1;
                }
                if k >= nwalkers {
                    break;
                }

                let ibeta = k / nwalkers_per_beta;
                let jbeta = k - ibeta * nwalkers_per_beta;

                let lp_y = flogprob(&proposed_pt_z[ibeta][jbeta].0);

                {
                    let mut nlp = new_logprob.lock().unwrap();
                    nlp[ibeta][jbeta] = lp_y;
                }
            }
        };

        if nthread > 1 {
            scope(|s| {
                for _ in 0..nthread {
                    s.spawn(|_| create_task()());
                }
            });
        } else {
            let task = create_task();
            task();
        }
    }

    let new_logprob = new_logprob.into_inner().unwrap();
    for (ibeta, (proposed_pt_z1, (new_logprob1, (pair_id1, &beta)))) in proposed_pt_z
        .into_iter()
        .zip(
            new_logprob
                .into_iter()
                .zip(pair_id.into_iter().zip(beta_list.iter())),
        )
        .enumerate()
    {
        for (_i, ((pt, z, flags), (new_lp, (i1, _i2)))) in proposed_pt_z1
            .into_iter()
            .zip(new_logprob1.into_iter().zip(pair_id1.into_iter()))
            .enumerate()
        {
            let nphi = T::from(flags.iter().filter(|&&x| x).count()).unwrap();
            let n = ibeta * nwalkers_per_beta + i1;
            let lp_last_y = cached_logprob[n];
            let delta_lp = new_lp - lp_last_y;
            let q = ((nphi - one::<T>()) * (z.ln()) + delta_lp * beta).exp();
            if rng.gen_range(T::zero(), T::one()) < q {
                ensemble[n] = pt;
                cached_logprob[n] = new_lp;
            }
        }
    }
}
