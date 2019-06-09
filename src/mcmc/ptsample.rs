#![allow(clippy::many_single_char_names)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::mutex_atomic)]
use rayon::scope;
use std;
use std::ops::IndexMut;
use std::ops::{Add, Mul, Sub};
use std::sync::Mutex;

use num_traits::float::Float;
use num_traits::identities::{one, zero};
use num_traits::NumCast;

use rand::distributions::uniform::SampleUniform;
use rand::distributions::Distribution;
use rand::distributions::Standard;
use rand::seq::SliceRandom;
use rand::Rng;
//use std::sync::Arc;
use super::mcmc_errors::McmcErr;
use super::utils::{draw_z, scale_vec};
use crate::linear_space::LinearSpace;
use crate::utils::{HasLen, InitFromLen, ItemSwapable};

pub fn create_sampler<'a, T, U, V, W, X, F>(
    flogprob: F,
    mut ensemble_logprob: (W, X),
    mut rng: U,
    beta_list: X,
    a: T,
    nthread: usize,
) -> Box<dyn 'a + FnMut(&mut dyn FnMut(&Result<(W, X), McmcErr>), bool) -> ()>
where
    T: 'static
        + Float
        + NumCast
        + std::cmp::PartialOrd
        + SampleUniform
        + Sync
        + Send
        + std::fmt::Display,
    Standard: Distribution<T>,
    U: 'static + Rng,
    V: Clone + LinearSpace<T> + Sync + Send + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    W: 'static + Clone + IndexMut<usize, Output = V> + HasLen + Sync + Send + Sized + ItemSwapable,
    X: 'static
        + Clone
        + IndexMut<usize, Output = T>
        + HasLen
        + Sync
        + InitFromLen
        + Send
        + Sized
        + ItemSwapable,
    F: 'a + Fn(&V) -> T + Send + Sync,
{
    Box::new(
        move |handler: &mut dyn FnMut(&Result<(W, X), McmcErr>), sw: bool| {
            let result = sample(
                &flogprob,
                &ensemble_logprob,
                &mut rng,
                &beta_list,
                sw,
                a,
                nthread,
            );
            handler(&result);
            if let Ok(x) = result {
                ensemble_logprob = x
            }
        },
    )
}

pub fn create_sampler_st<'a, T, U, V, W, X, F>(
    flogprob: F,
    mut ensemble_logprob: (W, X),
    mut rng: U,
    beta_list: X,
    a: T,
) -> Box<dyn 'a + FnMut(&mut dyn FnMut(&Result<(W, X), McmcErr>), bool) -> ()>
where
    T: 'static + Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Display,
    Standard: Distribution<T>,
    U: 'static + Rng,
    V: Clone + LinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    W: 'static + Clone + IndexMut<usize, Output = V> + HasLen + Sized + ItemSwapable,
    X: 'static + Clone + IndexMut<usize, Output = T> + HasLen + InitFromLen + Sized + ItemSwapable,
    F: 'a + Fn(&V) -> T,
{
    Box::new(
        move |handler: &mut dyn FnMut(&Result<(W, X), McmcErr>), sw: bool| {
            let result = sample_st(&flogprob, &ensemble_logprob, &mut rng, &beta_list, sw, a);
            handler(&result);
            if let Ok(x) = result {
                ensemble_logprob = x;
            }
        },
    )
}

pub fn sample<T, U, V, W, X, F>(
    flogprob: &F,
    ensemble_logprob: &(W, X),
    rng: &mut U,
    beta_list: &X,
    perform_swap: bool,
    a: T,
    nthread: usize,
) -> Result<(W, X), McmcErr>
where
    T: Float
        + NumCast
        + std::cmp::PartialOrd
        + SampleUniform
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display,
    Standard: Distribution<T>,
    U: Rng,
    V: Clone + LinearSpace<T> + Sync + Send + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    W: Clone
        + IndexMut<usize, Output = V>
        + HasLen
        + std::marker::Sync
        + std::marker::Send
        + ItemSwapable,
    X: Clone
        + IndexMut<usize, Output = T>
        + HasLen
        + std::marker::Sync
        + InitFromLen
        + std::marker::Send
        + ItemSwapable,
    F: Fn(&V) -> T + std::marker::Sync + std::marker::Send,
{
    if perform_swap {
        let mut ensemble_logprob1 = (ensemble_logprob.0.clone(), ensemble_logprob.1.clone());
        swap_walkers(&mut ensemble_logprob1, rng, beta_list)?;
        only_sample(flogprob, &ensemble_logprob1, rng, beta_list, a, nthread)
    } else {
        only_sample(flogprob, ensemble_logprob, rng, beta_list, a, nthread)
    }
}

pub fn sample_st<T, U, V, W, X, F>(
    flogprob: &F,
    ensemble_logprob: &(W, X),
    rng: &mut U,
    beta_list: &X,
    perform_swap: bool,
    a: T,
) -> Result<(W, X), McmcErr>
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Display,
    Standard: Distribution<T>,
    U: Rng,
    V: Clone + LinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    W: Clone + IndexMut<usize, Output = V> + HasLen + ItemSwapable,
    X: Clone + IndexMut<usize, Output = T> + HasLen + InitFromLen + ItemSwapable,
    F: Fn(&V) -> T,
{
    if perform_swap {
        let mut ensemble_logprob1 = (ensemble_logprob.0.clone(), ensemble_logprob.1.clone());
        swap_walkers(&mut ensemble_logprob1, rng, beta_list)?;
        only_sample_st(flogprob, &ensemble_logprob1, rng, beta_list, a)
    } else {
        only_sample_st(flogprob, ensemble_logprob, rng, beta_list, a)
    }
}

fn exchange_prob<T>(lp1: T, lp2: T, beta1: T, beta2: T) -> T
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Display,
    Standard: Distribution<T>,
{
    let x = ((beta2 - beta1) * (-lp2 + lp1)).exp();

    if x > one::<T>() {
        one::<T>()
    } else {
        x
    }
}

pub fn swap_walkers<T, U, V, W, X>(
    ensemble_logprob: &mut (W, X),
    rng: &mut U,
    beta_list: &X,
) -> Result<(), McmcErr>
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Display,
    Standard: Distribution<T>,
    U: Rng,
    V: Clone + LinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    W: Clone + IndexMut<usize, Output = V> + HasLen + ItemSwapable,
    X: Clone + IndexMut<usize, Output = T> + HasLen + InitFromLen + ItemSwapable,
{
    //let mut new_ensemble = ensemble_logprob.0.clone();
    //let mut new_logprob = ensemble_logprob.1.clone();
    let (ref mut new_ensemble, ref mut new_logprob) = *ensemble_logprob;
    let nbeta = beta_list.len();
    let nwalker_per_beta = new_ensemble.len() / nbeta;
    if nwalker_per_beta * nbeta != new_ensemble.len() {
        //panic!("Error nensemble/nbeta%0!=0");
        return Err(McmcErr::NWalkersMismatchesNBeta);
    }
    let mut jvec: Vec<usize> = (0..nwalker_per_beta).collect();

    if new_ensemble.len() == new_logprob.len() {
        for i in (1..nbeta).rev() {
            //println!("ibeta={}", i);
            let beta1 = beta_list[i];
            let beta2 = beta_list[i - 1];
            if beta1 >= beta2 {
                //panic!("beta list must be in decreasing order, with no duplicatation");
                return Err(McmcErr::BetaNotInDecrOrd);
            }
            //rng.shuffle(&mut jvec);
            jvec.shuffle(rng);
            //let jvec=shuffle(&jvec, &mut rng);
            for j in 0..nwalker_per_beta {
                let j1 = jvec[j];
                let j2 = j;

                let lp1 = new_logprob[i * nwalker_per_beta + j1];
                let lp2 = new_logprob[(i - 1) * nwalker_per_beta + j2];
                let ep = exchange_prob(lp1, lp2, beta1, beta2);
                //println!("{}",ep);
                let r: T = rng.gen_range(zero::<T>(), one::<T>());
                if r < ep {
                    new_ensemble
                        .swap_items(i * nwalker_per_beta + j1, (i - 1) * nwalker_per_beta + j2);
                    new_logprob
                        .swap_items(i * nwalker_per_beta + j1, (i - 1) * nwalker_per_beta + j2);
                }
            }
        }
    }

    Ok(())
}

fn only_sample<T, U, V, W, X, F>(
    flogprob: &F,
    ensemble_logprob: &(W, X),
    rng: &mut U,
    beta_list: &X,
    a: T,
    nthread: usize,
) -> Result<(W, X), McmcErr>
where
    T: Float
        + NumCast
        + std::cmp::PartialOrd
        + SampleUniform
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::Display,
    Standard: Distribution<T>,
    U: Rng,
    V: Clone + LinearSpace<T> + Sync + Send + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    W: Clone
        + IndexMut<usize, Output = V>
        + HasLen
        + std::marker::Sync
        + std::marker::Send
        + ItemSwapable,
    X: Clone
        + IndexMut<usize, Output = T>
        + HasLen
        + std::marker::Sync
        + InitFromLen
        + std::marker::Send
        + ItemSwapable,
    F: Fn(&V) -> T + std::marker::Sync + std::marker::Send,
{
    let (ref ensemble, ref cached_logprob) = *ensemble_logprob;

    let result_ensemble = ensemble.clone();
    let mut result_logprob = cached_logprob.clone();

    //let pflogprob=Arc::new(flogprob);

    let ensemble = result_ensemble.clone();
    let cached_logprob = result_logprob.clone();

    let nbeta = beta_list.len();
    let nwalkers = ensemble.len() / nbeta;

    if nwalkers == 0 {
        return Err(McmcErr::NWalkersIsZero);
    }
    if nwalkers % 2 != 0 {
        return Err(McmcErr::NWalkersIsNotEven);
    }

    if nbeta * nwalkers != ensemble.len() {
        return Err(McmcErr::NWalkersMismatchesNBeta);
    }

    let ndims: T = NumCast::from(ensemble[0].dimension()).unwrap();

    let half_nwalkers = nwalkers / 2;
    let mut walker_group: Vec<Vec<Vec<usize>>> = Vec::new();
    walker_group.reserve(nbeta);
    let mut walker_group_id: Vec<Vec<usize>> = Vec::new();
    walker_group_id.reserve(nbeta);

    let mut rvec: Vec<Vec<T>> = Vec::new();
    let mut jvec: Vec<Vec<usize>> = Vec::new();
    let mut zvec: Vec<Vec<T>> = Vec::new();

    for i in 0..nbeta {
        walker_group.push(vec![Vec::new(), Vec::new()]);
        walker_group[i][0].reserve(half_nwalkers);
        walker_group[i][1].reserve(half_nwalkers);

        walker_group_id.push(Vec::new());
        walker_group_id[i].reserve(nwalkers);

        rvec.push(Vec::new());
        jvec.push(Vec::new());
        zvec.push(Vec::new());

        rvec[i].reserve(nwalkers);
        jvec[i].reserve(nwalkers);
        zvec[i].reserve(nwalkers);

        for j in 0..nwalkers {
            let mut gid: usize = rng.gen_range(0, 2);
            if walker_group[i][gid].len() == half_nwalkers {
                gid = 1 - gid;
            }
            walker_group[i][gid].push(j);
            walker_group_id[i].push(gid);
            rvec[i].push(rng.gen_range(zero::<T>(), one::<T>()));
            jvec[i].push(rng.gen_range(0, half_nwalkers));
            zvec[i].push(draw_z(rng, a));
        }
    }

    let atomic_k = Mutex::new(0);
    let lp_cached = result_logprob.len() == result_ensemble.len();

    if !lp_cached {
        //result_logprob.resize(result_ensemble.len(), T::zero());
        result_logprob = X::init(result_ensemble.len());
    }
    //let lp_cached=cached_logprob.len()!=0;
    let result_ensemble = Mutex::new(result_ensemble);
    let result_logprob = Mutex::new(result_logprob);
    {
        let create_task = || {
            let atomic_k = &atomic_k;
            let result_ensemble = &result_ensemble;
            let result_logprob = &result_logprob;
            let ensemble = &ensemble;
            let cached_logprob = &cached_logprob;
            let zvec = &zvec;
            let walker_group = &walker_group;
            let walker_group_id = &walker_group_id;
            let jvec = &jvec;
            //let rvec=Arc::clone(&rvec);
            let rvec = &rvec;
            let flogprob = flogprob;
            //let nwalkers=nwalkers;
            move || loop {
                let n: usize;
                {
                    let mut k1 = atomic_k.lock().unwrap();
                    n = *k1;
                    *k1 += 1;
                }
                if n >= nwalkers * nbeta {
                    break;
                }

                let ibeta = n / nwalkers;
                let k = n - ibeta * nwalkers;

                let lp_last_y = if !lp_cached {
                    //let yy1 = flogprob(&ensemble[ibeta * nwalkers + k]);
                    let yy1 = flogprob(&ensemble[ibeta * nwalkers + k]);
                    let mut lpyy = result_logprob.lock().unwrap();
                    lpyy[ibeta * nwalkers + k] = yy1;
                    yy1
                } else {
                    cached_logprob[ibeta * nwalkers + k]
                };

                let i = walker_group_id[ibeta][k];
                let j = jvec[ibeta][k];
                let ni = 1 - i;
                let z = zvec[ibeta][k];
                let r = rvec[ibeta][k];
                let new_y = scale_vec(
                    &ensemble[ibeta * nwalkers + k],
                    &ensemble[ibeta * nwalkers + walker_group[ibeta][ni][j]],
                    z,
                );
                let lp_y = flogprob(&new_y);
                let beta = beta_list[ibeta];
                let delta_lp = lp_y - lp_last_y;
                let q = ((ndims - one::<T>()) * (z.ln()) + delta_lp * beta).exp();
                {
                    let mut yy = result_ensemble.lock().unwrap();
                    let mut lpyy = result_logprob.lock().unwrap();
                    if r <= q {
                        yy[ibeta * nwalkers + k] = new_y;
                        lpyy[ibeta * nwalkers + k] = lp_y;
                    }
                }
            }
        };

        if nthread > 1 {
            /*let mut pool = Pool::new(nthread as u32);
            pool.scoped(|scope| {
                for _ in 0..nthread {
                    scope.execute(create_task());
                }
            });*/

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

    let result_ensemble = result_ensemble.into_inner().unwrap();
    let result_logprob = result_logprob.into_inner().unwrap();

    Ok((result_ensemble, result_logprob))
}

fn only_sample_st<T, U, V, W, X, F>(
    flogprob: &F,
    ensemble_logprob: &(W, X),
    rng: &mut U,
    beta_list: &X,
    a: T,
) -> Result<(W, X), McmcErr>
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Display,
    Standard: Distribution<T>,
    U: Rng,
    V: Clone + LinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    W: Clone + IndexMut<usize, Output = V> + HasLen + ItemSwapable,
    X: Clone + IndexMut<usize, Output = T> + HasLen + InitFromLen + ItemSwapable,
    F: Fn(&V) -> T,
{
    let (ref ensemble, ref cached_logprob) = *ensemble_logprob;

    let mut result_ensemble = ensemble.clone();
    let mut result_logprob = cached_logprob.clone();

    let nbeta = beta_list.len();
    let nwalkers = ensemble.len() / nbeta;

    if nwalkers == 0 {
        return Err(McmcErr::NWalkersIsZero);
    }
    if nwalkers % 2 != 0 {
        return Err(McmcErr::NWalkersIsNotEven);
    }

    if nbeta * nwalkers != ensemble.len() {
        return Err(McmcErr::NWalkersMismatchesNBeta);
    }

    let ndims: T = NumCast::from(ensemble[0].dimension()).unwrap();

    let half_nwalkers = nwalkers / 2;
    let mut walker_group: Vec<Vec<Vec<usize>>> = Vec::new();
    walker_group.reserve(nbeta);
    let mut walker_group_id: Vec<Vec<usize>> = Vec::new();
    walker_group_id.reserve(nbeta);

    let mut rvec: Vec<Vec<T>> = Vec::new();
    let mut jvec: Vec<Vec<usize>> = Vec::new();
    let mut zvec: Vec<Vec<T>> = Vec::new();

    for i in 0..nbeta {
        walker_group.push(vec![Vec::new(), Vec::new()]);
        walker_group[i][0].reserve(half_nwalkers);
        walker_group[i][1].reserve(half_nwalkers);

        walker_group_id.push(Vec::new());
        walker_group_id[i].reserve(nwalkers);

        rvec.push(Vec::new());
        jvec.push(Vec::new());
        zvec.push(Vec::new());

        rvec[i].reserve(nwalkers);
        jvec[i].reserve(nwalkers);
        zvec[i].reserve(nwalkers);

        for j in 0..nwalkers {
            let mut gid: usize = rng.gen_range(0, 2);
            if walker_group[i][gid].len() == half_nwalkers {
                gid = 1 - gid;
            }
            walker_group[i][gid].push(j);
            walker_group_id[i].push(gid);
            rvec[i].push(rng.gen_range(zero::<T>(), one::<T>()));
            jvec[i].push(rng.gen_range(0, half_nwalkers));
            zvec[i].push(draw_z(rng, a));
        }
    }

    let lp_cached = result_logprob.len() == result_ensemble.len();

    if !lp_cached {
        //result_logprob.resize(result_ensemble.len(), T::zero());
        result_logprob = X::init(result_ensemble.len());
    }
    //let lp_cached=cached_logprob.len()!=0;

    for n in 0..nwalkers * nbeta {
        let ibeta = n / nwalkers;
        let k = n - ibeta * nwalkers;
        let lp_last_y = if !lp_cached {
            //let yy1 = flogprob(&ensemble[ibeta * nwalkers + k]);
            let yy1 = flogprob(&ensemble[ibeta * nwalkers + k]);

            result_logprob[ibeta * nwalkers + k] = yy1;
            yy1
        } else {
            cached_logprob[ibeta * nwalkers + k]
        };

        let i = walker_group_id[ibeta][k];
        let j = jvec[ibeta][k];
        let ni = 1 - i;
        let z = zvec[ibeta][k];
        let r = rvec[ibeta][k];
        let new_y = scale_vec(
            &ensemble[ibeta * nwalkers + k],
            &ensemble[ibeta * nwalkers + walker_group[ibeta][ni][j]],
            z,
        );
        let lp_y = flogprob(&new_y);
        let beta = beta_list[ibeta];
        let delta_lp = lp_y - lp_last_y;
        let q = ((ndims - one::<T>()) * (z.ln()) + delta_lp * beta).exp();

        if r <= q {
            result_ensemble[ibeta * nwalkers + k] = new_y;
            result_logprob[ibeta * nwalkers + k] = lp_y;
        }
    }

    Ok((result_ensemble, result_logprob))
}
