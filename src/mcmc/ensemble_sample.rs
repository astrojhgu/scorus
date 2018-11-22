use rayon::scope;
use std;

use num_traits::float::Float;
use num_traits::identities::{one, zero};
use num_traits::NumCast;
use rand::distributions::uniform::SampleUniform;
use rand::distributions::Distribution;
use rand::distributions::Standard;
use rand::Rng;
use std::marker::{Send, Sync};
use std::ops::IndexMut;
use std::sync::Mutex;
//use std::sync::Arc;

use super::mcmc_errors::McmcErr;
use super::utils::draw_z;
use crate::utils::HasLen;
use crate::utils::InitFromLen;

use super::utils::scale_vec;

pub fn create_sampler<'a, T, U, V, W, X, F>(
    flogprob: F,
    mut ensemble_logprob: (W, X),
    mut rng: U,
    a: T,
    nthread: usize,
) -> Box<dyn 'a + FnMut(&mut FnMut(&Result<(W, X), McmcErr>)) -> ()>
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
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sync + Send + Sized,
    W: 'static + Clone + IndexMut<usize, Output = V> + HasLen + Sync + Send + Drop + Sized,
    X: 'static
        + Clone
        + IndexMut<usize, Output = T>
        + HasLen
        + Sync
        + InitFromLen
        + Send
        + Drop
        + Sized,
    F: 'a + Fn(&V) -> T + Send + Sync + Sized,
{
    Box::new(move |handler: &mut FnMut(&Result<(W, X), McmcErr>)| {
        let result = sample(&flogprob, &ensemble_logprob, &mut rng, a, nthread);
        handler(&result);
        match result {
            Ok(x) => ensemble_logprob = x,
            _ => (),
        }
    })
}

pub fn create_sampler_st<'a, T, U, V, W, X, F>(
    flogprob: F,
    mut ensemble_logprob: (W, X),
    mut rng: U,
    a: T,
) -> Box<dyn 'a + FnMut(&mut FnMut(&Result<(W, X), McmcErr>)) -> ()>
where
    T: 'static + Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Display,
    Standard: Distribution<T>,
    U: 'static + Rng,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sized,
    W: 'static + Clone + IndexMut<usize, Output = V> + HasLen + Drop + Sized,
    X: 'static + Clone + IndexMut<usize, Output = T> + HasLen + InitFromLen + Drop + Sized,
    F: 'a + Fn(&V) -> T + Sized,
{
    Box::new(move |handler: &mut FnMut(&Result<(W, X), McmcErr>)| {
        let result = sample_st(&flogprob, &ensemble_logprob, &mut rng, a);
        handler(&result);
        match result {
            Ok(x) => ensemble_logprob = x,
            _ => (),
        }
    })
}

pub fn sample<T, U, V, W, X, F>(
    flogprob: &F,
    ensemble_logprob: &(W, X),
    rng: &mut U,
    a: T,
    nthread: usize,
) -> Result<(W, X), McmcErr>
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Display,
    Standard: Distribution<T>,
    U: Rng,
    V: Clone + IndexMut<usize, Output = T> + HasLen + Sync + Send,
    W: Clone + IndexMut<usize, Output = V> + HasLen + Sync + Send + Drop,
    X: Clone + IndexMut<usize, Output = T> + HasLen + Sync + InitFromLen + Send + Drop,
    F: Fn(&V) -> T + Send + Sync + ?Sized,
{
    let (ref ensemble, ref cached_logprob) = *ensemble_logprob;
    //    let cached_logprob = &ensemble_logprob.1;
    let result_ensemble = ensemble.clone();
    let mut result_logprob = cached_logprob.clone();
    //let pflogprob=Arc::new(&flogprob);
    let nwalkers = ensemble.len();

    if nwalkers == 0 {
        return Err(McmcErr::NWalkersIsZero);
    }

    if nwalkers % 2 != 0 {
        return Err(McmcErr::NWalkersIsNotEven);
    }

    let ndims: T = NumCast::from(ensemble[0].len()).unwrap();

    let half_nwalkers = nwalkers / 2;
    let mut walker_group: Vec<Vec<usize>> = vec![Vec::new(), Vec::new()];
    walker_group[0].reserve(half_nwalkers);
    walker_group[1].reserve(half_nwalkers);
    let mut walker_group_id: Vec<usize> = Vec::new();
    walker_group_id.reserve(nwalkers);
    let mut rvec: Vec<T> = Vec::new();
    let mut jvec: Vec<usize> = Vec::new();
    let mut zvec: Vec<T> = Vec::new();
    rvec.reserve(nwalkers);
    jvec.reserve(nwalkers);
    zvec.reserve(nwalkers);
    for i in 0..nwalkers {
        let mut gid: usize = rng.gen_range(0, 2);

        if walker_group[gid].len() == half_nwalkers {
            gid = 1 - gid;
        }
        walker_group[gid].push(i);
        walker_group_id.push(gid);
        rvec.push(rng.gen_range(zero::<T>(), one::<T>()));
        jvec.push(rng.gen_range(0, half_nwalkers));
        zvec.push(draw_z(rng, a));
    }

    let atomic_k = Mutex::new(0);
    let lp_cached = result_logprob.len() == nwalkers;

    if !lp_cached {
        //result_logprob.resize(nwalkers, T::zero());
        result_logprob = X::init(nwalkers);
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
            let flogprob = flogprob;
            //let rvec=Arc::clone(&rvec);
            let rvec = &rvec;
            //let nwalkers=nwalkers;
            let task = move || loop {
                let k: usize;
                {
                    let mut k1 = atomic_k.lock().unwrap();
                    k = *k1;
                    *k1 += 1;
                }
                if k >= nwalkers {
                    break;
                }

                let lp_last_y = match lp_cached {
                    false => {
                        let yy1 = flogprob(&ensemble[k]);
                        let mut lpyy = result_logprob.lock().unwrap();
                        lpyy[k] = yy1;
                        yy1
                    }
                    _ => cached_logprob[k],
                };

                let i = walker_group_id[k];
                let j = jvec[k];
                let ni = 1 - i;
                let z = zvec[k];
                let r = rvec[k];
                let new_y = scale_vec(&ensemble[k], &ensemble[walker_group[ni][j]], z);
                let lp_y = flogprob(&new_y);

                let q = ((ndims - one::<T>()) * (z.ln()) + lp_y - lp_last_y).exp();
                {
                    let mut yy = result_ensemble.lock().unwrap();
                    let mut lpyy = result_logprob.lock().unwrap();
                    if r <= q {
                        yy[k] = new_y;
                        lpyy[k] = lp_y;
                    }
                }
            };
            task
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

pub fn sample_st<T, U, V, W, X, F>(
    flogprob: &F,
    ensemble_logprob: &(W, X),
    rng: &mut U,
    a: T,
) -> Result<(W, X), McmcErr>
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Display,
    Standard: Distribution<T>,
    U: Rng,
    V: Clone + IndexMut<usize, Output = T> + HasLen,
    W: Clone + IndexMut<usize, Output = V> + HasLen + Drop,
    X: Clone + IndexMut<usize, Output = T> + HasLen + InitFromLen + Drop,
    F: Fn(&V) -> T + ?Sized,
{
    let (ref ensemble, ref cached_logprob) = *ensemble_logprob;
    //    let cached_logprob = &ensemble_logprob.1;
    let mut result_ensemble = ensemble.clone();
    let mut result_logprob = cached_logprob.clone();
    //let pflogprob=Arc::new(&flogprob);
    let nwalkers = ensemble.len();

    if nwalkers == 0 {
        return Err(McmcErr::NWalkersIsZero);
    }

    if nwalkers % 2 != 0 {
        return Err(McmcErr::NWalkersIsNotEven);
    }

    let ndims: T = NumCast::from(ensemble[0].len()).unwrap();

    let half_nwalkers = nwalkers / 2;
    let mut walker_group: Vec<Vec<usize>> = vec![Vec::new(), Vec::new()];
    walker_group[0].reserve(half_nwalkers);
    walker_group[1].reserve(half_nwalkers);
    let mut walker_group_id: Vec<usize> = Vec::new();
    walker_group_id.reserve(nwalkers);
    let mut rvec: Vec<T> = Vec::new();
    let mut jvec: Vec<usize> = Vec::new();
    let mut zvec: Vec<T> = Vec::new();
    rvec.reserve(nwalkers);
    jvec.reserve(nwalkers);
    zvec.reserve(nwalkers);
    for i in 0..nwalkers {
        let mut gid: usize = rng.gen_range(0, 2);

        if walker_group[gid].len() == half_nwalkers {
            gid = 1 - gid;
        }
        walker_group[gid].push(i);
        walker_group_id.push(gid);
        rvec.push(rng.gen_range(zero::<T>(), one::<T>()));
        jvec.push(rng.gen_range(0, half_nwalkers));
        zvec.push(draw_z(rng, a));
    }

    let lp_cached = result_logprob.len() == nwalkers;

    if !lp_cached {
        //result_logprob.resize(nwalkers, T::zero());
        result_logprob = X::init(nwalkers);
    }
    //let lp_cached=cached_logprob.len()!=0;

    for k in 0..nwalkers {
        let lp_last_y = match lp_cached {
            false => {
                let yy1 = flogprob(&ensemble[k]);
                result_logprob[k] = yy1;
                yy1
            }
            _ => cached_logprob[k],
        };

        let i = walker_group_id[k];
        let j = jvec[k];
        let ni = 1 - i;
        let z = zvec[k];
        let r = rvec[k];
        let new_y = scale_vec(&ensemble[k], &ensemble[walker_group[ni][j]], z);
        let lp_y = flogprob(&new_y);

        let q = ((ndims - one::<T>()) * (z.ln()) + lp_y - lp_last_y).exp();
        {
            if r <= q {
                result_ensemble[k] = new_y;
                result_logprob[k] = lp_y;
            }
        }
    }

    Ok((result_ensemble, result_logprob))
}
