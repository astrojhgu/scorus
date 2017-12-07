extern crate rand;
extern crate scoped_threadpool;
extern crate std;
use scoped_threadpool::Pool;

//use std::sync::Arc;
use mcmc_errors::McmcErrs;
use utils::draw_z;
use utils::HasLength;
use utils::Resizeable;
use utils::ItemSwapable;
use std::sync::Mutex;
use utils::scale_vec;
use utils::swap_walkers;
use std::ops::IndexMut;
use num_traits::float::Float;
use num_traits::NumCast;
use num_traits::identities::one;
use num_traits::identities::zero;


pub fn sample<T, U, V, W, X, F>(
    flogprob: &F,
    ensemble_logprob: &(W, X),
    mut rng: &mut U,
    beta_list: &X,
    perform_swap: bool,
    a: T,
    nthread: usize,
) -> Result<(W, X), McmcErrs>
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
    F: Fn(&V) -> T + std::marker::Sync + std::marker::Send,
{
    let (result_ensemble, mut result_logprob) =
        swap_walkers(ensemble_logprob, &mut rng, &beta_list, perform_swap)?;

    //let pflogprob=Arc::new(flogprob);

    let ensemble = result_ensemble.clone();
    let cached_logprob = result_logprob.clone();

    let nbeta = beta_list.length();
    let nwalkers = ensemble.length() / nbeta;

    if nwalkers == 0 {
        return Err(McmcErrs::NWalkersIsZero);
    }
    if nwalkers % 2 != 0 {
        return Err(McmcErrs::NWalkersIsNotEven);
    }


    if nbeta * nwalkers != ensemble.length() {
        return Err(McmcErrs::NWalkersMismatchesNBeta);
    }



    let ndims: T = NumCast::from(ensemble[0].length()).unwrap();

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
            rvec[i].push(rng.gen_range(zero(), one()));
            jvec[i].push(rng.gen_range(0, half_nwalkers));
            zvec[i].push(draw_z(rng, a));
        }
    }


    let atomic_k = Mutex::new(0);
    let lp_cached = result_logprob.length() == result_ensemble.length();

    if !lp_cached {
        result_logprob.resize(result_ensemble.length());
    }
    //let lp_cached=cached_logprob.length()!=0;
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
            let flogprob = &flogprob;
            //let nwalkers=nwalkers;
            let task = move || loop {
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

                let lp_last_y = match lp_cached {
                    false => {
                        //let yy1 = flogprob(&ensemble[ibeta * nwalkers + k]);
                        let yy1 = flogprob(&ensemble[ibeta * nwalkers + k]);
                        let mut lpyy = result_logprob.lock().unwrap();
                        lpyy[ibeta * nwalkers + k] = yy1;
                        yy1
                    }
                    _ => cached_logprob[ibeta * nwalkers + k],
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
            };
            task
        };

        if nthread > 1 {
            let mut pool = Pool::new(nthread as u32);
            pool.scoped(|scope| {
                for _ in 0..nthread {
                    scope.execute(create_task());
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
