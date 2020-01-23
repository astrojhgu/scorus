#![allow(clippy::too_many_arguments)]
#![allow(clippy::eq_op)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::type_complexity)]
#![allow(clippy::mutex_atomic)]
//use rayon::scope;
use rayon::scope;
use std;

use num_traits::float::Float;
//use num_traits::identities::{one, zero};
//use num_traits::NumCast;
use rand::distributions::uniform::SampleUniform;
use rand::distributions::Distribution;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::StandardNormal;

//use std::marker::{Send, Sync};
use crate::linear_space::FiniteLinearSpace;
use crate::utils::HasLen;
use crate::utils::InitFromLen;
use crate::utils::ItemSwapable;
use rand::distributions::Standard;
use std::ops::{Add, IndexMut, Mul, Sub};
use std::sync::Mutex;
use super::super::utils::swap_walkers;
use super::utils::{calc_gamma, replace_flag};

pub fn sample_dx<T, U, V, W>(old: &W,ibeta: usize, i: usize, delta: usize,nbeta: usize,  rng: &mut U) -> V
where
    T: Float + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Debug,
    Standard: Distribution<T>,
    U: Rng,
    V: Clone + FiniteLinearSpace<T> + Sync + Send + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    W: Clone + IndexMut<usize, Output = V> + HasLen + Sync + Send+ ItemSwapable,
{
    //step 1a
    let delta = delta.min(old.len()/nbeta / 2);
    let nchains_per_beta=old.len()/nbeta;
    let mut j_list: Vec<_> = (0..nchains_per_beta).filter(|&x| x != i).collect();
    j_list.shuffle(rng);
    let n_list: Vec<_> = j_list.iter().take(delta).cloned().collect();
    let j_list: Vec<_> = j_list.iter().skip(delta).take(delta).cloned().collect();
    //println!("{} {} {} {}", ibeta, nchains_per_beta, nbeta, i);
    let start_point = &old[ibeta*nchains_per_beta+i];
    let mut dx = start_point - start_point;
    for (&j, &n) in j_list.iter().zip(n_list.iter()) {
        dx = &dx + &(&old[ibeta*nchains_per_beta+j] - &old[ibeta*nchains_per_beta+n]);
    }
    dx
}

pub fn propose_point<T, U, V, W>(
    old: &W,
    ibeta: usize,
    i: usize,
    delta: usize,
    nbeta: usize, 
    cr: T,
    b: T,
    b_star: T,
    gamma_func: &Option<Box<dyn Fn(usize, usize) -> T>>,
    rng: &mut U,
) -> V
where
    T: Float + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Debug,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    U: Rng,
    V: Clone + FiniteLinearSpace<T> + Sync + Send + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    W: Clone + IndexMut<usize, Output = V> + HasLen + Sync + Send+ ItemSwapable,
{
    let nchains_per_beta=old.len()/nbeta;
    let dx = sample_dx(old, ibeta, i, delta, nbeta, rng);
    let (flag, dprime) = replace_flag(dx.dimension(), cr, rng);
    //println!("{:?} {}", flag, dprime);
    let gamma = if let Some(g) = gamma_func {
        g(delta, dprime)
    } else {
        calc_gamma(delta, dprime)
    };

    let mut proposed = old[ibeta*nchains_per_beta+i].clone();
    for d in 0..proposed.dimension() {
        if flag[d] {
            proposed[d] = proposed[d]
                + dx[d] * gamma * (T::one() + rng.gen_range(-b, b))
                + rng.sample(StandardNormal) * b_star;
        }
    }
    proposed
}

pub fn accept<T, U, V, W, X>(old: &mut W, old_lp: &mut X,ibeta: usize, i: usize, proposed: V, lp: T, beta_list: &X, rng: &mut U)->bool
where
    T: Float + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Debug,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    U: Rng,
    V: Clone + FiniteLinearSpace<T> + Sync + Send + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    W: Clone + IndexMut<usize, Output = V> + HasLen + Sync + Send + ItemSwapable,
    X: Clone + IndexMut<usize, Output = T> + HasLen + Sync + InitFromLen + Send + ItemSwapable,
{
    let nbeta=beta_list.len();
    let nchains_per_beta=old.len()/nbeta;
    let alpha = ((lp - old_lp[i])*beta_list[ibeta]).exp();
    if rng.gen_range(T::zero(), T::one()) < alpha {
        //print!("a");
        old[ibeta*nchains_per_beta+i] = proposed;
        old_lp[ibeta*nchains_per_beta+i] = lp;
        true
    }else{
        //println!("reject {:?} {:?}", lp, old_lp[i]);
        false
    }
    
}

pub fn init_chain<T, V, W, F>(ensemble: W, flogprob: &F, njobs: usize) -> (W, Vec<T>)
where
    T: Float + std::cmp::PartialOrd + Sync + Send + std::fmt::Debug,
    V: Clone + FiniteLinearSpace<T> + Sync + Send + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    W: Clone + IndexMut<usize, Output = V> + HasLen + Sync + Send+ ItemSwapable,
    F: Fn(&V) -> T + Send + Sync + ?Sized,
{
    let nchains = ensemble.len();

    let next_lp = Mutex::new(vec![T::zero(); nchains]);
    let atomic_k = Mutex::new(0);
    let create_task = || {
        let atomic_k = &atomic_k;
        let next_lp = &next_lp;
        let ensemble = &ensemble;

        move || loop {
            let k: usize;
            {
                let mut k1 = atomic_k.lock().unwrap();
                k = *k1;
                *k1 += 1;
            }
            if k >= nchains {
                break;
            }
            let p = &ensemble[k];
            let lp = flogprob(p);
            {
                let mut lp1 = next_lp.lock().unwrap();
                lp1[k] = lp;
            }
        }
    };

    if njobs > 1 {
        scope(|s| {
            for _ in 0..njobs {
                s.spawn(|_| create_task()());
            }
        });
    } else {
        let task = create_task();
        task();
    }

    (ensemble, next_lp.into_inner().unwrap())
}

pub fn sample<T, U, V, W, X, F>(
    ensemble_logprob: &mut (W, X),
    flogprob: &F,
    delta: usize,
    cr: T,
    b: T,
    b_star: T,
    rng: &mut U,
    beta_list: &X,
    gamma_func: &Option<Box<dyn Fn(usize, usize) -> T>>,
    support_func: &Option<Box<dyn Fn(&V) -> bool>>,
    njobs: usize,
    perform_swap: bool,
)->Vec<bool> where
    T: Float + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Debug,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    U: Rng,
    V: Clone + FiniteLinearSpace<T> + Sync + Send + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    W: Clone + IndexMut<usize, Output = V> + HasLen + Sync + Send + ItemSwapable,
    X: Clone + IndexMut<usize, Output = T> + HasLen + Sync + InitFromLen + Send + ItemSwapable,
    F: Fn(&V) -> T + Send + Sync + ?Sized,
{
    let nchains = ensemble_logprob.1.len();
    let nbeta=beta_list.len();
    let nchains_per_beta=nchains/nbeta;

    assert!(nbeta*nchains_per_beta==nchains);

    if perform_swap{
        swap_walkers(ensemble_logprob, rng, beta_list).unwrap();
    }

    let proposed_points: Vec<_> = (0..nchains)
        .map(|i| {
            let mut p;
            let ibeta=i/nchains_per_beta;
            loop{
                p=propose_point(
                &ensemble_logprob.0,
                ibeta,
                i-ibeta*nchains_per_beta,
                delta,
                nbeta,
                cr,
                b,
                b_star,
                gamma_func,
                rng,
            );
                if let Some(sp)=support_func{
                    if sp(&p){
                        return p;
                    }
                }else{
                    return p;
                }
            }
        })
        .collect();

    let next_lp = Mutex::new(vec![T::zero(); nchains]);
    let atomic_k = Mutex::new(0);

    let create_task = || {
        let atomic_k = &atomic_k;
        let next_lp = &next_lp;
        let proposed_points = &proposed_points;
        move || loop {
            let k: usize;
            {
                let mut k1 = atomic_k.lock().unwrap();
                k = *k1;
                *k1 += 1;
            }
            if k >= nchains {
                break;
            }
            let p = &proposed_points[k];
            let lp = flogprob(p);
            {
                let mut lp1 = next_lp.lock().unwrap();
                lp1[k] = lp;
            }
        }
    };

    if njobs > 1 {
        scope(|s| {
            for _ in 0..njobs {
                s.spawn(|_| create_task()());
            }
        });
    } else {
        let task = create_task();
        task();
    }

    let next_lp = next_lp.into_inner().unwrap();
    let mut accepted=Vec::new();
    for (i, (proposed, lp)) in proposed_points
        .into_iter()
        .zip(next_lp.into_iter())
        .enumerate()
    {
        let ibeta=i/nchains_per_beta;
        let a=accept(
            &mut ensemble_logprob.0,
            &mut ensemble_logprob.1,
            ibeta,
            i-ibeta*nchains_per_beta,
            proposed,
            lp,
            beta_list,
            rng,
        );
        accepted.push(a);
    }
    //println!("{:?}", accepted);
    accepted  
}
