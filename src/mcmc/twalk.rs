#![allow(clippy::many_single_char_names)]
#![allow(clippy::type_complexity)]
#![allow(clippy::mutex_atomic)]
use num_traits::float::{Float, FloatConst};
use rayon::scope;
use std;
use std::sync::Mutex;
//use num_traits::identities::{one, zero};
use num_traits::NumCast;
use rand::distributions::uniform::SampleUniform;
use rand::distributions::Distribution;
use rand::distributions::Standard;
use rand::Rng;
use rand_distr::StandardNormal;
//use std::marker::{Send, Sync};
use std::ops::{Add, Mul, Sub};
//use std::sync::Arc;
use crate::utils::{HasLen, InitFromLen};
use rand::seq::SliceRandom;
use std::ops::IndexMut;
//use super::mcmc_errors::McmcErr;
//use super::utils::{draw_z, scale_vec};

use crate::linear_space::FiniteLinearSpace;
//use crate::utils::HasLen;
//use crate::utils::InitFromLen;

#[derive(Clone, Copy, Debug)]
pub enum TWalkKernal {
    Walk,
    Traverse,
    Blow,
    Hop,
}

impl TWalkKernal {
    pub fn random<T, U>(fw: &[T; 4], rng: &mut U) -> TWalkKernal
    where
        T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
        Standard: Distribution<T>,
        U: Rng,
    {
        let u = rng.gen_range(T::zero(), fw[fw.len() - 1]);
        if u < fw[0] {
            TWalkKernal::Walk
        } else if fw[0] <= u && u < fw[1] {
            TWalkKernal::Traverse
        } else if fw[1] <= u && u < fw[2] {
            TWalkKernal::Blow
        } else {
            TWalkKernal::Hop
        }
    }

    pub fn to_usize(self) -> usize {
        match self {
            TWalkKernal::Walk => 0,
            TWalkKernal::Traverse => 1,
            TWalkKernal::Blow => 2,
            TWalkKernal::Hop => 3,
        }
    }
}

pub struct TWalkParams<T>
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
{
    pub aw: T,
    pub at: T,
    pub pphi: T,
    pub fw: [T; 4],
}

impl<T> TWalkParams<T>
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
{
    pub fn new(n: usize) -> Self {
        let n1phi = T::from(4.0).unwrap();
        let pphi = T::from(n).unwrap().min(n1phi) / T::from(n).unwrap();
        let fw: Vec<_> = vec![0.4918, 0.4918, 0.0082 + 0.0082, 0.0]
            .into_iter()
            .map(|x| T::from(x).unwrap())
            .scan(T::zero(), |st, x| {
                *st = *st + x;
                Some(*st)
            })
            .collect();
        let fw = [fw[0], fw[1], fw[2], fw[3]];
        //println!("{:?}", fw);
        TWalkParams {
            aw: T::from(1.5).unwrap(),
            at: T::from(6.0).unwrap(),
            pphi,
            fw,
        }
    }

    pub fn with_fw(mut self, fw: [T; 4]) -> Self {
        let fw: Vec<T> = vec![fw[0], fw[1], fw[2], fw[3]]
            .into_iter()
            .map(|x| T::from(x).unwrap())
            .scan(T::zero(), |st, x| {
                *st = *st + x;
                Some(*st)
            })
            .collect();
        let fw = [fw[0], fw[1], fw[2], fw[3]];
        self.fw = fw;
        self
    }

    pub fn with_pphi(self, pphi: T) -> Self {
        TWalkParams { pphi, ..self }
    }
}

#[derive(Clone)]
pub struct TWalkState<T, V>
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    V: Clone + FiniteLinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    pub x: V,
    pub xp: V,
    pub u: T,
    pub up: T,
    //pub kernel: TWalkKernal,
}

impl<T, V> TWalkState<T, V>
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    V: Clone + FiniteLinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    pub fn new<F>(x: &V, xp: &V, flogprob: &F) -> TWalkState<T, V>
    where
        F: Fn(&V) -> T + ?Sized,
    {
        TWalkState {
            x: x.clone(),
            xp: xp.clone(),
            u: flogprob(x),
            up: flogprob(xp),
            //kernel: TWalkKernal::Nothing,
        }
    }

    pub fn dimension(&self) -> usize {
        self.x.dimension()
    }

    pub fn diff_count(&self, rhs: &Self) -> (usize, usize) {
        let mut result = (0, 0);
        for i in 0..self.dimension() {
            if self.x[i] != rhs.x[i] {
                result.0 += 1;
            }
            if self.xp[i] != rhs.xp[i] {
                result.1 += 1;
            }
        }
        result
    }

    pub fn elements_changed(&self, rhs: &Self) -> (Vec<bool>, Vec<bool>) {
        let mut result = (vec![false; self.dimension()], vec![false; self.dimension()]);
        for i in 0..self.dimension() {
            result.0[i] = self.x[i] != rhs.x[i];
            result.1[i] = self.xp[i] != rhs.xp[i];
        }
        result
    }
}

#[derive(Clone)]
pub struct TWalkResult<T, V>
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    V: Clone,
{
    pub a: T,
    pub original: V,
    pub proposed: V,
    pub u: T,
    pub u_proposed: T,
    pub accepted: bool,
    pub update_flags: Vec<bool>,
    pub last_kernel: TWalkKernal,
}

pub fn elements_changed<T, V>(x1: &V, x2: &V) -> Vec<bool>
where
    T: Float + NumCast + std::cmp::PartialOrd + std::fmt::Debug,
    V: Clone + FiniteLinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    let mut result = vec![false; x1.dimension()];
    for i in 0..x1.dimension() {
        result[i] = x1[i] != x2[i]
    }
    result
}

fn all_different<T, V>(x1: &V, x2: &V) -> bool
where
    T: Float + NumCast + std::cmp::PartialOrd + std::fmt::Debug,
    V: Clone + FiniteLinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    for i in 0..x1.dimension() {
        if x1[i] == x2[i] {
            return false;
        }
    }
    true
}

fn sqr_norm<T, V>(x: &V) -> T
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    V: Clone + FiniteLinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    let mut result = T::zero();
    let n = x.dimension();
    for i in 0..n {
        result = result + x[i].powi(2);
    }
    result
}

pub fn gen_update_flags<T, U>(n: usize, pphi: T, rng: &mut U) -> Vec<bool>
where
    T: Float + SampleUniform,
    U: Rng,
{
    loop {
        let result: Vec<_> = (0..n)
            .map(|_| rng.gen_range(T::zero(), T::one()) < pphi)
            .collect();
        if result.iter().any(|&b| b) {
            break result;
        }
    }
}

pub fn sim_walk<T, U, V>(x: &V, xp: &V, rng: &mut U, param: &TWalkParams<T>) -> (V, Vec<bool>)
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    U: Rng,
    V: Clone + FiniteLinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    let aw = param.aw;
    let n = x.dimension();
    let mut result = x.clone();
    let dx = x - xp;
    let two = T::one() + T::one();
    let update_flags = gen_update_flags(n, param.pphi, rng);
    //println!("{:?}", update_flags.iter().enumerate().filter(|(i, &f)| f).collect::<Vec<_>>());
    for i in 0..n {
        if update_flags[i] {
            let u = rng.gen_range(T::zero(), T::one());
            let z = (param.aw / (T::one() + aw)) * (aw * u.powi(2) + two * u - T::one());
            result[i] = x[i] + dx[i] * z;
        }
        //print!("{:?} ", result[i]);
    }
    //println!();
    (result, update_flags)
}

pub fn sim_beta<T, U>(rng: &mut U, param: &TWalkParams<T>) -> T
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    U: Rng,
{
    let two = T::one() + T::one();
    let at = param.at;
    let x = rng.gen_range(T::zero(), T::one());
    if x == T::zero() {
        x
    } else if rng.gen_range(T::zero(), T::one()) < (at - T::one()) / (two * at) {
        x.powf(T::one() / (at + T::one()))
    } else {
        x.powf(T::one() / (T::one() - at))
    }
}

pub fn sim_traverse<T, U, V>(
    x: &V,
    xp: &V,
    beta: T,
    rng: &mut U,
    param: &TWalkParams<T>,
) -> (V, Vec<bool>)
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    U: Rng,
    V: Clone + FiniteLinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    let n = x.dimension();

    let mut result = x.clone();
    let update_flags = gen_update_flags(n, param.pphi, rng);
    //println!("{:?}", update_flags.iter().enumerate().filter(|(i, &f)| f).collect::<Vec<_>>());
    for i in 0..n {
        if update_flags[i] {
            result[i] = xp[i] + beta * (xp[i] - x[i]);
        }
        //print!("{:?} ", result[i]);
    }
    //println!();
    (result, update_flags)
}

pub fn sim_blow<T, U, V>(x: &V, xp: &V, rng: &mut U, param: &TWalkParams<T>) -> (V, Vec<bool>)
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    U: Rng,
    V: Clone + FiniteLinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    let n = x.dimension();

    let mut sigma = T::zero();
    let dx = xp - x;
    let update_flags = gen_update_flags(n, param.pphi, rng);
    //println!("{:?}", update_flags.iter().enumerate().filter(|(i, &f)| f).collect::<Vec<_>>());
    for i in 0..n {
        if update_flags[i] {
            sigma = sigma.max(dx[i].abs());
        }
    }
    let mut result = x.clone();
    for i in 0..n {
        if update_flags[i] {
            result[i] = xp[i] + sigma * rng.sample(StandardNormal);
        }
    }

    (result, update_flags)
}

pub fn g_blow_u<T, V>(h: &V, x: &V, xp: &V, phi: &[bool]) -> T
where
    T: Float + FloatConst + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    V: Clone + FiniteLinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    let n = x.dimension();
    let nphi = phi.iter().filter(|&&b| b).count();
    let mut sigma = T::zero();
    let dx = xp - x;
    for i in 0..n {
        if phi[i] {
            sigma = sigma.max(dx[i].abs());
        }
    }
    let two = T::one() + T::one();
    let ln2pi = (two * T::PI()).ln();
    let nphi = T::from(nphi).unwrap();
    if nphi > T::zero() {
        (nphi / two) * ln2pi + nphi * sigma.ln() + sqr_norm(&(h - xp)) / two / sigma.powi(2)
    } else {
        T::zero()
    }
}

pub fn sim_hop<T, U, V>(x: &V, xp: &V, rng: &mut U, param: &TWalkParams<T>) -> (V, Vec<bool>)
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    U: Rng,
    V: Clone + FiniteLinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    let n = x.dimension();
    let mut sigma = T::zero();
    let dx = xp - x;
    let update_flags = gen_update_flags(n, param.pphi, rng);

    for i in 0..n {
        if update_flags[i] {
            sigma = sigma.max(dx[i].abs());
        }
    }

    sigma = sigma / T::from(3).unwrap();
    let mut result = x.clone();
    for i in 0..n {
        if update_flags[i] {
            result[i] = x[i] + sigma * rng.sample(StandardNormal);
        }
    }

    (result, update_flags)
}

pub fn g_hop_u<T, V>(h: &V, x: &V, xp: &V, phi: &[bool]) -> T
where
    T: Float + FloatConst + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    V: Clone + FiniteLinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    g_blow_u(h, x, xp, phi)
}

//pub fn calc_a<T, V>(yp: &V, xp: &V, up_prop: T, up: T, phi: &[bool], kernel: TWalkKernal, beta: Option<T>)->T
pub fn calc_a<T, V>(
    x: &V,
    (xp, up): (&V, T),
    (yp, up_prop): (&V, T),
    phi: &[bool],
    kernel: TWalkKernal,
    beta: Option<T>,
) -> T
where
    T: Float + FloatConst + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    V: Clone + FiniteLinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    let nphi = phi.iter().filter(|&&b| b).count();

    if nphi == 0 || !all_different(yp, x) {
        T::zero()
    } else {
        match kernel {
            TWalkKernal::Walk => (up_prop - up).exp(),
            TWalkKernal::Traverse => {
                ((up_prop - up) - T::from(nphi as isize - 2).unwrap() * beta.unwrap().ln()).exp()
            }
            TWalkKernal::Blow => {
                let w1 = g_blow_u(&yp, xp, x, phi);
                let w2 = g_blow_u(xp, &yp, x, phi);
                ((up_prop - up) + (w2 - w1)).exp()
            }
            TWalkKernal::Hop => {
                let w1 = g_hop_u(&yp, xp, x, &phi);
                let w2 = g_hop_u(xp, &yp, x, &phi);
                ((up_prop - up) + (w2 - w1)).exp()
            }
        }
    }
}

pub fn propose_move<T, V, U>(
    xp: &V,
    x: &V,
    rng: &mut U,
    param: &TWalkParams<T>,
) -> (V, Vec<bool>, TWalkKernal)
where
    T: Float + FloatConst + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    U: Rng,
    V: Clone + FiniteLinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    let kernel = TWalkKernal::random(&param.fw, rng);
    let (proposed, phi) = match kernel {
        TWalkKernal::Walk => sim_walk(xp, x, rng, param),
        TWalkKernal::Traverse => {
            let beta = sim_beta(rng, param);
            sim_traverse(xp, x, beta, rng, param)
        }
        TWalkKernal::Blow => sim_blow(xp, x, rng, param),
        TWalkKernal::Hop => sim_hop(xp, x, rng, param),
    };
    (proposed, phi, kernel)
}

pub fn sample_st<T, U, V, F>(
    flogprob: &F,
    state: &mut TWalkState<T, V>,
    param: &TWalkParams<T>,
    rng: &mut U,
) where
    T: Float + FloatConst + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    U: Rng,
    V: Clone + FiniteLinearSpace<T> + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    F: Fn(&V) -> T + ?Sized,
{
    let (yp1, phi1, kernel1) = propose_move(&state.xp, &state.x, rng, param);
    let (yp2, phi2, kernel2) = propose_move(&state.x, &state.xp, rng, param);

    let up_prop1 = flogprob(&yp1);
    let up_prop2 = flogprob(&yp2);
    let a1 = calc_a(
        &state.x,
        (&state.xp, state.up),
        (&yp1, up_prop1),
        &phi1,
        kernel1,
        Some(sim_beta(rng, param)),
    );
    let a2 = calc_a(
        &state.xp,
        (&state.x, state.u),
        (&yp2, up_prop2),
        &phi2,
        kernel2,
        Some(sim_beta(rng, param)),
    );

    if rng.gen_range(T::zero(), T::one()) < a1 {
        state.xp = yp1;
        state.up = up_prop1;
    }

    if rng.gen_range(T::zero(), T::one()) < a2 {
        state.x = yp2;
        state.u = up_prop2;
    }
}

pub fn sample<T, U, V, W, X, F>(
    flogprob: &F,
    ensemble_logprob: &mut (W, X),
    param: &TWalkParams<T>,
    rng: &mut U,
    nthreads: usize,
) where
    T: Float
        + FloatConst
        + NumCast
        + std::cmp::PartialOrd
        + SampleUniform
        + std::fmt::Debug
        + std::marker::Send,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    U: Rng,
    V: Clone + FiniteLinearSpace<T> + Sized + std::marker::Sync,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    W: Clone + IndexMut<usize, Output = V> + HasLen + Sync + Send,
    X: Clone + IndexMut<usize, Output = T> + HasLen + Sync + InitFromLen + Send,
    F: Fn(&V) -> T + Send + Sync + ?Sized,
{
    let nwalkers = ensemble_logprob.0.len();

    let pair_id: Vec<_> = {
        let mut a: Vec<_> = (0..nwalkers).collect();
        a.shuffle(rng);
        a.chunks(2).map(|a| (a[0], a[1])).collect()
    };
    assert!(pair_id.len() * 2 == nwalkers);

    let proposed_points: Vec<_> = pair_id
        .iter()
        .map(|&(i1, i2)| propose_move(&ensemble_logprob.0[i1], &ensemble_logprob.0[i2], rng, param))
        .collect();

    let logprobs = Mutex::new(vec![T::zero(); proposed_points.len()]);
    let atomic_k = Mutex::new(0);

    let create_task = || {
        let atomic_k = &atomic_k;
        let logprobs = &logprobs;
        let proposed_points = &proposed_points;
        move || loop {
            let k: usize;
            {
                let mut k1 = atomic_k.lock().unwrap();
                k = *k1;
                *k1 += 1;
            }
            if k * 2 >= nwalkers {
                break;
            }
            let lp = flogprob(&proposed_points[k].0);
            {
                let mut lps = logprobs.lock().unwrap();
                lps[k] = lp;
            }
        }
    };

    if nthreads == 1 {
        let task = create_task();
        task();
    } else {
        scope(|s| {
            for _ in 0..nthreads {
                s.spawn(|_| create_task()());
            }
        });
    }

    let logprobs: Vec<_> = proposed_points.iter().map(|x| flogprob(&x.0)).collect();

    for ((&(i1, i2), &up_prop1), (yp1, phi, k)) in pair_id
        .iter()
        .zip(logprobs.iter())
        .zip(proposed_points.into_iter())
    {
        //let up_prop2=flogprob(&yp2);
        let a1 = calc_a(
            &ensemble_logprob.0[i2],
            (&ensemble_logprob.0[i1], ensemble_logprob.1[i1]),
            (&yp1, up_prop1),
            &phi,
            k,
            Some(sim_beta(rng, param)),
        );
        //let a2=calc_a(&ensemble_logprob.0[i1], (&ensemble_logprob.0[i2], ensemble_logprob.1[i2]), (&yp2, up_prop2), &phi2, kernel2, Some(sim_beta(rng, param)));

        if rng.gen_range(T::zero(), T::one()) < a1 {
            ensemble_logprob.0[i1] = yp1;
            ensemble_logprob.1[i1] = up_prop1;
        }
        /*
        if rng.gen_range(T::zero(), T::one()) < a2 {
            ensemble_logprob.0[i2] = yp2;
            ensemble_logprob.1[i2] = up_prop2;
        }*/
    }
}
