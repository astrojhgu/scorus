#![allow(clippy::many_single_char_names)]
#![allow(clippy::type_complexity)]
#![allow(clippy::mutex_atomic)]
//use rayon::scope;
use std;

use num_traits::float::{Float, FloatConst};
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
    pub fn random<T, U>(fw: &[T], rng: &mut U) -> TWalkKernal
    where
        T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
        Standard: Distribution<T>,
        U: Rng,
    {
        let u = rng.gen_range(T::zero(), fw[fw.len() - 1]);
        if u < fw[1] {
            TWalkKernal::Walk
        } else if fw[1] <= u && u < fw[2] {
            TWalkKernal::Traverse
        } else if fw[2] <= u && u < fw[3] {
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
    pub fw: Vec<T>,
}

impl<T> TWalkParams<T>
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
{
    pub fn new(n: usize) -> Self {
        let n1phi = T::from(4.0).unwrap();
        let pphi = T::from(n).unwrap().min(n1phi) / T::from(n).unwrap();
        let fw: Vec<_> = vec![0.0000, 0.4918, 0.4918, 0.0082 + 0.0082, 0.0]
            .into_iter()
            .map(|x| T::from(x).unwrap())
            .scan(T::zero(), |st, x| {
                *st = *st + x;
                Some(*st)
            })
            .collect();
        //println!("{:?}", fw);
        TWalkParams {
            aw: T::from(1.5).unwrap(),
            at: T::from(6.0).unwrap(),
            pphi,
            fw,
        }
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

pub fn sample_st<T, U, V, F>(
    flogprob: &F,
    state: &mut TWalkState<T, V>,
    param: &TWalkParams<T>,
    rng: &mut U,
) -> TWalkResult<T, V>
where
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
    let two = T::one() + T::one();
    let kernel = TWalkKernal::random(&param.fw, rng);
    //println!("{:?}", kernel);
    let mut proposed = state.clone();
    let (x, xp, up) = if rng.gen_range(T::zero(), two) < T::one() {
        (&proposed.x, &mut proposed.xp, &mut proposed.up)
    } else {
        (&proposed.xp, &mut proposed.x, &mut proposed.u)
    };

    let mut result = match kernel {
        TWalkKernal::Walk => {
            let (yp, phi) = sim_walk(xp, x, rng, param);
            let nphi = phi.iter().filter(|&&b| b).count();
            let (up_prop, a) = if all_different(&yp, x) && nphi > 0 {
                let up_prop = flogprob(&yp);
                (up_prop, (up_prop - *up).exp())
            } else {
                (-T::infinity(), T::zero())
            };
            let result = TWalkResult {
                a,
                original: xp.clone(),
                proposed: yp.clone(),
                u: *up,
                u_proposed: up_prop,
                accepted: false,
                last_kernel: kernel,
                update_flags: phi,
            };
            *xp = yp;
            *up = up_prop;
            result
        }
        TWalkKernal::Traverse => {
            let beta = sim_beta(rng, param);
            //println!("beta={:?}", beta);
            let (yp, phi) = sim_traverse(xp, x, beta, rng, param);
            let nphi = phi.iter().filter(|&&b| b).count();

            let (up_prop, a) = if all_different(&yp, x) && nphi > 0 {
                let up_prop = flogprob(&yp);
                let a = ((up_prop - *up) - T::from(nphi as isize - 2).unwrap() * beta.ln()).exp();
                (up_prop, a)
            } else {
                (-T::infinity(), T::zero())
            };
            let result = TWalkResult {
                a,
                original: xp.clone(),
                proposed: yp.clone(),
                u: *up,
                u_proposed: up_prop,
                accepted: false,
                last_kernel: kernel,
                update_flags: phi,
            };
            *xp = yp;
            *up = up_prop;
            result
        }
        TWalkKernal::Blow => {
            let (yp, phi) = sim_blow(xp, x, rng, param);
            let nphi = phi.iter().filter(|&&b| b).count();

            let (up_prop, a) = if all_different(&yp, x) && nphi > 0 {
                let up_prop = flogprob(&yp);
                let w1 = g_blow_u(&yp, xp, x, &phi);
                let w2 = g_blow_u(xp, &yp, x, &phi);
                let a = ((up_prop - *up) + (w2 - w1)).exp();
                (up_prop, a)
            } else {
                (-T::infinity(), T::zero())
            };
            let result = TWalkResult {
                a,
                original: xp.clone(),
                proposed: yp.clone(),
                u: *up,
                u_proposed: up_prop,
                accepted: false,
                last_kernel: kernel,
                update_flags: phi,
            };
            *xp = yp;
            *up = up_prop;
            result
        }
        TWalkKernal::Hop => {
            let (yp, phi) = sim_hop(xp, &x, rng, param);
            let nphi = phi.iter().filter(|&&b| b).count();
            let (up_prop, a) = if all_different(&yp, x) && nphi > 0 {
                let up_prop = flogprob(&yp);
                let w1 = g_hop_u(&yp, xp, x, &phi);
                let w2 = g_hop_u(xp, &yp, x, &phi);
                let a = ((up_prop - *up) + (w2 - w1)).exp();
                (up_prop, a)
            } else {
                (-T::infinity(), T::zero())
            };
            let result = TWalkResult {
                a,
                original: xp.clone(),
                proposed: yp.clone(),
                u: *up,
                u_proposed: up_prop,
                accepted: false,
                last_kernel: kernel,
                update_flags: phi,
            };
            *xp = yp;
            *up = up_prop;
            result
        }
    };

    if rng.gen_range(T::zero(), T::one()) < result.a {
        *state = proposed;
        result.accepted = true;
    }
    result
}
