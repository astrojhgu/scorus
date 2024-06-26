#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::type_complexity)]
#![allow(clippy::mutex_atomic)]

/**
 * Implemented according to https://arxiv.org/abs/1111.4246
 * and
 * https://github.com/mfouesneau/NUTS
 */
use std;

use num::traits::float::Float;
use rand::{
    distributions::{uniform::SampleUniform, Distribution, Standard, Uniform},
    Rng,
};
use rand_distr::{Exp1, StandardNormal};
//use std::marker::{Send, Sync};

use std::ops::{Add, Mul, Sub};

use crate::linear_space::InnerProdSpace;

pub struct NutsState<T>
where
    T: Float,
{
    pub m: usize,
    pub h_bar: T,
    pub epsilon: T,
    pub epsilon_bar: T,
    pub mu: T,
}

impl<T> NutsState<T>
where
    T: Float,
{
    pub fn new() -> NutsState<T> {
        NutsState::<T> {
            m: 0,
            h_bar: T::zero(),
            epsilon: T::zero(),
            epsilon_bar: T::zero(),
            mu: T::zero(),
        }
    }
}

impl<T> Default for NutsState<T>
where
    T: Float,
{
    fn default() -> NutsState<T> {
        Self::new()
    }
}

pub fn leapfrog<T, V, F>(theta: &V, r: &V, grad: &V, epsilon: T, fg: &F) -> (V, V, V, T)
where
    T: Float + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    V: Clone + InnerProdSpace<T>,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    F: Fn(&V) -> (T, V),
{
    let two = T::one() + T::one();
    let half = T::one() / two;
    let rprime = r + &(grad * (epsilon * half));

    let thetaprime = theta + &(&rprime * epsilon);

    let (logpprime, gradprime) = fg(&thetaprime);
    let rprime = &rprime + &(&gradprime * (epsilon * half));
    (thetaprime, rprime, gradprime, logpprime)
}

pub fn any_inf<T, V>(x: &V) -> bool
where
    T: Float + SampleUniform + std::fmt::Debug,
    V: Clone + InnerProdSpace<T>,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    for i in 0..x.dimension() {
        if !x[i].is_normal() {
            return true;
        }
    }
    false
}

#[allow(clippy::eq_op)]
pub fn normal_random_like<T, V, U>(x0: &V, rng: &mut U) -> V
where
    T: Float + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    V: Clone + InnerProdSpace<T>,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    U: Rng,
{
    let mut y = V::zeros_like(x0);
    for i in 0..y.dimension() {
        y[i] = rng.sample(StandardNormal);
    }
    y
}

pub fn find_reasonable_epsilon<T, V, F, U>(
    theta0: &V,
    grad0: &V,
    logp0: T,
    fg: &F,
    rng: &mut U,
) -> T
where
    T: Float + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    V: Clone + InnerProdSpace<T>,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    F: Fn(&V) -> (T, V),
    U: Rng,
{
    let two = T::one() + T::one();
    let half = T::one() / two;
    let mut epsilon = T::one();

    let r0 = normal_random_like(theta0, rng);
    let (_, mut rprime, mut gradprime, mut logpprime) = leapfrog(theta0, &r0, grad0, epsilon, fg);
    let mut k = T::one();
    while !logpprime.is_normal() || any_inf(&gradprime) {
        k = k / two;
        let a = leapfrog(theta0, &r0, grad0, epsilon * k, fg);
        rprime = a.1;
        gradprime = a.2;
        logpprime = a.3;
    }

    epsilon = epsilon * k / two;

    let mut logacceptprob = logpprime - logp0 - (rprime.dot(&rprime) - r0.dot(&r0)) * half;

    let a = if logacceptprob > half.ln() {
        T::one()
    } else {
        -T::one()
    };

    while a * logacceptprob > -a * two.ln() {
        epsilon = epsilon * two.powf(a);
        let a = leapfrog(theta0, &r0, grad0, epsilon, fg);
        rprime = a.1;
        logpprime = a.3;

        logacceptprob = logpprime - logp0 - (rprime.dot(&rprime) - r0.dot(&r0)) * half;
    }

    epsilon
}

pub fn stop_criterion<T, V>(thetaminus: &V, thetaplus: &V, rminus: &V, rplus: &V) -> bool
where
    T: Float + SampleUniform + std::fmt::Debug,
    V: Clone + InnerProdSpace<T>,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
{
    let dtheta = thetaplus - thetaminus;
    dtheta.dot(rminus) >= T::zero() && dtheta.dot(rplus) >= T::zero()
}

pub fn build_tree<T, V, F, U>(
    theta: &V,
    r: &V,
    grad: &V,
    logu: T,
    v: isize,
    j: usize,
    epsilon: T,
    f: &F,
    joint0: T,
    rng: &mut U,
) -> (V, V, V, V, V, V, V, V, T, usize, usize, T, usize)
where
    T: Float + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    V: Clone + InnerProdSpace<T> + Clone + std::fmt::Debug,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    F: Fn(&V) -> (T, V),
    U: Rng,
{
    let uniform = Uniform::new(T::zero(), T::one());
    let two = T::one() + T::one();
    let half = T::one() / two;
    let (
        mut thetaminus,
        mut rminus,
        mut gradminus,
        mut thetaplus,
        mut rplus,
        mut gradplus,
        mut thetaprime,
        mut gradprime,
        mut logpprime,
        mut nprime,
        mut sprime,
        mut alphaprime,
        mut nalphaprime,
    );
    if j == 0 {
        let a = leapfrog(theta, r, grad, T::from(v).unwrap() * epsilon, f);
        thetaprime = a.0;
        let rprime = a.1;
        gradprime = a.2;
        logpprime = a.3;

        let joint = logpprime - rprime.dot(&rprime) * half;

        nprime = if logu < joint { 1 } else { 0 };
        sprime = if (logu - T::from(1000).unwrap()) < joint {
            1
        } else {
            0
        };
        thetaminus = thetaprime.clone();
        thetaplus = thetaprime.clone();
        rminus = rprime.clone();
        rplus = rprime;
        gradminus = gradprime.clone();
        gradplus = gradprime.clone();
        alphaprime = T::min(T::one(), T::exp(joint - joint0));
        nalphaprime = 1;
    } else {
        let a = build_tree(theta, r, grad, logu, v, j - 1, epsilon, f, joint0, rng);
        thetaminus = a.0;
        rminus = a.1;
        gradminus = a.2;
        thetaplus = a.3;
        rplus = a.4;
        gradplus = a.5;
        thetaprime = a.6;
        gradprime = a.7;
        logpprime = a.8;
        nprime = a.9;
        sprime = a.10;
        alphaprime = a.11;
        nalphaprime = a.12;

        let thetaprime2;
        let gradprime2;
        let logpprime2;
        let nprime2;
        let sprime2;
        let alphaprime2;
        let nalphaprime2;
        if sprime == 1 {
            if v == -1 {
                let a = build_tree(
                    &thetaminus,
                    &rminus,
                    &gradminus,
                    logu,
                    v,
                    j - 1,
                    epsilon,
                    f,
                    joint0,
                    rng,
                );
                thetaminus = a.0;
                rminus = a.1;
                gradminus = a.2;
                thetaprime2 = a.6;
                gradprime2 = a.7;
                logpprime2 = a.8;
                nprime2 = a.9;
                sprime2 = a.10;
                alphaprime2 = a.11;
                nalphaprime2 = a.12;
            } else {
                let a = build_tree(
                    &thetaplus,
                    &rplus,
                    &gradplus,
                    logu,
                    v,
                    j - 1,
                    epsilon,
                    f,
                    joint0,
                    rng,
                );
                thetaplus = a.3;
                rplus = a.4;
                gradplus = a.5;
                thetaprime2 = a.6;
                gradprime2 = a.7;
                logpprime2 = a.8;
                nprime2 = a.9;
                sprime2 = a.10;
                alphaprime2 = a.11;
                nalphaprime2 = a.12;
            }

            if rng.sample(uniform)
                < T::from(nprime2).unwrap() / T::max(T::from(nprime + nprime2).unwrap(), T::one())
            {
                thetaprime = thetaprime2;
                gradprime = gradprime2;
                logpprime = logpprime2
            }

            nprime += nprime2;
            sprime = if sprime > 0
                && sprime2 > 0
                && stop_criterion(&thetaminus, &thetaplus, &rminus, &rplus)
            {
                1
            } else {
                0
            };
            alphaprime = alphaprime + alphaprime2;
            nalphaprime += nalphaprime2;
            //eprintln!("{:?} {:?}", thetaminus, thetaplus);
        }
    }
    (
        thetaminus,
        rminus,
        gradminus,
        thetaplus,
        rplus,
        gradplus,
        thetaprime,
        gradprime,
        logpprime,
        nprime,
        sprime,
        alphaprime,
        nalphaprime,
    )
}

pub fn nuts6<T, V, F, U>(
    f: &F,
    theta0: &mut V,
    logp0: &mut T,
    grad0: &mut V,
    delta: T,
    nutss: &mut NutsState<T>,
    burning: bool,
    rng: &mut U,
) where
    T: Float + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    Exp1: Distribution<T>,
    V: Clone + InnerProdSpace<T> + Clone + std::fmt::Debug,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    F: Fn(&V) -> (T, V),
    U: Rng,
{
    let two = T::one() + T::one();
    let half = T::one() / two;
    //let (mut logp0, mut grad0)=f(theta0);
    let gamma = T::from(0.05).unwrap();
    let t0 = 10;
    let kappa = T::from(0.75).unwrap();

    if nutss.m == 0 {
        nutss.epsilon_bar = T::one();
        nutss.h_bar = T::zero();
        nutss.epsilon = find_reasonable_epsilon(theta0, grad0, *logp0, f, rng);
        nutss.mu = (T::from(10).unwrap() * nutss.epsilon).ln();
        nutss.m = 1;
    }

    let r0 = normal_random_like(theta0, rng);
    //println!("{}, r0={:?}",nutss.m, r0);
    let joint = *logp0 - r0.dot(&r0) * half;
    let logu = joint - rng.sample(Exp1);

    let mut thetaminus = theta0.clone();
    let mut thetaplus = theta0.clone();
    let mut rminus = r0.clone();
    let mut rplus = r0;
    let mut gradminus = grad0.clone();
    let mut gradplus = grad0.clone();

    let mut j = 0;
    let mut n = 1;
    let mut s = true;
    let (
        mut thetaprime,
        mut gradprime,
        mut logpprime,
        mut nprime,
        mut sprime,
        mut alpha,
        mut nalpha,
    );
    while s {
        let v = if rng.sample(Uniform::new(T::zero(), T::one())) < half {
            1
        } else {
            -1
        };
        if v == -1 {
            let a = build_tree(
                &thetaminus,
                &rminus,
                &gradminus,
                logu,
                v,
                j,
                nutss.epsilon,
                f,
                joint,
                rng,
            );
            thetaminus = a.0;
            rminus = a.1;
            gradminus = a.2;
            thetaprime = a.6;
            gradprime = a.7;
            logpprime = a.8;
            nprime = a.9;
            sprime = a.10;
            alpha = a.11;
            nalpha = a.12;
        } else {
            let a = build_tree(
                &thetaplus,
                &rplus,
                &gradplus,
                logu,
                v,
                j,
                nutss.epsilon,
                f,
                joint,
                rng,
            );
            thetaplus = a.3;
            rplus = a.4;
            gradplus = a.5;
            thetaprime = a.6;
            gradprime = a.7;
            logpprime = a.8;
            nprime = a.9;
            sprime = a.10;
            alpha = a.11;
            nalpha = a.12;
        }

        let tmp = T::one().min(T::from(nprime).unwrap() / T::from(n).unwrap());
        //eprintln!("{}", nprime);
        if sprime == 1 && rng.sample(Uniform::new(T::zero(), T::one())) < tmp {
            *theta0 = thetaprime.clone();
            *logp0 = logpprime;
            *grad0 = gradprime.clone();
        }

        n += nprime;
        s = sprime > 0 && stop_criterion(&thetaminus, &thetaplus, &rminus, &rplus);
        j += 1;

        let mut eta = T::one() / T::from(nutss.m + t0).unwrap();
        nutss.h_bar =
            (T::one() - eta) * nutss.h_bar + eta * (delta - alpha / T::from(nalpha).unwrap());
        if burning {
            nutss.epsilon =
                (nutss.mu - T::sqrt(T::from(nutss.m).unwrap()) / gamma * nutss.h_bar).exp();
            eta = T::from(nutss.m).unwrap().powf(-kappa);
            nutss.epsilon_bar =
                T::exp((T::one() - eta) * T::ln(nutss.epsilon_bar) + eta * T::ln(nutss.epsilon));
        } else {
            nutss.epsilon = nutss.epsilon_bar;
        }
    }
    nutss.m += 1;
}
