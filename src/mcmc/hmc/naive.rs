#![allow(clippy::too_many_arguments)]
use std::ops::{Add, Mul, Sub};

use super::utils::leapfrog;
use crate::linear_space::InnerProdSpace;
use num::traits::Float;
use rand::{
    distributions::{uniform::SampleUniform, Distribution, Standard, Uniform},
    Rng,
};
use rand_distr::StandardNormal;

use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

pub struct HmcParam<T>
where
    T: Float,
{
    target_accept_ratio: T,
    adj_factor: T,
}

impl<T> HmcParam<T>
where
    T: Float,
{
    pub fn new(target_accept_ratio: T, adj_factor: T) -> HmcParam<T> {
        HmcParam {
            target_accept_ratio,
            adj_factor,
        }
    }

    pub fn quick_adj(target_accept_ratio: T) -> HmcParam<T> {
        Self::new(target_accept_ratio, T::from(0.01).unwrap())
    }

    pub fn slow_adj(target_accept_ratio: T) -> HmcParam<T> {
        Self::new(target_accept_ratio, T::from(0.000_001).unwrap())
    }

    pub fn fixed(target_accept_ratio: T) -> HmcParam<T> {
        Self::new(target_accept_ratio, T::zero())
    }
}

impl<T> std::default::Default for HmcParam<T>
where
    T: Float,
{
    fn default() -> Self {
        HmcParam::new(T::from(0.99).unwrap(), T::from(0.000_001).unwrap())
    }
}

pub fn sample<T, U, V, F, G>(
    flogprob: &F,
    grad_logprob: &G,
    q0: &mut V,
    lp: &mut T,
    last_grad_logprob: &mut V,
    rng: &mut U,
    epsilon: &mut T,
    l: usize,
    param: &HmcParam<T>,
) -> bool
where
    T: Float + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    U: Rng,
    V: Clone + InnerProdSpace<T> + Sync + Send + Sized,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    F: Fn(&V) -> T,
    G: Fn(&V) -> V,
{
    let two = T::one() + T::one();

    let mut p = V::zeros_like(q0);
    for i in 0..p.dimension() {
        p[i] = rng.sample(StandardNormal);
    }

    let m_inv = T::one();
    let kinetic = |p: &V| p.dot(&p) / two * m_inv;

    let current_k = kinetic(&p);
    let mut q = q0.clone();
    let h_p = |p: &V| p * m_inv;
    let h_q = |q: &V| &grad_logprob(q) * (-T::one());
    //let mut last_hq_q=h_q(&q);
    let mut last_hq_q_tmp = last_grad_logprob as &V * (-T::one());
    for _i in 0..l {
        leapfrog(&mut q, &mut p, &mut last_hq_q_tmp, *epsilon, &h_q, &h_p);
    }
    let current_u = -*lp;
    let proposed_u = -flogprob(&q);
    let proposed_k = kinetic(&p);
    //println!("{:?}", (current_u-proposed_u+current_k-proposed_k));
    let dh = current_u - proposed_u + current_k - proposed_k;

    if dh.is_finite() && rng.sample(Uniform::new(T::zero(), T::one())) < dh.exp() {
        *q0 = q;
        *lp = -proposed_u;
        *last_grad_logprob = &last_hq_q_tmp * (-T::one());
        if rng.sample(Uniform::new(T::zero(), T::one())) < T::one() - param.target_accept_ratio {
            *epsilon = *epsilon * T::from(T::one() + param.adj_factor).unwrap();
        }
        true
    } else {
        if rng.sample(Uniform::new(T::zero(), T::one())) < param.target_accept_ratio {
            *epsilon = *epsilon / T::from(T::one() + param.adj_factor).unwrap();
        }
        false
    }
}

pub fn sample_ensemble_pt<T, U, V, F, G>(
    flogprob: &F,
    grad_logprob: &G,
    q0: &mut [V],
    lp: &mut [T],
    rng: &mut U,
    epsilon: &mut [T],
    beta_list: &[T],
    l: usize,
    param: &HmcParam<T>,
) -> Vec<usize>
where
    T: Float + SampleUniform + std::fmt::Debug + Sync + Send,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    U: Rng,
    V: Clone + InnerProdSpace<T> + Sync + Send + std::fmt::Debug,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    F: Fn(&V) -> T + Sync,
    G: Fn(&V) -> V + Sync,
{
    let mut last_grad_logprob: Vec<_> = q0.iter().map(|x| grad_logprob(x)).collect();
    sample_ensemble_pt_impl(
        flogprob,
        grad_logprob,
        q0,
        lp,
        &mut last_grad_logprob,
        rng,
        epsilon,
        beta_list,
        l,
        param,
    )
}

pub fn sample_ensemble_pt_impl<T, U, V, F, G>(
    flogprob: &F,
    grad_logprob: &G,
    q0: &mut [V],
    lp: &mut [T],
    last_grad_logprob: &mut [V],
    rng: &mut U,
    epsilon: &mut [T],
    beta_list: &[T],
    l: usize,
    param: &HmcParam<T>,
) -> Vec<usize>
where
    T: Float + SampleUniform + std::fmt::Debug + Sync + Send,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
    U: Rng,
    V: Clone + InnerProdSpace<T> + Sync + Send + std::fmt::Debug,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    F: Fn(&V) -> T + Sync,
    G: Fn(&V) -> V + Sync,
{
    let nbeta = beta_list.len();
    let mut accepted_cnt = vec![0; nbeta];
    let n_per_beta = q0.len() / nbeta;
    assert_eq!(q0.len(), lp.len());
    assert_eq!(q0.len(), last_grad_logprob.len());
    assert_eq!(epsilon.len(), nbeta);

    let two = T::one() + T::one();

    let mut p: Vec<_> = q0
        .iter()
        .map(|q1| {
            let mut p = V::zeros_like(q1);
            for i in 0..p.dimension() {
                p[i] = rng.sample(StandardNormal);
            }
            p
        })
        .collect();

    let m_inv = T::one();
    let kinetic = |p: &V| p.dot(&p) / two * m_inv;

    let current_k: Vec<_> = p.iter().map(kinetic).collect();
    //println!("{:?}", current_k);
    let mut q = q0.to_vec();

    //let mut last_hq_q=h_q(&q);

    let mut last_hq_q_tmp: Vec<_> = last_grad_logprob
        .iter()
        .enumerate()
        .map(|(i, x)| x * (-T::one() * beta_list[i / n_per_beta]))
        .collect();

    q.par_iter_mut()
        .zip(p.par_iter_mut().zip(last_hq_q_tmp.par_iter_mut()))
        .enumerate()
        .for_each(|(i, (q, (p, lhqq)))| {
            //q.iter_mut()
            //    .zip(p.iter_mut().zip(last_hq_q_tmp.iter_mut()))
            //    .enumerate()
            //    .for_each(|(i, (q, (p, lhqq)))| {
            let ibeta = i / n_per_beta;
            let beta = beta_list[ibeta];
            let e = epsilon[ibeta];
            let h_p = |p: &V| p * m_inv;
            let h_q = |q: &V| &grad_logprob(q) * (-beta);
            for _i in 0..l {
                leapfrog(q, p, lhqq, e, &h_q, &h_p);
            }
        });

    let current_u: Vec<_> = lp.iter().map(|&x| -x).collect();
    let proposed_u: Vec<_> = q.par_iter().map(|q1| -flogprob(q1)).collect();

    let proposed_k: Vec<_> = p.iter_mut().map(|p1| kinetic(&p1)).collect();
    //println!("{:?}", (current_u-proposed_u+current_k-proposed_k));

    let dh: Vec<_> = current_u
        .iter()
        .zip(
            proposed_u
                .iter()
                .zip(current_k.iter().zip(proposed_k.iter())),
        )
        .enumerate()
        .map(|(i, (&u0, (&u1, (&k0, &k1))))| {
            let ibeta = i / n_per_beta;
            let beta = beta_list[ibeta];
            beta * (u0 - u1) + k0 - k1
        })
        .collect();

    assert_eq!(dh.len(), q0.len());
    dh.iter()
        .zip(
            q0.iter_mut().zip(
                q.into_iter().zip(
                    lp.iter_mut().zip(
                        last_grad_logprob
                            .iter_mut()
                            .zip(proposed_u.into_iter().zip(last_hq_q_tmp.iter())),
                    ),
                ),
            ),
        )
        .enumerate()
        .for_each(|(i, (&dh1, (q01, (q1, (lp1, (lgl1, (pu, lhqt)))))))| {
            let ibeta = i / n_per_beta;
            //println!("{} {} {}", ibeta, n_per_beta, i);

            if dh1.is_finite() && rng.sample(Uniform::new(T::zero(), T::one())) < dh1.exp() {
                *q01 = q1;
                *lp1 = -pu;
                *lgl1 = lhqt * (-T::one());
                accepted_cnt[ibeta] += 1;
                //println!("{:?}", accepted_cnt);
                if rng.sample(Uniform::new(T::zero(), T::one()))
                    < T::one() - param.target_accept_ratio
                {
                    epsilon[ibeta] = epsilon[ibeta] * T::from(T::one() + param.adj_factor).unwrap();
                }
            } else if rng.sample(Uniform::new(T::zero(), T::one())) < param.target_accept_ratio {
                epsilon[ibeta] = epsilon[ibeta] / T::from(T::one() + param.adj_factor).unwrap();
            }
        });
    accepted_cnt
}
