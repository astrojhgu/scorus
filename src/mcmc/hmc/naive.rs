use std::ops::{Add, Sub, Mul};

use num_traits::Float;
use rand::Rng;
use rand::distributions::Standard;
use rand::distributions::Distribution;
use rand::distributions::uniform::SampleUniform;
use rand_distr::StandardNormal;
use super::utils::leapfrog;
use crate::linear_space::InnerProdSpace;


pub struct HmcParam<T>
where T: Float
{
    target_accept_ratio: T, 
    adj_factor: T,
}

impl<T> HmcParam<T>
where T: Float{
    pub fn new(target_accept_ratio:T , adj_factor:T )->HmcParam<T>{
        HmcParam{
            target_accept_ratio, 
            adj_factor
        }
    }

    pub fn quick_adj(target_accept_ratio:T)->HmcParam<T>{
        Self::new(target_accept_ratio, T::from(0.01).unwrap())
    }

    pub fn slow_adj(target_accept_ratio:T)->HmcParam<T>{
        Self::new(target_accept_ratio, T::from(0.000001).unwrap())
    }
}

impl<T> std::default::Default for HmcParam<T>
where T: Float
{
    fn default()->Self{
        HmcParam::new(T::from(0.99).unwrap(), T::from(0.000001).unwrap())
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
    param: &HmcParam<T>
)->bool
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
G: Fn(&V) -> V{
    let two=T::one()+T::one();
    
    let mut p=V::zeros_like(q0);
    for i in 0..p.dimension(){
        p[i]=rng.sample(StandardNormal);
    }

    let m_inv=T::one();
    let kinetic=|p: &V|{
        p.dot(&p)/two*m_inv
    };

    let current_k=kinetic(&p);
    let mut q=q0.clone();
    let h_p=|p: &V| &p.clone()*m_inv;
    let h_q=|q: &V| &grad_logprob(q)*(-T::one());
    //let mut last_hq_q=h_q(&q);
    let mut last_hq_q_tmp=last_grad_logprob as &V*(-T::one());
    for _i in 0..l{
        leapfrog(&mut q, &mut p, &mut last_hq_q_tmp, *epsilon, &h_q, &h_p);
    }
    let current_u=-*lp;
    let proposed_u=-flogprob(&q);
    let proposed_k=kinetic(&p);
    //println!("{:?}", (current_u-proposed_u+current_k-proposed_k));
    if rng.gen_range(T::zero(), T::one())<(current_u-proposed_u+current_k-proposed_k).exp(){
        *q0=q;
        *lp=-proposed_u;
        *last_grad_logprob=&last_hq_q_tmp*(-T::one());
        if rng.gen_range(T::zero(), T::one())<T::one()-param.target_accept_ratio{
            *epsilon=*epsilon*T::from(T::one()+param.adj_factor).unwrap();
        }
        true
    }else{
        if rng.gen_range(T::zero(), T::one())<param.target_accept_ratio{
            *epsilon=*epsilon/T::from(T::one()+param.adj_factor).unwrap();
        }
        false
    }
}
