#![allow(clippy::many_single_char_names)]
#![allow(clippy::type_complexity)]
#![allow(clippy::mutex_atomic)]
//use rayon::scope;
use std;

use num_traits::float::{Float};
//use num_traits::identities::{one, zero};
use num_traits::NumCast;
use rand::distributions::uniform::SampleUniform;
use rand::distributions::Distribution;
use rand_distr::StandardNormal;
use rand::Rng;
use rand::seq::SliceRandom;

//use std::marker::{Send, Sync};
use std::ops::{Add, Mul, Sub};
use std::collections::VecDeque;
use crate::linear_space::FiniteLinearSpace;


use super::utils::{multinomial, calc_gamma, replace_flag};

pub struct DreamState<T,V>
where 
T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
StandardNormal: Distribution<T>,
V: Clone + FiniteLinearSpace<T> + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>
{
    pub b: T,
    pub b_star: T,
    pub x: Vec<V>,
    pub lgp: Vec<T>,
    pub niter: usize,
}

impl<T,V> DreamState<T,V>
where 
T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
StandardNormal: Distribution<T>,
V: Clone + FiniteLinearSpace<T> + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>{
    pub fn new<F>(x: Vec<V>, flogprob: &F, b: T, b_star: T)->DreamState<T,V>
    where F: Fn(&V)->T+?Sized
    {
        let lgp=x.iter().map(|x|{
            flogprob(x)
        }).collect();
        DreamState{
            b, 
            b_star, 
            x,
            lgp,
            niter: 0,
        }
    }

    pub fn nchains(&self)->usize{
        self.x.len()
    }


    pub fn sample_dx<U: Rng>(&self, i: usize, delta: usize, rng: &mut U)->V{
        //step 1a
        let delta=delta.min(self.x.len()/2);
        let mut j_list:Vec<_>=(0..self.x.len()).filter(|&x| x!=i).collect();
        j_list.shuffle(rng);
        let n_list:Vec<_>=j_list.iter().take(delta).cloned().collect();
        let j_list:Vec<_>=j_list.iter().skip(delta).take(delta).cloned().collect();
        let start_point=&self.x[i];
        let mut dx=start_point-start_point;
        for (&j, &n) in j_list.iter().zip(n_list.iter()){
            dx=&dx+&(&self.x[j]-&self.x[n]);
        }
        dx
    }


    pub fn propose_point<R: Rng>(&self, i: usize, delta: usize, cr: T, rng: &mut R)->V{
        let dx=self.sample_dx(i, delta, rng);
        let (flag, dprime)=replace_flag(dx.dimension(), cr, rng);
        //println!("{:?} {}", flag, dprime);
        let gamma=calc_gamma(delta, dprime);
        let mut proposed=self.x[i].clone();
        for d in 0..proposed.dimension(){
            if flag[d]{
                proposed[d]=proposed[d]+ dx[d] * gamma*(T::one()+rng.gen_range(-self.b, self.b))+rng.sample(StandardNormal)*self.b_star;
            }
        }
        proposed
    }

    pub fn accept<R, F>(&mut self, i: usize, proposed: V, flogprob: &F, rng: &mut R)
    where 
        R: Rng, 
        F: Fn(&V)->T+?Sized
    {
        let lgp=flogprob(&proposed);
        let alpha=(lgp-self.lgp[i]).exp();
        if rng.gen_range(T::zero(), T::one()) < alpha{
            self.x[i]=proposed;
            self.lgp[i]=lgp;
        }
    }

    pub fn evolve_single_chain<R,F>(&mut self, i: usize, flogprob: &F,delta: usize, cr: T, rng: &mut R)
    where
        R: Rng, 
        F: Fn(&V)->T+?Sized
    {
        let proposed=self.propose_point(i, delta, cr, rng);
        self.accept(i, proposed, flogprob, rng);
    }

    pub fn evolve_all_chains<R,F>(&mut self, flogprob: &F, delta:usize, cr: T, rng: &mut R)
    where
        R: Rng, 
        F: Fn(&V)->T+?Sized
    {
        for i in 0..self.nchains(){
            self.evolve_single_chain(i, flogprob, delta, cr, rng);
        }
    }    
}
