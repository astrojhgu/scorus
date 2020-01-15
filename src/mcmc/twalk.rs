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
use rand_distr::StandardNormal;
use rand::Rng;
//use std::marker::{Send, Sync};
use std::ops::{Add, Mul, Sub};
//use std::sync::Arc;

//use super::mcmc_errors::McmcErr;
//use super::utils::{draw_z, scale_vec};

use crate::linear_space::FiniteLinearSpace;
//use crate::utils::HasLen;
//use crate::utils::InitFromLen;

pub struct TWalkParams<T>
where T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
{
    pub aw: T,
    pub at: T,
    pub pphi: T,
    pub fw: Vec<T>,
}

#[derive(Clone, Copy, Debug)]
pub enum TWalkKernal{
    Nothing,
    Walk,
    Traverse,
    Blow,
    Hop
}

impl TWalkKernal{
    pub fn random<T, U>(fw: &Vec<T>, rng: &mut U)->TWalkKernal
    where T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
    Standard: Distribution<T>,
    U: Rng,
    {
        let u=rng.gen_range(T::zero(), fw[fw.len()-1]);
        if u<fw[0]{
            TWalkKernal::Nothing
        }else if fw[0]<=u && u<fw[1]{
            TWalkKernal::Walk
        }else if fw[1] <= u && u<fw[2]{
            TWalkKernal::Traverse
        }else if fw[2] <= u && u<fw[3]{
            TWalkKernal::Blow
        }else{
            TWalkKernal::Hop
        }
    }
}



#[derive(Clone)]
pub struct TWalkState<T, V>
where T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
V: Clone + FiniteLinearSpace<T> + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,{
    pub x: V,
    pub xp: V,
    pub u: Option<T>,
    pub up: Option<T>,
    pub kernel: TWalkKernal,
}

impl<T,V> TWalkState<T,V>
where T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
V: Clone + FiniteLinearSpace<T> + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,
{
    pub fn new<F>(x: &V, xp: &V, flogprob: &F)->TWalkState<T,V>
    where
        F: Fn(&V) -> T + ?Sized,
    {
        TWalkState{
            x:x.clone(),
            xp: xp.clone(),
            u: Some(flogprob(x)),
            up: Some(flogprob(xp)),
            kernel: TWalkKernal::Nothing,
        }
    }

    pub fn dimension(&self)->usize{
        self.x.dimension()
    }
}


impl<T> TWalkParams<T>
where T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,{
    pub fn new(n: usize)->Self{
        let n1phi=T::from(4.0).unwrap();
        let pphi=T::from(n).unwrap().min(n1phi)/T::from(n).unwrap();
        let fw:Vec<_>=vec![0.0000, 0.4918, 0.4918, 0.0082 + 0.0082, 0.0].into_iter().map(|x|T::from(x).unwrap()).scan(T::zero(), |st, x|{
            *st=*st+x;
            Some(*st)
        }).collect();
        //println!("{:?}", fw);
        TWalkParams{
            aw: T::from(1.5).unwrap(),
            at: T::from(6.0).unwrap(),
            pphi,
            fw
        }
    }
}

pub fn no_zero<T, V>(x: &V)->bool
where T: Float + NumCast + std::cmp::PartialOrd + std::fmt::Debug,
V: Clone + FiniteLinearSpace<T> + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,
{
    for i in 0..x.dimension(){
        if x[i].abs()<=T::zero(){
            return false
        }
    }
    true
}


pub fn sqr_norm<T, V>(x: &V)->T
where T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
Standard: Distribution<T>,
V: Clone + FiniteLinearSpace<T> + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,{
    let mut result=T::zero();
    let n=x.dimension();
    for i in 0..n{
        result=result+x[i].powi(2);
    }
    result
}

pub fn sim_walk<T, U, V>(x: &V, xp: &V, rng: &mut U, param: &TWalkParams<T>)->V
where T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
Standard: Distribution<T>,
U: Rng,
V: Clone + FiniteLinearSpace<T> + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,
{
    let aw=param.aw;
    let n=x.dimension();
    let mut result=x.clone();
    let dx=x-xp;
    let two=T::one()+T::one();
    for i in 0..n{
        if rng.gen_range(T::zero(), T::one())<param.pphi{
            let u=rng.gen_range(T::zero(), T::one());
            let z=(param.aw/(T::one()+aw))*(aw*u.powi(2) + two*u - T::one());
            result[i]=x[i]+dx[i]*z;
        }
    }
    result
}


pub fn sim_beta<T,U>(rng: &mut U, param: &TWalkParams<T>)->T
where T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
Standard: Distribution<T>,
U: Rng
{
    let two=T::one()+T::one();
    let at=param.at;
    let x=rng.gen_range(T::zero(), T::one());
    if x==T::zero(){
        x
    }else if rng.gen_range(T::zero(), T::one())< (at-T::one())/(two*at) {
        x.powf(T::one()/(at+T::one()))
    }else{
        x.powf(T::one()/(T::one()-at))
    }
}

pub fn sim_tranverse<T, U, V>(x: &V, xp: &V, beta: T, rng: &mut U, param: &TWalkParams<T>)->(V, usize)
where T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
Standard: Distribution<T>,
U: Rng,
V: Clone + FiniteLinearSpace<T> + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,
{
    let n=x.dimension();
    let mut nphi=0;
    let mut result=x.clone();
    for i in 0..n{
        if rng.gen_range(T::zero(), T::one())<param.pphi{
            nphi+=1;
            result[i]=xp[i]+beta*(xp[i]-x[i]);
        }
    }
    (result, nphi)
}

pub fn sim_blow<T, U, V>(x: &V, xp: &V, rng: &mut U, param: &TWalkParams<T>)->(V, Vec<bool>)
where T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
Standard: Distribution<T>,
StandardNormal: Distribution<T>,
U: Rng,
V: Clone + FiniteLinearSpace<T> + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,{
    let n=x.dimension();
    let mut phi=vec![false;n];
    let mut sigma=T::zero();
    let dx=xp-x;
    for i in 0..n{
        if rng.gen_range(T::zero(), T::one())<param.pphi{
            phi[i]=true;
            if sigma<dx[i].abs(){
                sigma=dx[i].abs();
            }
        }
    }
    let mut result=x.clone();
    for i in 0..n{
        if phi[i]{
            result[i]=xp[i] + sigma*rng.sample(StandardNormal);
        }
    }

    (result, phi)
}

pub fn g_blow_u<T, V>(h: &V, x: &V, xp: &V, phi: &Vec<bool>)->T
where T: Float + FloatConst+ NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
Standard: Distribution<T>,
StandardNormal: Distribution<T>,
V: Clone + FiniteLinearSpace<T> + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,{
    let n=x.dimension();
    let nphi=phi.iter().filter(|&&b| b).count();
    let mut sigma=T::zero();
    let dx=xp-x;
    for i in 0..n{
        if phi[i]{
            if sigma<dx[i].abs(){
                sigma=dx[i].abs();
            }
        }
    }
    let two=T::one()+T::one();
    let ln2pi=(two*T::PI()).ln();
    let nphi=T::from(nphi).unwrap();
    if nphi>T::zero(){
        (nphi/two)*ln2pi + nphi*sigma.ln() + sqr_norm(&(h - xp))/two/sigma.powi(2)
    }else{
        T::zero()
    }
}

pub fn sim_hop<T, U, V>(x: &V, xp: &V, rng: &mut U, param: &TWalkParams<T>)->(V, Vec<bool>)
where T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
Standard: Distribution<T>,
StandardNormal: Distribution<T>,
U: Rng,
V: Clone + FiniteLinearSpace<T> + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,{
    let n=x.dimension();
    let mut phi=vec![false;n];
    let mut sigma=T::zero();
    let dx=xp-x;
    for i in 0..n{
        if rng.gen_range(T::zero(), T::one())<param.pphi{
            phi[i]=true;
            if sigma<dx[i].abs(){
                sigma=dx[i].abs();
            }
        }
    }

    sigma=sigma/T::from(3).unwrap();
    let mut result=x.clone();
    for i in 0..n{
        if phi[i]{
            result[i]=x[i] + sigma*rng.sample(StandardNormal);
        }
    }

    (result, phi)
}

pub fn g_hop_u<T, V>(h: &V, x: &V, xp: &V, phi: &Vec<bool>)->T
where T: Float + FloatConst+ NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
Standard: Distribution<T>,
StandardNormal: Distribution<T>,
V: Clone + FiniteLinearSpace<T> + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,
{
    g_blow_u(h, x, xp, phi)
}




pub fn one_move<T, U, V, F>(
    flogprob: &F,
    state: &TWalkState<T, V>,
    param: &TWalkParams<T>,
    rng: &mut U,
)->(TWalkState<T,V>, T)
where 
T: Float + FloatConst + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
Standard: Distribution<T>,
StandardNormal: Distribution<T>,
U: Rng,
V: Clone + FiniteLinearSpace<T> + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,
F: Fn(&V) -> T + ?Sized,{
    let two=T::one()+T::one();
    let kernel=TWalkKernal::random(&param.fw, rng);
    //eprintln!("{:?}", kernel);
    let (y, yp, u_prop, up_prop, a)=
    match kernel{
        TWalkKernal::Nothing=>{
            let y=state.xp.clone();
            let u_prop=state.up;
            let yp=state.x.clone();
            let up_prop=state.u;
            (y, yp, u_prop, up_prop, T::one())
        },
        TWalkKernal::Walk=>{
            if rng.gen_range(T::zero(), two)<T::one(){
                let yp=sim_walk(&state.xp, &state.x, rng, param);
                let y=state.x.clone();
                let u_prop=state.u;
                let (up_prop, a)=if no_zero(&(&yp-&y)){
                    let up_prop=flogprob(&yp);
                    (Some(up_prop),
                    (up_prop-state.up.unwrap()).exp())
                    
                }else{
                    (None,
                    T::zero())
                };
                (y, yp, u_prop, up_prop, a)
            }else{
                let y=sim_walk(&state.x, &state.xp, rng, param);
                let yp=state.xp.clone();
                let up_prop=state.up;
                let (u_prop, a)=if no_zero(&(&yp-&y)){
                    let u_prop=flogprob(&y);
                    (Some(u_prop), (u_prop-state.u.unwrap()).exp())
                }else{
                    (None, T::zero())
                };
                (y, yp, u_prop, up_prop, a)
            }
        }
        TWalkKernal::Traverse=>{
            let beta=sim_beta(rng, param);
            if rng.gen_range(T::zero(), two)<T::one(){
                let (yp, nphi)= sim_tranverse(&state.xp, &state.x, beta, rng, param);
                let y=state.x.clone();
                let u_prop=state.u;
                let up_prop=flogprob(&yp);
                let a=if nphi==0{
                    T::one()
                }else{
                    ((up_prop-state.up.unwrap())-T::from(nphi as isize-2).unwrap()*beta.ln()).exp()
                };
                (y, yp, u_prop, Some(up_prop), a)
            }else{
                let (y, nphi)=sim_tranverse(&state.x, &state.xp, beta, rng, param);
                let yp=state.xp.clone();
                let up_prop=state.up;
                let u_prop=flogprob(&y);
                let a=if nphi==0{
                    T::one()
                }else{
                    ((u_prop-state.u.unwrap())-T::from(nphi as isize-2).unwrap()*beta.ln()).exp()
                };
                (y, yp, Some(u_prop), up_prop, a)
            }
        }
        TWalkKernal::Blow=>{
            if rng.gen_range(T::zero(), two)<T::one(){
                let (yp, phi)=sim_blow(&state.xp, &state.x, rng, param);
                let y=state.x.clone();
                let u_prop=state.u;
                let (up_prop, a)=if no_zero(&(&yp-&y)){
                    let up_prop=flogprob(&yp);
                    let w1=g_blow_u(&yp, &state.xp, &state.x, &phi);
                    let w2=g_blow_u(&state.xp, &yp, &state.x, &phi);
                    let a=((up_prop-state.up.unwrap())+(w2-w1)).exp();
                    (Some(up_prop), a)
                }else{
                    (None, T::zero())
                };
                (y, yp, u_prop, up_prop, a) 
            }else{
                let (y, phi)=sim_blow(&state.x, &state.xp, rng, param);
                let yp=state.xp.clone();
                let up_prop=state.up;
                let (u_prop, a)=if no_zero(&(&y-&yp)){
                    let u_prop=flogprob(&y);
                    let w1=g_blow_u(&y, &state.x, &state.xp, &phi);
                    let w2=g_blow_u(&state.x, &y, &state.xp, &phi);
                    let a=((u_prop-state.u.unwrap())+(w2-w1)).exp();
                    (Some(u_prop), a)
                }else{
                    (None, T::zero())
                };
                (y, yp, u_prop, up_prop, a)
            }
        }
        TWalkKernal::Hop=>{
            if rng.gen_range(T::zero(), two)<T::one(){
                let (yp, phi)=sim_hop(&state.xp, &state.x, rng, param);
                let y=state.x.clone();
                let u_prop=state.u;
                let (up_prop, a)=if no_zero(&(&yp-&y)){
                    let up_prop=flogprob(&yp);
                    let w1=g_hop_u(&yp, &state.xp, &state.x, &phi);
                    let w2=g_hop_u(&state.xp, &yp, &state.x, &phi);
                    let a=((up_prop-state.up.unwrap())+(w2-w1)).exp();
                    (Some(up_prop), a)
                }else{
                    (None, T::zero())
                };
                (y, yp, u_prop, up_prop, a)
            }
            else{
                let (y, phi)=sim_hop(&state.x, &state.xp, rng, param);
                let yp=state.xp.clone();
                let up_prop=state.up;
                let (u_prop, a)=if no_zero(&(&y-&yp)){
                    let u_prop=flogprob(&y);
                    let w1=g_hop_u(&y, &state.x, &state.xp, &phi);
                    let w2=g_hop_u(&state.x, &y, &state.xp, &phi);
                    let a=((u_prop-state.u.unwrap())+(w2-w1)).exp();
                    (Some(u_prop), a)
                }else{
                    (None, T::zero())
                };
                (y, yp, u_prop, up_prop, a)
            }
        }
    };

    (TWalkState{
        x: y, 
        xp: yp, 
        u: u_prop,
        up: up_prop,
        kernel: kernel
    }, a)
}

pub fn sample_st<T, U, V, F>(
    flogprob: &F,
    state: &TWalkState<T, V>,
    param: &TWalkParams<T>,
    rng: &mut U,
)->TWalkState<T,V>
where 
T: Float + FloatConst + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Debug,
Standard: Distribution<T>,
StandardNormal: Distribution<T>,
U: Rng,
V: Clone + FiniteLinearSpace<T> + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,
F: Fn(&V) -> T + ?Sized,{
    let (next_state, a)=one_move(flogprob, state, param, rng);

    if rng.gen_range(T::zero(), T::one()) < a{
        //accept
        next_state
    }else{
        state.clone()
    }
}