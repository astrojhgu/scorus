#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::type_complexity)]
#![allow(clippy::mutex_atomic)]

use rayon::scope;
use std;

use num_traits::float::Float;
use num_traits::identities::one;
use num_traits::NumCast;
use rand::distributions::uniform::SampleUniform;
use rand::distributions::Distribution;
use rand::distributions::Standard;
use rand_distr::Exp1;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::StandardNormal;
use std::marker::{Send, Sync};

use std::ops::{Add, Mul, Sub};
use std::sync::Mutex;

use crate::linear_space::InnerProdSpace;
use std::fs::File;
use std::io::Write;
macro_rules!  dump_rand{
    ($rng:ident.gen_range(T::zero(), T::one())) => {{
        let x=$rng.gen_range(T::zero(), T::one());
        let mut f=File::with_options().append(true).create(true).open("rand_dump.txt").unwrap();
        writeln!(&mut f , "{:?} uniform {}", x, line!()).unwrap();
        x
    }
    };

    ($rng:ident.sample($d: ident)) =>{{
        let x=$rng.sample($d);
        let mut f=File::with_options().append(true).create(true).open("rand_dump.txt").unwrap();
        writeln!(&mut f, "{:?} {} {}", x, stringify!($d), line!()).unwrap();
        x
    }
    }
}


pub struct NutsState<T>
where T:Float
{
    pub m: usize, 
    pub Hbar: T,
    pub epsilon: T, 
    pub epsilon_bar: T, 
    pub mu: T,
}

impl<T> NutsState<T>
where T:Float{
    pub fn new()->NutsState<T>{
        NutsState::<T>{
            m: 0,
            Hbar: T::zero(),
            epsilon: T::zero(), 
            epsilon_bar: T::zero(),
            mu: T::zero(),
        }
    }
}

pub fn leapfrog<T, V, F>(theta: &V, r: &V, grad: &V, epsilon: T, fg: &F) -> (V, V, V, T)
where 
T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Debug,
Standard: Distribution<T>,
V: Clone + InnerProdSpace<T> + Sync + Send + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,
F: Fn(&V) ->(T, V) + Send + Sync,{
    let two=T::one()+T::one();
    let half=T::one()/two;
    let rprime=r + &(grad * (epsilon*half));

    let thetaprime = theta + &(&rprime*epsilon);

    let (logpprime, gradprime) = fg(&thetaprime);
    let rprime = &rprime + &(&gradprime *( epsilon *half));
    (thetaprime, rprime, gradprime, logpprime)
}

pub fn any_inf<T, V>(x: &V)->bool
where T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Debug,
V: Clone + InnerProdSpace<T> + Sync + Send + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,
{
    for i in 0..x.dimension(){
        if !x[i].is_normal(){
            return true;
        }
    }
    false
}

pub fn normal_random_like<T, V, U>(x0: &V, rng: &mut U)->V
where T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Debug,
Standard: Distribution<T>,
StandardNormal: Distribution<T>,
V: Clone + InnerProdSpace<T> + Sync + Send + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,
U: Rng,{
    let mut y=x0-x0;
    for i in 0..y.dimension(){
        y[i]=dump_rand!(rng.sample(StandardNormal));
    }
    y
}

pub fn find_reasonable_epsilon<T,V,F, U>(theta0: &V, grad0: &V, logp0: T, fg: &F, rng: &mut U)->T
where T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Debug,
Standard: Distribution<T>,
StandardNormal: Distribution<T>,
V: Clone + InnerProdSpace<T> + Sync + Send + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,
F: Fn(&V) ->(T, V) + Send + Sync,
U: Rng,
{
    let two=T::one()+T::one();
    let half=T::one()/two;
    let mut epsilon = T::one();
    
    let r0=normal_random_like(theta0, rng);
    let (_, mut rprime, mut gradprime, mut logpprime) = leapfrog(theta0, &r0, grad0, epsilon, fg);
    let mut k = T::one();
    while !logpprime.is_normal() || any_inf(&gradprime){
        k=k/two;
        let a=leapfrog(theta0, &r0, &grad0, epsilon * k, fg);
        rprime=a.1;
        gradprime=a.2;
        logpprime=a.3;
    }

    epsilon = epsilon * k/two;

    let mut logacceptprob = logpprime-logp0-(rprime.dot(&rprime)-r0.dot(&r0))*half;

    let a=if logacceptprob > half.ln() {
        T::one()
    }else{
        -T::one()
    };
    
    while a * logacceptprob > -a * two.ln(){
        epsilon = epsilon * two.powf(a);
        let a=leapfrog(theta0, &r0, &grad0, epsilon, fg);
        rprime=a.1;
        logpprime=a.3;
        
        logacceptprob = logpprime-logp0-(rprime.dot(&rprime)-r0.dot(&r0))*half;
    }

    epsilon
}

pub fn stop_criterion<T, V>(thetaminus: &V, thetaplus: &V, rminus: &V, rplus: &V)->bool
where T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Debug,
V: Clone + InnerProdSpace<T> + Sync + Send + Sized,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,
{
    let dtheta=thetaplus-thetaminus;
    dtheta.dot(rminus) >=T::zero() && dtheta.dot(rplus) >=T::zero()
}

pub fn build_tree<T, V, F, U>(theta: &V, r: &V, grad: &V, logu: T, v: isize, j: usize, epsilon: T, f: &F, joint0: T, rng: &mut U)->(V, V, V, V, V, V, V, V, T, usize, usize, T, usize)
where 
T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Debug,
Standard: Distribution<T>,
V: Clone + InnerProdSpace<T> + Sync + Send + Sized + Clone + std::fmt::Debug,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,
F: Fn(&V) ->(T, V) + Send + Sync,
U: Rng,
{
    let two=T::one()+T::one();
    let half=T::one()/two;
    let (mut thetaminus, mut rminus, mut gradminus, mut thetaplus, mut rplus, mut gradplus, mut thetaprime, mut gradprime, mut logpprime, mut nprime, mut sprime, mut alphaprime, mut nalphaprime);
    if j == 0{       
        let a= leapfrog(theta, r, grad, T::from(v).unwrap() * epsilon, f);
        thetaprime=a.0;
        let rprime=a.1;
        gradprime=a.2;
        logpprime=a.3;
        
        let joint = logpprime - rprime.dot(&rprime)*half;
        
        nprime = if logu < joint {1}else {0};
        sprime = if (logu - T::from(1000).unwrap()) < joint {1} else {0};
        thetaminus = thetaprime.clone();
        thetaplus = thetaprime.clone();
        rminus = rprime.clone();
        rplus = rprime.clone();
        gradminus = gradprime.clone();
        gradplus = gradprime.clone();
        alphaprime = T::min(T::one(), T::exp(joint - joint0));
        nalphaprime = 1;
    }
    else{       
        let a= build_tree(theta, r, grad, logu, v, j - 1, epsilon, f, joint0, rng);
        thetaminus=a.0;rminus=a.1;gradminus=a.2; thetaplus=a.3;rplus=a.4;gradplus=a.5;thetaprime=a.6;gradprime=a.7;logpprime=a.8;nprime=a.9;sprime=a.10;alphaprime=a.11;nalphaprime=a.12;

        let thetaprime2;
        let gradprime2;
        let logpprime2;
        let nprime2;
        let sprime2;
        let alphaprime2;
        let nalphaprime2;
        if sprime == 1{
            if v == -1 {
                let a = build_tree(&thetaminus, &rminus, &gradminus, logu, v, j - 1, epsilon, f, joint0, rng);
                thetaminus=a.0;
                rminus=a.1;
                gradminus=a.2;
                thetaprime2=a.6;
                gradprime2=a.7;
                logpprime2=a.8;
                nprime2=a.9;
                sprime2=a.10;
                alphaprime2=a.11;
                nalphaprime2=a.12;
            }
            else{
                let a= build_tree(&thetaplus, &rplus, &gradplus, logu, v, j - 1, epsilon, f, joint0, rng);
                thetaplus=a.3;
                rplus=a.4;
                gradplus=a.5;
                thetaprime2=a.6;
                gradprime2=a.7;
                logpprime2=a.8;
                nprime2=a.9;
                sprime2=a.10;
                alphaprime2=a.11;
                nalphaprime2=a.12;
            }
                
            if dump_rand!(rng.gen_range(T::zero(), T::one()))<T::from(nprime2).unwrap()/T::max(T::from(nprime+nprime2).unwrap(), T::one()){
                thetaprime = thetaprime2.clone();
                gradprime = gradprime2.clone();
                logpprime = logpprime2
            }
            
            nprime = nprime+nprime2;
            sprime = if sprime>0 && sprime2>0 && stop_criterion(&thetaminus, &thetaplus, &rminus, &rplus){1}else {0};
            alphaprime = alphaprime + alphaprime2;
            nalphaprime = nalphaprime + nalphaprime2;
            //eprintln!("{:?} {:?}", thetaminus, thetaplus);
        }
        
    }
    (thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alphaprime, nalphaprime)
}

pub fn nuts6<T, V, F, U>(f: &F, theta0: &mut V, logp0: &mut T, grad0: &mut V, delta: T, nutss: &mut NutsState<T>, burning: bool, rng: &mut U)
where
T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Debug,
Standard: Distribution<T>,
StandardNormal: Distribution<T>,
Exp1: Distribution<T>,
V: Clone + InnerProdSpace<T> + Sync + Send + Sized + Clone + std::fmt::Debug,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,
F: Fn(&V) ->(T, V) + Send + Sync,
U: Rng,
{
    let two=T::one()+T::one();
    let half=T::one()/two;
    let D=theta0.dimension();
    //let (mut logp0, mut grad0)=f(theta0);
    let gamma=T::from(0.05).unwrap();
    let t0=10;
    let kappa=T::from(0.75).unwrap();
    
    
    if nutss.m==0{
        nutss.epsilon_bar=T::one();
        nutss.Hbar=T::zero();
        nutss.epsilon=find_reasonable_epsilon(theta0, &grad0, *logp0, f, rng);
        
        nutss.mu=(T::from(10).unwrap()*nutss.epsilon).ln();
        nutss.m=1;
    }
    
    let M=1000;
    let Madapt=10;
    
    while nutss.m<M+Madapt{
        
        let r0=normal_random_like(theta0, rng);
        println!("{}, r0={:?}",nutss.m, r0);
        let joint=*logp0-r0.dot(&r0)*half;
        let logu=joint - dump_rand!(rng.sample(Exp1));

        let mut thetaminus=theta0.clone();
        let mut thetaplus=theta0.clone();
        let mut rminus=r0.clone();
        let mut rplus=r0.clone();
        let mut gradminus=grad0.clone();
        let mut gradplus=grad0.clone();

        let mut j=0;
        let mut n=1;
        let mut s=1;
        let (mut thetaprime, mut gradprime,mut logpprime,mut nprime,mut sprime, mut alpha,mut nalpha);
        alpha=T::zero();
        nalpha=0;
        while s==1{
            let v=if dump_rand!(rng.gen_range(T::zero(), T::one())) < half {1} else{-1};
            if v==-1{
                let a=build_tree(&thetaminus, &rminus, &gradminus, logu, v, j, nutss.epsilon, f, joint, rng);
                thetaminus=a.0;
                rminus=a.1;
                gradminus=a.2;
                thetaprime=a.6;
                gradprime=a.7;
                logpprime=a.8;
                nprime=a.9;
                sprime=a.10;
                alpha=a.11;
                nalpha=a.12;
            }else{
                let a=build_tree(&thetaplus, &rplus, &gradplus, logu, v, j, nutss.epsilon, f, joint, rng);
                thetaplus=a.3;
                rplus=a.4;
                gradplus=a.5;
                thetaprime=a.6;
                gradprime=a.7;
                logpprime=a.8;
                nprime=a.9;
                sprime=a.10;
                alpha=a.11;
                nalpha=a.12;
            }

            let tmp=T::one().min(T::from(nprime).unwrap()/T::from(n).unwrap());
            //eprintln!("{}", nprime);
            if sprime==1 && dump_rand!(rng.gen_range(T::zero(), T::one()))<tmp{
                *theta0=thetaprime.clone();
                *logp0=logpprime;
                *grad0=gradprime.clone();
            }
            
            n+=nprime;
            s=if sprime >0 && stop_criterion(&thetaminus, &thetaplus, &rminus, &rplus) {1} else {0};
            j+=1;
        }

        let mut eta=T::one()/T::from(nutss.m+t0).unwrap();
        nutss.Hbar=(T::one()-eta)*nutss.Hbar+eta*(delta-alpha/T::from(nalpha).unwrap());
        if nutss.m<=Madapt{
            nutss.epsilon=(nutss.mu - T::sqrt(T::from(nutss.m).unwrap()) / gamma * nutss.Hbar).exp();
            eta=T::from(nutss.m).unwrap().powf(-kappa);
            nutss.epsilon_bar=T::exp((T::one() - eta) * T::ln(nutss.epsilon_bar) + eta * T::ln(nutss.epsilon));
        }else{
            nutss.epsilon=nutss.epsilon_bar;
        }
        nutss.m+=1;
    }
    //println!("{}",samples.len());
    
}
/*
pub fn nuts6<T, V, F, U>(f: &F, theta0: &mut V, logp: &mut T, grad0: &mut V,  delta: T, rng: &mut U, nutss: &mut NutsState<T>, m_adapt: usize)
where
T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + Sync + Send + std::fmt::Debug,
Standard: Distribution<T>,
StandardNormal: Distribution<T>,
Exp1: Distribution<T>,
V: Clone + InnerProdSpace<T> + Sync + Send + Sized + Clone + std::fmt::Debug,
for<'b> &'b V: Add<Output = V>,
for<'b> &'b V: Sub<Output = V>,
for<'b> &'b V: Mul<T, Output = V>,
F: Fn(&V) ->(T, V) + Send + Sync,
U: Rng,
{
    let gamma=T::from(0.05).unwrap();
    let t0=10;
    let kappa=T::from(0.75).unwrap();
    let mu=(T::from(10).unwrap()*nutss.epsilon).ln();

    let two=T::one()+T::one();
    let half=T::one()/two;

    let r0=normal_random_like(theta0, rng);
    let joint = *logp-r0.dot(&r0)*half;
    let logu=joint-dump_rand!(rng.sample(Exp1));
    let mut thetaminus=theta0.clone();
    let mut thetaplus=theta0.clone();
    let mut rminus=r0.clone();
    let mut rplus=r0.clone();
    let mut gradminus=grad0.clone();
    let mut gradplus=grad0.clone();
    let (mut thetaprime, mut gradprime,mut logpprime,mut nprime,mut sprime, mut alpha,mut nalpha);
    let mut j=0;
    let mut n=1;

    if nutss.m==0{
        nutss.epsilon=find_reasonable_epsilon(theta0, grad0, *logp, f, rng);
    }
    loop {
        let v = if dump_rand!(rng.gen_range(T::zero(), T::one()))<half {1}else {-1};
        if v==-1{
            let a=build_tree(&thetaminus, &rminus, &gradminus, logu, v, j, nutss.epsilon, f, joint, rng);
            thetaminus=a.0;
            rminus=a.1;
            gradminus=a.2;
            thetaprime=a.6;
            gradprime=a.7;
            logpprime=a.8;
            nprime=a.9;
            sprime=a.10;
            alpha=a.11;
            nalpha=a.12;
        }else{
            let a=build_tree(&thetaplus, &rplus, &gradplus, logu, v, j, nutss.epsilon, f, joint, rng);
            thetaplus=a.3;
            rplus=a.4;
            gradplus=a.5;
            thetaprime=a.6;
            gradprime=a.7;
            logpprime=a.8;
            nprime=a.9;
            sprime=a.10;
            alpha=a.11;
            nalpha=a.12;
        }

        let tmp=T::one().min(T::from(nprime).unwrap()/T::from(n).unwrap());
        eprintln!("{}", nprime);
        if sprime==1 && dump_rand!(rng.gen_range(T::zero(), T::one()))<tmp{
            *theta0=thetaprime.clone();
            *logp=logpprime;
            *grad0=gradprime.clone();
        }
        n+=nprime;
        j+=1;
        eprintln!("sprime={:?} {} {:?} {:?}", sprime, stop_criterion(&thetaminus, &thetaplus, &rminus, &rplus), thetaminus, thetaplus);
        let s=sprime >0 && stop_criterion(&thetaminus, &thetaplus, &rminus, &rplus);
        if !s {
            break;
        }
    }

    let mut eta=T::one()/T::from(nutss.m+t0).unwrap();
    nutss.Hbar=(T::one()-eta)*nutss.Hbar+eta*(delta-alpha/T::from(nalpha).unwrap());
    if nutss.m<m_adapt{
        nutss.epsilon=(mu-T::from(nutss.m).unwrap().sqrt()/gamma*nutss.Hbar).exp();
        eta=T::from(nutss.m).unwrap().powf(-kappa);
        nutss.epsilon_bar=T::exp((T::one() - eta) * T::ln(nutss.epsilon_bar) + eta * T::ln(nutss.epsilon));
    }
    else{
        nutss.epsilon=nutss.epsilon_bar
    }
    nutss.m+=1;
}*/

