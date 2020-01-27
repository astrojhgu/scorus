#![allow(unused_imports)]
#![allow(dead_code)]
//sextern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate scorus;

use std::fs::File;
use std::io::Write;
use rand::Rng;
use num_traits::float::Float;
use quickersort::sort_by;
use scorus::linear_space::type_wrapper::LsVec;
use scorus::mcmc::twalk::{sample_st,sample, TWalkKernal, TWalkParams, TWalkState};
fn normal_dist(x: &LsVec<f64, Vec<f64>>) -> f64 {
    let mut result = 0_f64;
    for &i in &x.0 {
        result -= i * i / 2.0;
    }
    result
}

fn rosenbrock(x: &LsVec<f64, Vec<f64>>) -> f64 {
    let mut result = 0.0;
    for i in 0..x.0.len() - 1 {
        result += 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2);
    }
    -result
}

fn main() {
    let ndim = 100;
    let nwalkers=32;
    let param = TWalkParams::<f64>::new(ndim).with_pphi(0.01);
    //println!("{:?}", param.fw);
    //std::process::exit(0);
    let mut rng = rand::thread_rng();
    //let mut rng = rand::StdRng::new().unwrap();

    //let aa=(x,y);
    //let mut x=shuffle(&x, &mut rng);
    let mut state = TWalkState::new(
        &LsVec(vec![0.0; ndim]),
        &LsVec(vec![1.0; ndim]),
        &normal_dist,
    );
    //let mut walkers=vec![LsVec(vec![-1.0; ndim]), LsVec(vec![1.0; ndim])];

    let mut walkers:Vec<_>=(0..nwalkers).map(|_| LsVec((0..ndim).map(|_|{
        rng.gen_range(-1.0, 1.0)
    }).collect::<Vec<_>>())
    ).collect();
    let mut logprob:Vec<_>=walkers.iter().map(|x| normal_dist(x)).collect();
    let mut ensemble_logprob=(walkers, logprob);

    let thin = 100;
    for i in 0..1000000 {
        //sample_st(&normal_dist, &mut state, &param, &mut rng);
        sample(&normal_dist, &mut ensemble_logprob, &param, &mut rng, 4);
        //sample1(&normal_dist, &mut (&mut walkers, &mut logprob), &param, &mut rng);
        if i % thin == 0 {
            println!("{:?} {:?}", ensemble_logprob.0[0][0], ensemble_logprob.0[0][1]);
            //println!("{:?} {:?}", walkers[0][0], walkers[0][1]);
            //println!("{:?} {:?}", state.x[0], state.x[1]);
            //println!("{} {:?}", result.accepted, state.x);
        }
    }
    
}
