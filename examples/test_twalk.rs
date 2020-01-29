#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]
#![allow(dead_code)]
//sextern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate scorus;

use num_traits::float::Float;
use quickersort::sort_by;
use rand::Rng;
use scorus::linear_space::type_wrapper::LsVec;
use scorus::mcmc::ptsample::only_sample;
use scorus::mcmc::twalk::{sample, sample_st, TWalkKernal, TWalkParams, TWalkState};
use scorus::mcmc::utils::swap_walkers;
use std::fs::File;
use std::io::Write;
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
    let nbeta = 5;
    let beta_list: Vec<_> = (0..nbeta).map(|x| 2_f64.powi(-x)).collect();
    let ndim = 2;
    let nwalkers_per_beta = 32;
    let param = TWalkParams::<f64>::new(ndim)
        .with_pphi(1.0)
        .with_fw([0.1, 0.9]);
    //println!("{:?}", param.fw);
    //std::process::exit(0);
    let mut rng = rand::thread_rng();
    //let mut rng = rand::StdRng::new().unwrap();

    //let aa=(x,y);
    //let mut x=shuffle(&x, &mut rng);
    //let lp = &rosenbrock;
    let lp = &normal_dist;
    let mut state = TWalkState::new(&LsVec(vec![0.0; ndim]), &LsVec(vec![1.0; ndim]), lp);
    //let mut walkers=vec![LsVec(vec![-1.0; ndim]), LsVec(vec![1.0; ndim])];

    let walkers: Vec<_> = (0..nwalkers_per_beta * nbeta)
        .map(|_| {
            LsVec(
                (0..ndim)
                    .map(|_| rng.gen_range(-0.1, 0.1))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let logprob: Vec<_> = walkers.iter().map(|x| lp(x)).collect();
    let mut ensemble_logprob = (walkers, logprob);
    let mut sq_sum = 0.0;
    let mut sum = 0.0;
    let thin = 1000;
    for i in 0..10000000 {
        //sample_st(&normal_dist, &mut state, &param, &mut rng);
        if i % 10 == 0 {
            swap_walkers(&mut ensemble_logprob, &mut rng, &beta_list).unwrap();
        }

        sample(lp, &mut ensemble_logprob, &param, &mut rng, &beta_list, 12);
        //let el=only_sample(lp, &ensemble_logprob, &mut rng, &beta_list, 2.0, 12).unwrap();
        //ensemble_logprob=el;
        //sample(lp, &mut ensemble_logprob, &param, &mut rng, 4);
        //sample1(&normal_dist, &mut (&mut walkers, &mut logprob), &param, &mut rng);
        sq_sum += ensemble_logprob.0[0][0].powi(2);
        sum += ensemble_logprob.0[0][0];
        let stddev = (sq_sum / (i + 1) as f64 - (sum / (i + 1) as f64).powi(2)).sqrt();

        if i % thin == 0 {
            //println!( "{:?} {:?}", ensemble_logprob.0[1][0], ensemble_logprob.0[1][1]);
            //println!("{:?}", ensemble_logprob.0[1][0]);
            println!("{}", stddev);
            //println!("{:?} {:?}", walkers[0][0], walkers[0][1]);
            //println!("{:?} {:?}", state.x[0], state.x[1]);
            //println!("{} {:?}", result.accepted, state.x);
        }
    }
}
