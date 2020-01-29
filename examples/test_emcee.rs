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
use scorus::mcmc::mcmc_errors::McmcErr;
use scorus::mcmc::ensemble_sample::sample;
use std::fs::File;
use std::io::Write;
fn normal_dist(x: &LsVec<f64, Vec<f64>>) -> f64 {
    let mut result = 0_f64;
    for i in &x.0 {
        result -= i * i/2.0;
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
    let mut rng = rand::thread_rng();
    let ndim = 200;
    let niter=3000000;
    let mut ensemble = Vec::new();
    for i in 0..4 {
        ensemble.push(LsVec(
            (0..ndim)
                .map(|_| rng.gen_range(0.9, 1.1))
                .collect::<Vec<f64>>(),
        ));
    }

    let lpf=&normal_dist;
    let mut logprob:Vec<_>=ensemble.iter().map(|x|{lpf(x)}).collect();

    for k in 0..niter {
        //let aaa = ff(foo, &(x, y), &mut rng, 2.0, 1);
        sample(lpf, &mut ensemble, &mut logprob, &mut rng, 2.0, 0.1, 12);
        if k%10==0{
            println!("{} {}", ensemble[0][0], ensemble[0][1]);
        }

    }
}
