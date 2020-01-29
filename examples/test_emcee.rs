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

fn main() {
    let mut rng = rand::thread_rng();
    let ndim = 200;
    let niter=3000000;
    let mut ensemble = Vec::new();
    for i in 0..2 * ndim {
        ensemble.push(LsVec(
            (0..ndim)
                .map(|_| rng.gen_range(-1.0, 1.0))
                .collect::<Vec<f64>>(),
        ));
    }

    let mut logprob:Vec<_>=ensemble.iter().map(|x|{normal_dist(x)}).collect();

    let mut ensemble_logprob=(ensemble, logprob);
    for k in 0..niter {
        //let aaa = ff(foo, &(x, y), &mut rng, 2.0, 1);
        sample(&normal_dist, &mut ensemble_logprob, &mut rng, 2.0, 0.1, 4);
        if k%100==0{
            println!("{} {}", ensemble_logprob.0[0][0], ensemble_logprob.0[0][1]);
        }

    }
}
