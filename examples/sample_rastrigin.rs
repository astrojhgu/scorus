#![allow(unused_imports)]
#![allow(dead_code)]
//sextern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate scorus;
use num_traits::float::Float;
use num_traits::FloatConst;
use quickersort::sort_by;
use rand::Rng;
use scorus::linear_space::type_wrapper::LsVec;
use scorus::mcmc::ensemble_sample::{sample_pt, UpdateFlagSpec};
use scorus::mcmc::mcmc_errors::McmcErr;
use scorus::mcmc::utils::swap_walkers;
use std::fs::File;
use std::io::Write;

fn rastrigin(x: &LsVec<f64, Vec<f64>>) -> f64 {
    let n = x.len();
    let a = 10.0;
    let mut result = -(n as f64 * a);
    for i in x.0.iter() {
        result += a * (2.0 * f64::PI() * i).cos() - i.powi(2);
    }
    result
}

fn main() {
    let nbetas = 8;
    let blist: Vec<_> = (0..nbetas).map(|i| 2_f64.powi(-i)).collect();
    let nwalkers_per_beta = 4;
    let ndims = 20;
    let mut rng = rand::thread_rng();
    let mut ensemble: Vec<_> = (0..nwalkers_per_beta * nbetas)
        .map(|_| {
            let a: Vec<_> = (0..ndims).map(|_| rng.gen_range(-0.01, 0.01)).collect();
            LsVec(a)
        })
        .collect();

    let mut logprob: Vec<_> = ensemble.iter().map(|x| rastrigin(x)).collect();

    //let mut rng = rand::StdRng::new().unwrap();

    //let aa=(x,y);
    //let mut x=shuffle(&x, &mut rng);

    for k in 0..300000 {
        //let aaa = ff(foo, &(x, y), &mut rng, 2.0, 1);
        if k % 10 == 0 {
            swap_walkers(&mut ensemble, &mut logprob, &mut rng, &blist);
        }

        sample_pt(
            &rastrigin,
            &mut ensemble,
            &mut logprob,
            &mut rng,
            2.0,
            &mut UpdateFlagSpec::Prob(0.2),
            &blist,
            4,
        );
        if k % 100 == 0 {
            for i in 0..ensemble[0].len() {
                print!("{} ", ensemble[0][i]);
            }
            println!()
        }
    }
}
