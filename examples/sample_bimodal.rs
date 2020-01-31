#![allow(unused_imports)]
#![allow(dead_code)]
//sextern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate scorus;

use std::fs::File;
use std::io::Write;

use num_traits::float::Float;
use quickersort::sort_by;
use scorus::linear_space::type_wrapper::LsVec;
use scorus::mcmc::ensemble_sample::{sample_pt, UpdateFlagSpec};
use scorus::mcmc::mcmc_errors::McmcErr;
use scorus::mcmc::utils::swap_walkers;

fn normal_dist(x: &Vec<f64>) -> f64 {
    let mut result = 0_f64;
    for i in x {
        result -= i * i;
        if i.abs() > 1.5 {
            return -std::f64::INFINITY;
        }
    }
    result
}

fn bimodal(x: &LsVec<f64, Vec<f64>>) -> f64 {
    if x[0] < -15.0 || x[0] > 15.0 || x[1] < 0.0 || x[1] > 1.0 {
        return -std::f64::INFINITY;
    }

    let (mu, sigma) = if x[1] < 0.5 { (-5.0, 0.1) } else { (5.0, 1.0) };

    -(x[0] - mu) * (x[0] - mu) / (2.0 * sigma * sigma) - sigma.ln()
}

fn foo(x: &Vec<f64>) -> f64 {
    let x1 = x[0];
    if x1 < 0.0 || x1 > 1.0 || x[1] < 0.0 || x[1] > 1.0 {
        return -std::f64::INFINITY;
    }
    match x1 {
        x if x > 0.5 => (0.1).ln(),
        _ => (0.9).ln(),
    }
}

fn main() {
    let mut ensemble: Vec<_> = vec![
        vec![0.10, 0.20],
        vec![0.20, 0.10],
        vec![0.23, 0.21],
        vec![0.03, 0.22],
        vec![0.10, 0.20],
        vec![0.20, 0.24],
        vec![0.20, 0.12],
        vec![0.23, 0.12],
        vec![0.10, 0.20],
        vec![0.20, 0.10],
        vec![0.23, 0.21],
        vec![0.03, 0.22],
        vec![0.10, 0.20],
        vec![0.20, 0.24],
        vec![0.20, 0.12],
        vec![0.23, 0.12],
        vec![0.10, 0.20],
        vec![0.20, 0.10],
        vec![0.23, 0.21],
        vec![0.03, 0.22],
        vec![0.10, 0.20],
        vec![0.20, 0.24],
        vec![0.20, 0.12],
        vec![0.23, 0.12],
        vec![0.10, 0.20],
        vec![0.20, 0.10],
        vec![0.23, 0.21],
        vec![0.03, 0.22],
        vec![0.10, 0.20],
        vec![0.20, 0.24],
        vec![0.20, 0.12],
        vec![0.23, 0.12],
    ]
    .into_iter()
    .map(|x| LsVec(x))
    .collect();
    let mut logprob: Vec<_> = ensemble.iter().map(|x| bimodal(x)).collect();
    let mut rng = rand::thread_rng();
    //let mut rng = rand::StdRng::new().unwrap();

    //let aa=(x,y);
    //let mut x=shuffle(&x, &mut rng);

    let blist = vec![1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125];
    let nbeta = blist.len();
    let nwalkers = ensemble.len() / nbeta;
    let mut results: Vec<Vec<f64>> = Vec::new();
    let niter = 100000;
    for i in 0..nbeta {
        results.push(Vec::new());
        results[i].reserve(niter);
    }

    for k in 0..niter {
        //let aaa = ff(foo, &(x, y), &mut rng, 2.0, 1);
        if k % 10 == 0 {
            swap_walkers(&mut ensemble, &mut logprob, &mut rng, &blist);
        }

        sample_pt(
            &bimodal,
            &mut ensemble,
            &mut logprob,
            &mut rng,
            2.0,
            &mut UpdateFlagSpec::Prob(0.5),
            &blist,
            4,
        );
        for i in 0..nbeta {
            results[i].push(ensemble[i * nwalkers + 0][0]);
        }
    }

    for i in 0..nbeta {
        sort_by(&mut results[i], &|x, y| {
            if x > y {
                std::cmp::Ordering::Greater
            } else if x < y {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        });
    }

    let mut file = File::create("bimod_data.txt").unwrap();

    for j in 0..results[0].len() {
        for i in 0..nbeta {
            write!(file, "{} ", results[i][j]);
        }
        writeln!(file, "{}", j);
    }
}
