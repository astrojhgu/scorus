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
//use scorus::mcmc::mcmc_errors::McmcErr;
//use scorus::mcmc::ptsample::{create_sampler, create_sampler_st};
use scorus::mcmc::ensemble_sample::{sample_pt, UpdateFlagSpec};
use scorus::mcmc::utils::swap_walkers;
use std::fs::File;
use std::io::Write;
fn normal_dist(x: &LsVec<f64, Vec<f64>>) -> f64 {
    let mut result = 0_f64;
    for i in &x.0 {
        result -= i * i;
    }
    result
}

fn main() {
    let logprob = &normal_dist;
    let mut rng = rand::thread_rng();
    let ndim = 10;
    let blist: Vec<_> = (0..4).map(|i| 2_f64.powi(-i)).collect();
    let nbeta = blist.len();

    let mut ensemble = Vec::new();
    for i in 0..2 * ndim * nbeta {
        ensemble.push(LsVec(
            (0..ndim)
                .map(|_| rng.gen_range(-1.0, 1.0))
                .collect::<Vec<f64>>(),
        ));
    }

    let mut lp: Vec<_> = ensemble.iter().map(|x| logprob(x)).collect();

    let nwalkers_per_beta = ensemble.len() / nbeta;
    let mut results: Vec<Vec<f64>> = Vec::new();
    let niter = 10000;
    for i in 0..nbeta {
        results.push(Vec::new());
        results[i].reserve(niter);
    }

    //let mut sampler = create_sampler_st(bimodal, xy, rng, blist, 2.0);
    //let mut sampler = create_sampler(normal_dist, xy, rng, blist, 2.0, 4);

    let mut i = 0;
    let mut update_flag_func = move || -> Vec<bool> {
        i += 1;
        i = i % ndim;
        let result: Vec<bool> = (0..ndim).map(|j| j == i).collect();
        result
    };

    for k in 0..niter {
        //let aaa = ff(foo, &(x, y), &mut rng, 2.0, 1);

        if k % 10 == 0 {
            swap_walkers(&mut ensemble, &mut lp, &mut rng, &blist);
        }

        sample_pt(
            &logprob,
            &mut ensemble,
            &mut lp,
            &mut rng,
            2.0,
            //&mut UpdateFlagSpec::Pphi(0.01),
            &mut UpdateFlagSpec::Func(&mut update_flag_func),
            &blist,
            1,
        );
        for i in 0..nbeta {
            results[i].push(ensemble[i * nwalkers_per_beta + 0][0]);
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
            write!(file, "{} ", results[i][j]).unwrap();
        }
        writeln!(file, "{}", j).unwrap();
    }
}
