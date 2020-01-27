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
use scorus::mcmc::twalk::{sample_st, TWalkKernal, TWalkParams, TWalkState};
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

    let thin = 10000;
    let mut kernel_cnt = vec![0; 4];
    let mut accept_cnt = vec![0; 4];
    for i in 0..100000000 {
        let result = sample_st(&normal_dist, &mut state, &param, &mut rng);
        kernel_cnt[result.last_kernel.to_usize()] += 1;
        if result.accepted {
            accept_cnt[result.last_kernel.to_usize()] += 1;
        }
        if i % thin == 0 {
            println!("{:?} {:?}", state.x[0], state.x[1]);
            //println!("{} {:?}", result.accepted, state.x);
        }
    }
    eprintln!("{:?}", kernel_cnt);
    eprintln!("{:?}", accept_cnt);
    let accept_rate: Vec<_> = accept_cnt
        .iter()
        .zip(kernel_cnt.iter())
        .map(|(&a, &c)| if c == 0 { 0.0 } else { a as f64 / c as f64 })
        .collect();
    eprintln!("{:?}", accept_rate);
}
