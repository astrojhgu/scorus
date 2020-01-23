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
        result -= i * i/2.0;
    }
    result
}

fn main() {
    let ndim = 1000;
    let param = TWalkParams::<f64>::new(ndim).with_pphi(0.001);
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

    let thin=10000;
    for i in 0..10000000 {
        let result=sample_st(&normal_dist, &mut state, &param, &mut rng);
        if i%thin==0{
            println!("{:?} {:?}", state.x[0], state.x[1]);
            //println!("{} {:?}", result.accepted, state.x);
        }

    }
}
