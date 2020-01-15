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
use scorus::mcmc::twalk::{TWalkKernal, TWalkParams, TWalkState, sample_st};
fn normal_dist(x: &LsVec<f64, Vec<f64>>) -> f64 {
    let mut result = 0_f64;
    for &i in &x.0 {
        result -= i * i;
    }
    result
}


fn main() {
    let ndim=10;
    let param=TWalkParams::<f64>::new(ndim);
    //std::process::exit(0);
    let mut rng = rand::thread_rng();
    //let mut rng = rand::StdRng::new().unwrap();

    //let aa=(x,y);
    //let mut x=shuffle(&x, &mut rng);
    let mut state=TWalkState::new(&LsVec(vec![0.0;ndim]), &LsVec(vec![0.1;ndim]), &normal_dist);

    for i in 0..10000{
        state=sample_st(&normal_dist, &state, &param, &mut rng);
        println!("{:?} {:?}", state.x[0], state.x[1]);    
    }
}
