extern crate rand;
extern crate scorus;
extern crate tinymt;

use rand::SeedableRng;
use scorus::mcmc::nuts::{find_reasonable_epsilon, leapfrog, nuts6, NutsState};
use tinymt::TinyMT64;

use scorus::linear_space::type_wrapper::LsVec;
use std::fs::File;
use std::io::Write;
pub fn foo(x: &LsVec<f64, Vec<f64>>) -> (f64, LsVec<f64, Vec<f64>>) {
    let logp: f64 = -0.5 * x.0.iter().map(|x1| x1.powi(2)).sum::<f64>();
    let grad = LsVec(x.0.iter().map(|&x1| -x1).collect());
    (logp, grad)
}

pub fn main() {
    let mut rng = TinyMT64::from_seed(1234.into());

    let mut nutss = NutsState::<f64>::new();
    let mut x = LsVec(vec![1.0; 100]);
    let mut r0 = LsVec(vec![1.0; 100]);
    let (mut lp, mut grad) = foo(&x);

    //nuts6(&foo, &mut x,&mut lp, &mut grad,  0.6, &mut nutss, &mut rng);

    let mut of = File::create("a.txt").unwrap();
    for i in 0..10000 {
        nuts6(
            &foo,
            &mut x,
            &mut lp,
            &mut grad,
            0.6,
            &mut nutss,
            i < 10,
            &mut rng,
        );
        println!("m={}", nutss.m);
        if i >= 10 {
            writeln!(&mut of, "{} {}", x[0], x[1]);
        }
    }
}
