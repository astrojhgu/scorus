extern crate rand;
extern crate scorus;

use rand::thread_rng;
use scorus::linear_space::type_wrapper::LsVec;
use scorus::linear_space::LinearSpace;
use scorus::mcmc::hmc::naive::{sample, HmcParam};
use std::fs::File;
use std::io::Write;

fn rosenbrock(x: &[f64]) -> f64 {
    let mut result = 0.0;
    for i in 0..x.len() - 1 {
        result += 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2);
    }
    -result
}

fn delta(i: usize, j: usize) -> usize {
    if i == j {
        1
    } else {
        0
    }
}

fn diff_rosenbrock(x: &LsVec<f64, Vec<f64>>) -> LsVec<f64, Vec<f64>> {
    let mut result = LsVec(vec![0.0; x.dimension()]);
    for j in 0..x.dimension() {
        for i in 0..x.dimension() - 1 {
            result[j] -= 200.0
                * (x[i + 1] - x[i].powi(2))
                * (delta(j, i + 1) as f64 - 2.0 * x[i] * delta(i, j) as f64)
                + 2.0 * (x[i] - 1.0) * delta(i, j) as f64;
        }
    }
    result
}

fn rosenbrock_f(x: &LsVec<f64, Vec<f64>>) -> f64 {
    rosenbrock(&x)
}

pub fn main() {
    let mut rng = thread_rng();

    let mut x = LsVec(vec![1.0; 2]);
    let mut lp = rosenbrock_f(&x);
    let mut last_grad = diff_rosenbrock(&x);
    //nuts6(&foo, &mut x,&mut lp, &mut grad,  0.6, &mut nutss, &mut rng);

    let mut of = File::create("a.txt").unwrap();
    let mut accept_cnt = 0;
    let mut epsilon = 0.005;

    let param = HmcParam::quick_adj(0.7);
    for i in 0..10000000 {
        sample(
            &rosenbrock_f,
            &diff_rosenbrock,
            &mut x,
            &mut lp,
            &mut last_grad,
            &mut rng,
            &mut epsilon,
            2,
            &param,
        );
    }
    let param = HmcParam::slow_adj(0.7);
    for i in 0..100000000 {
        if sample(
            &rosenbrock_f,
            &diff_rosenbrock,
            &mut x,
            &mut lp,
            &mut last_grad,
            &mut rng,
            &mut epsilon,
            2,
            &param,
        ) {
            accept_cnt += 1;
        }
        if i % 100000 == 0 {
            println!("{} {}", accept_cnt as f64 / (i + 1) as f64, epsilon);
        }
        if i % 100 == 0 {
            writeln!(&mut of, "{} {}", x[0], x[1]).unwrap();
        }
    }
}
