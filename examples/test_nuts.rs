extern crate rand;
extern crate scorus;

use rand::thread_rng;
use scorus::linear_space::type_wrapper::LsVec;
use scorus::mcmc::nuts::{nuts6, NutsState};

use scorus::linear_space::LinearSpace;
use std::fs::File;
use std::io::Write;
pub fn foo(x: &LsVec<f64, Vec<f64>>) -> (f64, LsVec<f64, Vec<f64>>) {
    let logp: f64 = -0.5 * x.0.iter().map(|x1| x1.powi(2)).sum::<f64>();
    let grad = LsVec(x.0.iter().map(|&x1| -x1).collect());
    (logp, grad)
}

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

fn rosenbrock_f(x: &LsVec<f64, Vec<f64>>) -> (f64, LsVec<f64, Vec<f64>>) {
    (rosenbrock(&x), diff_rosenbrock(x))
}

pub fn main() {
    let mut rng = thread_rng();

    let mut nutss = NutsState::<f64>::new();
    let mut x = LsVec(vec![1.0; 100]);
    let (mut lp, mut grad) = rosenbrock_f(&x);

    //nuts6(&foo, &mut x,&mut lp, &mut grad,  0.6, &mut nutss, &mut rng);

    let mut of = File::create("a.txt").unwrap();
    for i in 0..100000 {
        nuts6(
            &rosenbrock_f,
            &mut x,
            &mut lp,
            &mut grad,
            0.6,
            &mut nutss,
            i < 5000,
            &mut rng,
        );
        if i % 100 == 0 {
            println!("m={}", nutss.m);
        }
        if i >= 5000 && i % 1 == 0 {
            writeln!(&mut of, "{} {}", x[0], x[1]).unwrap();
        }
    }
}
