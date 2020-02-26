extern crate num_traits;
extern crate rand;
extern crate scorus;
use num_traits::float::Float;
use num_traits::identities::{One, Zero};
use rand::thread_rng;
use scorus::autodiff::{eval, grad, F};
use scorus::linear_space::type_wrapper::LsVec;
use scorus::linear_space::LinearSpace;
use scorus::mcmc::nuts::{nuts6, NutsState};
use std::fs::File;
use std::io::Write;

fn rosenbrock(x: &[F<f64>]) -> F<f64> {
    let mut result = F::<f64>::zero();
    for i in 0..x.len() - 1 {
        result += F::<f64>::cst(100.0) * (x[i + 1] - x[i].powi(2)).powi(2)
            + (F::<f64>::one() - x[i]).powi(2);
    }
    -result
}

fn rosenbrock_f(x: &LsVec<f64, Vec<f64>>) -> (f64, LsVec<f64, Vec<f64>>) {
    (eval(rosenbrock, &x), LsVec(grad(rosenbrock, &x)))
}

pub fn main() {
    println!("{}", eval(rosenbrock, &[1.0, 1.0]));
    println!("{:?}", grad(rosenbrock, &[0.0, 0.0]));

    let mut rng = thread_rng();

    let mut nutss = NutsState::<f64>::new();
    let mut x = LsVec(vec![1.0; 2]);
    let (mut lp, mut grad) = rosenbrock_f(&x);

    //nuts6(&foo, &mut x,&mut lp, &mut grad,  0.6, &mut nutss, &mut rng);

    let mut of = File::create("a.txt").unwrap();
    for i in 0..20000000 {
        nuts6(
            &rosenbrock_f,
            &mut x,
            &mut lp,
            &mut grad,
            0.6,
            &mut nutss,
            i < 10000000,
            &mut rng,
        );
        if i % 1000 == 0 {
            println!("m={}", nutss.m);
        }
        if i >= 10000000 && i % 10 == 0 {
            writeln!(&mut of, "{} {}", x[0], x[1]).unwrap();
        }
    }
}
