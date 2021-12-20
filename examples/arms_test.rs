#![allow(unused_imports)]
extern crate scorus;
use num::traits::float::Float;
use rand::thread_rng;
use scorus::mcmc::arms::sample as arms;

fn main() {
    let mut rng = thread_rng();
    let mut nchange = 0;
    let mut xcur = 1.0;

    for _i in 0..1000 {
        xcur = arms(
            &|x: f64| (-x * x / (2.0 * 0.1 * 0.1)),
            (-100.0, 100.0),
            &vec![-1.0, 0.0, 1.0],
            xcur,
            10,
            &mut rng,
            &mut nchange,
        )
        .unwrap();
        println!("{}", xcur);
    }
}
