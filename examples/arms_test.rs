extern crate kxjs;
extern crate num_traits;
extern crate rand;
use kxjs::mcmc::arms::sample as arms;
use rand::thread_rng;
use num_traits::float::Float;

fn main() {
    let mut rng = thread_rng();
    let mut nchange = 0;
    let mut xcur = 1.0;

    for i in 0..1000 {
        xcur = arms(
            &|x: f64| (-x * x / (2.0 * 0.1 * 0.1)),
            (-100.0, 100.0),
            &vec![-1.0, 0.0, 1.0],
            1.0,
            10,
            &mut rng,
            &mut nchange,
        ).unwrap();
    }
}
