//extern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate rsmcmc;
use num_traits::float::Float;
use rsmcmc::ensemble_sample::sample as ff;
use rsmcmc::ptsample::sample as ff1;
use rsmcmc::shuffle;
use rand::Rng;
use quickersort::sort_by;
use rsmcmc::arms::sample;


fn normal_dist(x: &Vec<f64>) -> f64 {
    let mut result = 0_f64;
    for i in x {
        result -= i * i;
        if i.abs() > 1.5 {
            return -std::f64::INFINITY;
        }
    }
    result
}

fn bimodal(x: &Vec<f64>) -> f64 {
    if x[0] < -15.0 || x[0] > 15.0 || x[1] < 0.0 || x[1] > 1.0 {
        return -std::f64::INFINITY;
    }

    let (mu, sigma) = if x[1] < 0.5 { (-5.0, 0.1) } else { (5.0, 1.0) };

    -(x[0] - mu) * (x[0] - mu) / (2.0 * sigma * sigma) - sigma.ln()
}

fn foo(x: &Vec<f64>) -> f64 {
    let x1 = x[0];
    if x1 < 0.0 || x1 > 1.0 || x[1] < 0.0 || x[1] > 1.0 {
        return -std::f64::INFINITY;
    }
    match x1 {
        x if x > 0.5 => (0.1).ln(),
        _ => (0.9).ln(),
    }
}

fn unigauss(x: f64) -> f64 {
    -x * x
}

fn main() {
    let mut rng = rand::thread_rng();
    let mut cnt = 0;


    match sample(
        &unigauss,
        (-5.0, 5.0),
        &vec![-1.5, -1.0, 1.0, 1.5],
        0.0,
        10,
        &mut rng,
        &mut cnt,
    ) {
        Ok(x) => println!("{}", x),
        Err(x) => println!("{:?}", x),
    }
}
