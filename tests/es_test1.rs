extern crate num_traits;
extern crate rand;
extern crate rsmcmc;

use num_traits::float::Float;
use rsmcmc::get_one_init_realization;
use rsmcmc::ensemble_sample::sample;

const NITER: u32 = 100000;
const NWALKERS: u32 = 16;
const P: f64 = 0.2;

fn target_dist(x: &Vec<f64>) -> f64 {
    let x = x[0];
    if x < 0.0 || x > 1.0 {
        -std::f64::INFINITY
    } else {
        if x < 0.5 {
            P.ln()
        } else {
            (1.0 - P).ln()
        }
    }
}


#[test]
fn test() {
    let mut ensemble = Vec::new();
    let mut rng = rand::thread_rng();
    let mut cached_logprob = Vec::new();
    for i in 0..NWALKERS {
        ensemble.push(get_one_init_realization(&vec![0.0], &vec![1.0], &mut rng));
    }

    let mut result = Vec::new();
    for i in 0..NITER {
        let aaa = sample(&target_dist, (ensemble, cached_logprob), &mut rng, 2.0, 1).unwrap();
        ensemble = aaa.0;
        cached_logprob = aaa.1;
        result.push(ensemble[0][0]);
    }
    let mut cnt = 0;
    for x in &result {
        if *x < 0.5 {
            cnt += 1;
        }
    }

    let stddev = ((NITER as f64) * (1.0 - P) * P).sqrt();
    let expected = (NITER as f64) * P;
    let average = (cnt as f64);
    if (average - expected).abs() > 3.0 * stddev {
        panic!(format!(
            "Deviate too much: {} vs {} with stddev={}",
            expected,
            average,
            stddev
        ))
    }
}
