extern crate num_traits;
extern crate rand;
extern crate rsmcmc;

use rand::Rng;
use rand::thread_rng;
use rsmcmc::ensemble_sample::sample;
use num_traits::float::Float;
use num_traits::identities::zero;
use num_traits::identities::one;

use rsmcmc::functions::{logdbin, phi};

fn main() {
    let E: Vec<f64> = vec![
        2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 70.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0,
        62.0, 66.0, 74.0, 78.0, 82.0, 86.0, 90.0, 94.0, 98.0, 34.0,
    ];

    let nrec = vec![
        23.0, 71.0, 115.0, 159.0, 200.0, 221.0, 291.0, 244.0, 44.0, 221.0, 210.0, 182.0, 136.0,
        119.0, 79.0, 81.0, 61.0, 41.0, 32.0, 32.0, 31.0, 22.0, 18.0, 11.0, 277.0,
    ];

    let ninj = vec![
        96.0, 239.0, 295.0, 327.0, 345.0, 316.0, 349.0, 281.0, 45.0, 235.0, 217.0, 185.0, 140.0,
        121.0, 79.0, 81.0, 61.0, 41.0, 32.0, 32.0, 31.0, 22.0, 18.0, 11.0, 298.0,
    ];

    let logprob = |x: &Vec<f64>| {
        let energy = &E;
        let nrec = &nrec;
        let ninj = &ninj;

        let a = x[0];
        let b = x[1];
        let mu = x[2];
        let sigma = x[3];

        if a < zero() || a > one() || b < zero() || b > one() || mu < zero() || mu > 100.0
            || sigma < 1e-6 || sigma > 100.0
        {
            return f64::neg_infinity();
        }

        let mut logp = 0.0;
        for ((e, r), i) in energy.iter().zip(nrec.iter()).zip(ninj.iter()) {
            let eff = a + (b - a) * phi((*e - mu) / sigma);
            //println!("{} {} {} {} ", *e, *r, *i, eff);
            logp += logdbin(*r, eff, *i);
        }
        logp
    };

    //logprob(&vec![0.1, 0.9, 17.0, 12.0]);

    let mut rng = thread_rng();
    let mut ensemble = Vec::new();

    for _ in 0..16 {
        ensemble.push(rsmcmc::init_ensemble::get_one_init_realization(
            &vec![0.0, 0.9, 15.0, 10.0],
            &vec![0.1, 1.0, 17.0, 13.0],
            &mut rng,
        ));
    }

    //println!("{:?}", ensemble);

    let mut lp = Vec::<f64>::new();
    for i in 0..30000 {
        let aaa = sample(&logprob, (ensemble, lp), &mut rng, 2.0, 4).unwrap();
        ensemble = aaa.0;
        lp = aaa.1;
        if i > 1000 {
            let n = rng.gen_range(0, ensemble.len());
            println!(
                "{} {} {} {}",
                ensemble[n][0], ensemble[n][1], ensemble[n][2], ensemble[n][3]
            );
        }
    }
}
