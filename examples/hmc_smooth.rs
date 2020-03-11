extern crate rand;
extern crate rand_distr;
extern crate scorus;

use rand::thread_rng;
use rand::Rng;
use rand_distr::StandardNormal;
use scorus::linear_space::traits::IndexableLinearSpace;
use scorus::linear_space::type_wrapper::LsVec;
use scorus::mcmc::hmc::naive::{sample, HmcParam};
use std::fs::File;
use std::io::Write;

const smooth_param: f64 = 0.1;

fn logprob(ym: &[f64], x: &[f64], y: &[f64], sigma: &[f64]) -> f64 {
    let p1 = x
        .iter()
        .zip(y.iter().zip(sigma.iter().zip(ym.iter())))
        .map(|(&x, (&y, (&sigma, &ym)))| -(ym - y).powi(2) / (2.0 * sigma.powi(2)))
        .sum::<f64>();
    let n = x.len();

    let y2 = ym
        .windows(3)
        .map(|ym| {
            let y1 = ym[0];
            let y2 = ym[1];
            let y3 = ym[2];

            (y3 + y1 - 2.0 * y2).powi(2)
        })
        .sum::<f64>();
    let p2 = -y2 / 2.0 / smooth_param;

    p1 + p2
    //let p2=
}

fn delta(i: usize, j: usize) -> f64 {
    if i == j {
        1.0
    } else {
        0.0
    }
}

fn grad_logprob(ym: &[f64], x: &[f64], y: &[f64], sigma: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; x.len()];
    for q in 0..x.len() {
        for i in 0..x.len() {
            result[q] += (y[i] - ym[i]) * delta(i, q) / (2.0 * sigma[i].powi(2));
        }
        for i in 1..x.len() - 1 {
            result[q] += -(ym[i + 1] + ym[i - 1] - 2.0 * ym[i])
                * (delta(i + 1, q) + delta(i - 1, q) - 2.0 * delta(i, q))
                / smooth_param;
        }
    }
    result
}

fn main() {
    let mut rng = thread_rng();
    let x: Vec<_> = (0..100).map(|x| x as f64).collect();
    let y: Vec<_> = x
        .iter()
        .map(|&x| -> f64 {
            let x1: f64 = rng.sample(StandardNormal);
            x1 * 0.1 + (x / 10.0).sin()
        })
        .collect();

    let mut ym = vec![0.0; x.len()];

    let sigma: Vec<_> = vec![1.0; x.len()];

    eprintln!("{}", logprob(&ym, &x, &y, &sigma));
    eprintln!("{:?}", grad_logprob(&ym, &x, &y, &sigma));

    let lp_f = |p: &LsVec<f64, Vec<f64>>| logprob(&p, &x, &y, &sigma);
    let glp_f = |p: &LsVec<f64, Vec<f64>>| LsVec(grad_logprob(&p, &x, &y, &sigma));

    let mut q = LsVec(ym);
    let mut lp = lp_f(&q);
    let mut g = glp_f(&q);
    let param = HmcParam::quick_adj(0.6);

    let mut epsilon = 0.001;

    let mut accept_cnt = 0;

    let mut average_ym = LsVec(vec![0.0; x.len()]);
    let mut cnt = 0;
    for i in 0..1000000 {
        let accepted = sample(
            &lp_f,
            &glp_f,
            &mut q,
            &mut lp,
            &mut g,
            &mut rng,
            &mut epsilon,
            15,
            &param,
        );

        if accepted {
            accept_cnt += 1;
        }
        if i > 10 {
            average_ym = &average_ym + &q;
            cnt += 1;
        }

        if i % 1000 == 0 || i < 1000 {
            eprintln!(
                "{} {} {} {}",
                i,
                accept_cnt as f64 / (i + 1) as f64,
                epsilon,
                lp
            );
        }

        if i % 100 == 0 && i > 1000 {
            //println!("{} {}", q[20], q[21]);
            let mut of = File::create("a.txt").unwrap();
            for i in 0..x.len() {
                writeln!(&mut of, "{} {} {}", x[i], y[i], average_ym[i] / cnt as f64);
            }
        }
    }
}
