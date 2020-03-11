extern crate scorus;
use scorus::linear_space::type_wrapper::LsVec;
use scorus::mcmc::hmc::utils::leapfrog;

fn h_q(q: &LsVec<f64, Vec<f64>>) -> LsVec<f64, Vec<f64>> {
    let x = q[0];
    let y = q[1];
    let r = (x.powi(2) + y.powi(2)).sqrt();
    LsVec(vec![r.powi(-3) * x, r.powi(-3) * y])
}

fn h_p(p: &LsVec<f64, Vec<f64>>) -> LsVec<f64, Vec<f64>> {
    let vx = p[0];
    let vy = p[1];
    LsVec(vec![vx, vy])
}

pub fn main() {
    let mut q = LsVec(vec![1.0, 0.0]);
    let mut p = LsVec(vec![0.0, 1.4]);
    let mut last_hq = h_q(&q);

    for i in 0..150000 {
        leapfrog(&mut q, &mut p, &mut last_hq, 0.01, &h_q, &h_p);
        println!("{} {}", q[0], q[1]);
    }
}
