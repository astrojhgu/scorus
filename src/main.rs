#![allow(unused_imports)]
#![allow(dead_code)]
//extern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate scorus;

use num_traits::Bounded;
use rand::thread_rng;
use std::slice;

use scorus::map_proj::mollweide::{iproj, proj};
use std::fs::File;
use std::io::Write;

use scorus::coordinates::{SphCoord, Vec3d};
use scorus::polynomial::bernstein::bernstein_base;
use scorus::polynomial::bernstein::bernstein_poly;

use num_traits::float::{Float, FloatConst};
use rand::Rng;
use scorus::healpix::utils::{nest2ring, ring2nest};
use scorus::integration::adaptive_trapezoid::integrate;
use scorus::interpolation::linear1d::interp;
use scorus::interpolation::sph_nn::Interpolator;
use scorus::interpolation::spline;
use scorus::linear_space::type_wrapper::LsVec;
use scorus::mcmc::dream::dream::{init_chain, sample};
use scorus::mcmc::dream::utils::multinomial;
use scorus::opt::opt_errors::OptErr;
use scorus::opt::powell::fmin;
use scorus::opt::pso::*;
use scorus::opt::tolerance::Tolerance;
use scorus::polynomial::legendre;
use scorus::rand_vec::uniform_on_sphere::rand as rand_sph;
use scorus::sph_tessellation::Tessellation;
use scorus::utils::regulate;
use scorus::utils::types::{HasElement, HasLen, InitFromLen};

fn foo(x: &LsVec<f64, Vec<f64>>) -> f64 {
    -x.0.iter().map(|&x| x.powi(2) / 2.0).sum::<f64>()
}

fn foo2(x: f64) -> f64 {
    (x / 0.01).sin()
}

fn rosenbrock(x: &LsVec<f64, Vec<f64>>)->f64{
    let mut result=0.0;
    for i in 0..x.0.len()-1{
        result+=100.0*(x[i+1]-x[i].powi(2)).powi(2)+(1.0-x[i]).powi(2);
    }
    -result
}

fn main() {
    let mut rng = thread_rng();
    let ndim = 60;
    let nchains = 10;
    //let beta_list=vec![1.0, 0.5, 0.25, 0.125, 0.0625];
    let beta_list:Vec<_>=(0..20).map(|x|2.0.powi(-x)).collect();
    //let beta_list=vec![1.0, 0.5, 0.25];

    let nbeta=beta_list.len();
    let mut x = Vec::new();

    for _i in 0..nchains*nbeta {
        let p: Vec<_> = (0..ndim).map(|_| 1.0+rng.gen_range(-1.0e-3, 1.0e-3)).collect();
        x.push(LsVec(p));
    }

    let mut ensemble_lp = init_chain(x, &foo, 4);
    //println!("{:?}", ensemble_lp);
    let thin = 10;
    for i in 0..100_000 {
        sample(
            &mut ensemble_lp,
            &rosenbrock,
            4,
            0.1,
            0.01,
            1e-12,
            &mut rng,
            &beta_list,
            &Some(Box::new(move |a, b| {
                if i % 10 == 0 {
                    1.0
                } else {
                    2.38 / ((a * b) as f64).sqrt()
                }
            })),
            &None,
            4,
            i%10==0,
        );
        if i % thin == 0 {
            println!("{} {}", ensemble_lp.0[0][0], ensemble_lp.0[0][1]);
        }
    }
}
