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

fn main() {
    let mut rng = thread_rng();
    let ndim = 10;
    let nchains = 5;
    let mut x = Vec::new();

    for _i in 0..nchains {
        let p: Vec<_> = (0..ndim).map(|_| rng.gen_range(-1.0e-3, 1.0e-3)).collect();
        x.push(LsVec(p));
    }

    let mut ensemble_lp = init_chain(x, &foo, 4);

    let thin = 10;
    for i in 0..1_000_000 {
        sample(
            &mut ensemble_lp,
            &foo,
            4,
            0.5,
            0.1,
            1e-12,
            &mut rng,
            &Some(Box::new(move |a, b| {
                if i % 10 == 0 {
                    1.0
                } else {
                    2.38 / ((a * b) as f64).sqrt()
                }
            })),
            &None,
            4,
        );
        if i % thin == 0 {
            println!("{} {}", ensemble_lp.0[0][0], ensemble_lp.0[0][1]);
        }
    }
}
