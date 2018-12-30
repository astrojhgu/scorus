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
use scorus::healpix::utils::{nest2ring, ring2nest};
use scorus::integration::adaptive_trapezoid::integrate;
use scorus::interpolation::linear1d::interp;
use scorus::interpolation::sph_nn::Interpolator;
use scorus::interpolation::spline;
use scorus::opt::opt_errors::OptErr;
use scorus::opt::powell::fmin;
use scorus::opt::pso::*;
use scorus::opt::tolerance::Tolerance;
use scorus::polynomial::legendre;
use scorus::rand_vec::uniform_on_sphere::rand as rand_sph;
use scorus::sph_tessellation::Tessellation;
use scorus::utils::regulate;
use scorus::utils::types::{HasElement, HasLen, InitFromLen};

fn foo(x: f64) -> f64 {
    (x.powi(2) * 20.0).sin()
}

fn foo2(x: f64) -> f64 {
    (x / 0.01).sin()
}

fn main() {
    println!(
        "{}",
        integrate(&|x: f64| x.sin(), 1e-12, &[0.0, 1.0, 2.0, f64::PI()])
    );
}
