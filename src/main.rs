#![allow(unused_imports)]

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

use scorus::coordinates::{SphCoord, Vec3d};

use num_traits::float::{Float, FloatConst};
use scorus::healpix::utils::{nest2ring, ring2nest};
use scorus::interpolation::linear1d::interp;
use scorus::interpolation::sph_nn::Interpolator;
use scorus::interpolation::spline;
use scorus::opt::opt_errors::OptErr;
use scorus::opt::powell::fmin;
use scorus::opt::tolerance::Tolerance;
use scorus::rand_vec::uniform_on_sphere::rand as rand_sph;
use scorus::sph_tessellation::Tessellation;
use scorus::utils::regulate;
use scorus::polynomial::legendre;
use std::io::Write;
fn fobj(x: &Vec<f64>) -> f64 {
    x.windows(2).fold(0.0, |a, b| {
        a + 100.0 * (b[1] - b[0].powi(2)).powi(2) + (1.0 - b[0]).powi(2)
    })
}

fn main() {
    println!("{:?}", legendre::legendre2poly(&vec![0., 0.0, 0.0, 0.0,0.0,1.0]));
}
