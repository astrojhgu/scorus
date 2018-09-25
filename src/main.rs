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
use scorus::opt::pso::*;
use scorus::opt::tolerance::Tolerance;
use scorus::polynomial::legendre;
use scorus::rand_vec::uniform_on_sphere::rand as rand_sph;
use scorus::sph_tessellation::Tessellation;
use scorus::utils::regulate;
use scorus::utils::types::{HasElement, HasLen, InitFromLen};
use std::io::Write;

fn main() {
    let func = |x: &Vec<f64>| x.iter().map(|&x| -x * x).fold(0.0, |a, b| a + b);

    let mut rng = thread_rng();

    let mut po =
        ParticleSwarmMaximizer::new(&func, vec![-1.0, -1.0], vec![1.0, 1.0], None, 25, &mut rng);

    while !po.converged(0.8, 1e-15, 1e-15) {
        po.sample(&mut rng, 1.193, 1.193);
    }

    println!("{:?}", po.gbest);
}
