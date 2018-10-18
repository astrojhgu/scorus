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

fn foo(x:f64)->f64{
    (x.powi(2)*20.0).sin()
}

fn foo2(x:f64)->f64{
    (x/0.01).sin()
}

fn main() {
    let n=100000;
    let p:Vec<_>=(0..n+1).map(|i|{
        foo2(i as f64/n as f64)
    }).collect();
    let x: Vec<_> = (0..1000).map(|x| x as f64 / 1000.0).collect();

    let y=bernstein_poly(&x, &p);
    for i in 0..x.len(){
        println!("{} {} {}", x[i], y[i], foo2(x[i]))
    }
}
