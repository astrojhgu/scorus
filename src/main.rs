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
use scorus::mcmc::dream::utils::multinomial;
use scorus::mcmc::dream::dream::DreamState;
use scorus::linear_space::type_wrapper::LsVec;
use rand::Rng;

fn foo(x: &LsVec<f64, Vec<f64>>) -> f64 {
    -x.0.iter().map(|&x|{x.powi(2)/2.0}).sum::<f64>()
}

fn foo2(x: f64) -> f64 {
    (x / 0.01).sin()
}

fn main() {
    let mut rng=thread_rng();
    let ndim=1000;
    
    let mut x=Vec::new();

    for i in 0..16{
        let p:Vec<_>=(0..ndim).map(|_|{
            rng.gen_range(-1.0e-3, 1.0e-3)
        }).collect();
        x.push(LsVec(p));
    }

    let mut dream=DreamState::new(x, &foo, 0.001, 0.001);
    
    let thin=100;
    for i in 0..1000000{
        for j in 0..dream.nchains(){
            dream.evolve_single_chain(j, &foo, 4, 1., &mut rng);
        }
        if i%thin==0{
            //println!("{:?}", dream.x[0]);
            println!("{} {}", dream.x[0][0], dream.x[0][1]);
        }
    }
    
}
