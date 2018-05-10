#![allow(unused_imports)]

//extern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate scorus;

use rand::thread_rng;
use num_traits::Bounded;

use std::fs::File;
use scorus::map_proj::mollweide::{iproj, proj};

use scorus::coordinates::{SphCoord, Vec3d};

use scorus::interpolation::sph_nn::Interpolator;
use num_traits::float::{Float, FloatConst};
use std::io::Write;
use scorus::rand_vec::uniform_on_sphere::rand as rand_sph;
use scorus::interpolation::spline;
use scorus::interpolation::linear1d::interp;
use scorus::healpix::utils::{nest2ring, ring2nest};
use scorus::utils::regulate;
fn main() {
    let mut x=-20.0;
    while x<20.0{
        println!("{} {}", x, regulate(x, -1., 4.0));
        x+=0.1;
    }
}
