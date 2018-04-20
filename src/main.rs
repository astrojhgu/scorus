//extern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate scorus;

use rand::thread_rng;
use num_traits::Bounded;
use std::fs::File;
use scorus::map_proj::hammer::{iproj, proj};
use scorus::coordinates::{SphCoord, Vec3d};

use scorus::interpolation::sph_nn::Interpolator;
use num_traits::float::{Float, FloatConst};
use std::io::Write;
use scorus::rand_vec::uniform_on_sphere::rand as rand_sph;
use scorus::interpolation::spline;
use scorus::interpolation::linear1d::interp;
fn main() {
    let xlist = [5., 4., 3., 2., 1., 0.];
    let ylist = [1., 2., 1., 2., 1., 2.];
    let mut x = -1.0;
    while x < 7.0 {
        println!("{} {}", x, interp(x, &xlist, &ylist));
        x += 0.01;
    }
}
