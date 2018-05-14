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
use scorus::sph_tessellation::Tessellation;
fn main() {
    let mut tes=Tessellation::<f64>::octahedron();
    for i in 0..6 {
        tes.refine();
    }
    println!("o rust_logo");
    for v in &tes.vertices{
        let angle=v.angle_between(Vec3d::new(0.0, 0.0, 1.0));
        let v1=v.normalized()*angle.cos().powi(2);
        //println!("v {} {} {}", v1.x, v1.y, v1.z);
        //let v1=v.normalized();
        println!("v {} {} {}", v1.x, v1.y, v1.z);
    }

    for f in &tes.faces{
        //println!("f {}//{} {}//{} {}//{}", f[0]+1, f[0]+1, f[1]+1, f[1]+1, f[2]+1,f[2]+1);
        println!("f {} {} {}", f[0]+1, f[1]+1, f[2]+1);
    }
}
