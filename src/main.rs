#![allow(unused_imports)]

//extern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate scorus;

use std::slice;
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
use scorus::opt::powell::fmin;
use scorus::opt::opt_errors::OptErr;
use scorus::opt::tolerance::Tolerance;
fn fobj(x:&Vec<f64>)->f64{
    x.windows(2).fold(0.0, |a, b|{
        a+100.0*(b[1]-b[0].powi(2)).powi(2)+(1.0-b[0]).powi(2)
    })
}

fn main() {
    let ftol=Tolerance::Rel(1e-30);
    let mut result=vec![0.0, 0.0, 0.0];
    let result=
        {
            for i in 0..100 {
                result=match fmin(&fobj, &result, ftol, 2000){
                    (x, OptErr::MaxIterExceeded)=>{
                        println!("a");
                        x
                    },
                    (x, _) => x
                }
            }
            fmin(&fobj, &result, ftol, 2000).0
        };

    println!("{:?}", result);
}
