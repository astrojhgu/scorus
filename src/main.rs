//extern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate scorus;

use num_traits::float::Float;
use scorus::mcmc::ensemble_sample::sample as ff;
use scorus::mcmc::ptsample::sample as ff1;
use scorus::mcmc::shuffle;
use rand::Rng;
use quickersort::sort_by;
use scorus::mcmc::arms::sample;
use scorus::mcmc::arms::inv_int_exp_y;
use scorus::mcmc::arms::int_exp_y;
use scorus::mcmc::arms::init;
use scorus::mcmc::arms::dump_section_list;
use scorus::mcmc::arms::insert_point;
use scorus::mcmc::arms::update_scale;
use scorus::polynomial::legendre::eval as legendre;
use scorus::opt::linmin::linmin;
use scorus::opt::powell::fmin;

fn main() {
    let mut x = -1.0;
    while x < 1.0 {
        print!("{}", x);
        for i in 0..6 {
            print!(" {}", legendre(i, x));
        }
        println!();
        x += 0.01;
    }
    x = 1.0;
    print!("{}", x);
    for i in 0..6 {
        print!(" {}", legendre(i, x));
    }
    println!();
}
