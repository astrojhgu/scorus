//extern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate scorus;

use num_traits::float::Float;
use scorus::mcmc::ensemble_sample::sample as ff;
use scorus::mcmc::ptsample::sample as ff1;
use scorus::mcmc::sampler_closure::esample_closure;
//use scorus::mcmc::shuffle;
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
use scorus::polynomial::legendre::ln_factorial;
use scorus::polynomial::legendre::legendre2poly;

fn main() {
    let mut xx = {
        let x = vec![
            vec![0.10, 0.20],
            vec![0.20, 0.10],
            vec![0.23, 0.21],
            vec![0.03, 0.22],
            vec![0.10, 0.20],
            vec![0.20, 0.24],
            vec![0.20, 0.12],
            vec![0.23, 0.12],
            vec![0.10, 0.20],
            vec![0.20, 0.10],
            vec![0.23, 0.21],
            vec![0.03, 0.22],
            vec![0.10, 0.20],
            vec![0.20, 0.24],
            vec![0.20, 0.12],
            vec![0.23, 0.12],
            vec![0.10, 0.20],
            vec![0.20, 0.10],
            vec![0.23, 0.21],
            vec![0.03, 0.22],
            vec![0.10, 0.20],
            vec![0.20, 0.24],
            vec![0.20, 0.12],
            vec![0.23, 0.12],
            vec![0.10, 0.20],
            vec![0.20, 0.10],
            vec![0.23, 0.21],
            vec![0.03, 0.22],
            vec![0.10, 0.20],
            vec![0.20, 0.24],
            vec![0.20, 0.12],
            vec![0.23, 0.12],
        ];
        let y = vec![0.0];

        esample_closure(
            |x: &Vec<f64>| -x[0] * x[0] - x[1] * x[1],
            (x, y),
            rand::thread_rng(),
            2.0,
            1,
        )
    };

    for i in 0..10000 {
        let a = xx(true).unwrap().unwrap();
        println!("{} {}", a.0[0][0], a.0[0][1]);
    }
}
