//extern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate rsmcmc;

use num_traits::float::Float;
use rsmcmc::ensemble_sample::sample as ff;
use rsmcmc::ptsample::sample as ff1;
use rsmcmc::shuffle;
use rand::Rng;
use quickersort::sort_by;
use rsmcmc::arms::sample;
use rsmcmc::arms::inv_int_exp_y;
use rsmcmc::arms::int_exp_y;
use rsmcmc::arms::init;
use rsmcmc::arms::dump_section_list;
use rsmcmc::arms::insert_point;
use rsmcmc::arms::update_scale;

fn normal_dist(x: &Vec<f64>) -> f64 {
    let mut result = 0_f64;
    for i in x {
        result -= i * i;
        if i.abs() > 1.5 {
            return -std::f64::INFINITY;
        }
    }
    result
}

fn bimodal(x: &Vec<f64>) -> f64 {
    if x[0] < -15.0 || x[0] > 15.0 || x[1] < 0.0 || x[1] > 1.0 {
        return -std::f64::INFINITY;
    }

    let (mu, sigma) = if x[1] < 0.5 { (-5.0, 0.1) } else { (5.0, 1.0) };

    -(x[0] - mu) * (x[0] - mu) / (2.0 * sigma * sigma) - sigma.ln()
}

fn foo(x: &Vec<f64>) -> f64 {
    let x1 = x[0];
    if x1 < 0.0 || x1 > 1.0 || x[1] < 0.0 || x[1] > 1.0 {
        return -std::f64::INFINITY;
    }
    match x1 {
        x if x > 0.5 => (0.1).ln(),
        _ => (0.9).ln(),
    }
}

fn unigauss(x: f64) -> f64 {
    -x * x
}

fn bigauss(x: f64) -> f64 {
    if x < 0.0 {
        -(x + 1.) * (x + 1.)
    } else {
        -(x - 1.) * (x - 1.)
    }
}

fn main() {
    let mut rng = rand::thread_rng();
    let mut cnt = 0;
    let mut x = 0.0;
    /*
    println!(
        "{}",
        int_exp_y(
            1.0906301300564414,
            (1.090630130056439, -1.1894740805869248),
            (1.0906301300564414, -1.1894740805870607)
        ).unwrap()
    );
    println!(
        "{}",
        int_exp_y(1.5, (1.0906301300564414, -1.1894740805870607), (1.5, -1.25)).unwrap()
    );
    let s = rsmcmc::arms::Section {
        _x_l: 1.090630130056439,
        _x_u: 1.5,
        _x_i: Some(1.0906301300564414),
        _y_l: -1.1894740805869248,
        _y_u: -1.25,
        _y_i: Some(-1.1894740805870607),
        _int_exp_y_l: Some(0.06136897988341142),
        _int_exp_y_u: Some(0.05953959385346115),
        _cum_int_exp_y_l: Some(1.7060441118071998),
        _cum_int_exp_y_u: Some(1.765583705660661),
    };

    println!("{:?}", s.calc_int_exp_y().unwrap());
*/
    /*
    let mut scale=0.0;
    let xx=init(&unigauss, (-10.0, 10.0), &vec![-5.0, -1.0, 1.0, 5.0], &mut scale).unwrap();
    let xx=insert_point(&unigauss, xx, 3.0, scale).unwrap();
    //let xx=insert_point(&unigauss, xx, 0.0, scale).unwrap();
    let xx= update_scale(xx, &mut scale).unwrap();
    dump_section_list(&xx, Some(&unigauss), 0.1, scale);
    eprintln!("{}", scale);
    */

    for i in 0..100000 {
        x = sample(
            &bigauss,
            (-125.0, 125.0),
            &vec![-10.0, -1.5, -1.0, 1.0, 1.5, 20.0],
            x,
            10,
            &mut rng,
            &mut cnt,
        ).unwrap();
        println!("{}", x);
    }
}
