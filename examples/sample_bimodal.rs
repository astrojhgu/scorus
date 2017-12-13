//sextern crate std;
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

fn main() {
    let mut x = vec![
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
    let mut y = vec![0.0];
    //let mut rng = rand::thread_rng();
    let mut rng = rand::StdRng::new().unwrap();

    //let aa=(x,y);
    //let mut x=shuffle(&x, &mut rng);

    let blist = vec![1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125];
    let nbeta = blist.len();
    let nwalkers = x.len() / nbeta;
    let mut results: Vec<Vec<f64>> = Vec::new();
    let niter = 100000;
    for i in 0..nbeta {
        results.push(Vec::new());
        results[i].reserve(niter);
    }

    for k in 0..niter {
        //let aaa = ff(foo, &(x, y), &mut rng, 2.0, 1);
        let aaa = ff1(&bimodal, (x, y), &mut rng, &blist, k % 10 == 0, 2.0, 1).unwrap();
        //let aaa = ff1(foo, &(x, y), &mut rng, &blist,k%10==0, 2.0, 1);
        //let aaa=ff1(|x|{-x[0]*x[0]-x[1]*x[1]}, &(x,y), &mut rng, &blist, k%10==0, 2.0, 2);
        x = aaa.0;
        y = aaa.1;

        for i in 0..nbeta {
            results[i].push(x[i * nwalkers + 0][0]);
        }
    }

    for i in 0..nbeta {
        sort_by(&mut results[i], &|x, y| {
            if x > y {
                std::cmp::Ordering::Greater
            } else if x < y {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        });
        for j in 0..niter {
            println!("{} {}", results[i][j], j);
        }
        println!("no no no");
    }
}
