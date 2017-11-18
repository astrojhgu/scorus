//sextern crate std;
extern crate num_traits;
extern crate rand;
extern crate rsmcmc;
use rsmcmc::ensemble_sample::sample as ff;
use rsmcmc::ptsample::sample as ff1;
use rsmcmc::shuffle;

use rand::Rng;
fn logprob(x: &Vec<f64>) -> f64 {
    let mut result = 0_f64;
    for i in x {
        result -= i * i;
        if i.abs() > 1.5 {
            return -1.0 / 0.0;
        }
    }
    result
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
    ];
    let mut y = vec![0.0];
    //let mut rng = rand::thread_rng();
    let mut rng = rand::StdRng::new().unwrap();

    //let aa=(x,y);
    //let mut x=shuffle(&x, &mut rng);
    let blist = vec![1.0, 0.5];

    for k in 0..10000 {
        let aaa = ff1(logprob, &(x, y), &mut rng, &blist, k % 10 == 0, 2.0, 2);
        //let aaa=ff1(|x|{-x[0]*x[0]-x[1]*x[1]}, &(x,y), &mut rng, &blist, k%10==0, 2.0, 2);
        x = aaa.0;
        y = aaa.1;
        for i in 2..3 {
            println!("{} {}", x[i][0], x[i][1]);
        }
    }
}
