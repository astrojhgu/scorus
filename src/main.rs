//sextern crate std;
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
    }
    result
}


fn main() {
    let mut x = vec![
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 4.0],
        vec![4.3, 3.2],
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 4.0],
        vec![4.3, 3.2]
    ];
    let mut y = vec![0.0];
    //let mut rng = rand::thread_rng();
    let mut rng = rand::StdRng::new().unwrap();

    //let aa=(x,y);
    //let mut x=shuffle(&x, &mut rng);
    let blist=vec![1.0, 0.5];

    for k in 0..10000{
        let aaa=ff1(logprob, &(x,y), &mut rng, &blist, k%10==0 , 2.0, 2);
        x=aaa.0;
        y=aaa.1;
        for i in 0..1{
            println!("{} {}", x[i][0], x[i][1]);
        }
    }
}
