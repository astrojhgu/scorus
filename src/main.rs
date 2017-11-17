//sextern crate std;
extern crate rand;
extern crate rsmcmc;
use rsmcmc::ensemble_sample::sample as ff;


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
    ];
    let mut y = vec![0.0];
    //let mut rng = rand::thread_rng();
    let mut rng = rand::StdRng::new().unwrap();

    //let aa=(x,y);

    //    foo(&x, 1);
    for k in 0..10000000 {
        let aaa = ff(logprob, &(x, y), &mut rng, 2.0, 1);
        x = aaa.0;
        y = aaa.1;
        if k % 100 == 0 {
            //if false{
            for i in 0..1 {
                for j in 0..2 {
                    print!("{} ", x[i][j]);
                }
                println!("");
            }
        }
    }
}
